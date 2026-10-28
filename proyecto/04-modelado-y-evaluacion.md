# Radar de Licitaciones — Modelado y evaluación
> Documento técnico · Proyecto final PLN, UFM
> Estado: diseño propuesto · Actualizado: octubre 2026

## 1. Enfoque

Dos señales independientes combinadas en un score. Cada una responde una pregunta distinta.

| Señal | Pregunta | Técnica | Tema del syllabus |
|---|---|---|---|
| 1 | ¿Este concurso es para nosotros? | TF-IDF + regresión logística | Clasificación de texto; TF-IDF |
| 2 | ¿Se parece a algo que ya hicimos? | Embeddings de oraciones + similitud coseno | Semántica distribucional; modelos preentrenados |

```
score = α · P(relevante) + (1 − α) · similitud_máxima
```

- α se ajusta en el conjunto de validación maximizando precisión@10.
- Antes del score se aplican filtros de negocio: vigencia (obligatorio), modalidad, monto y entidad (configurables).
- Ambas señales están en [0, 1]. Si la distribución de similitudes queda comprimida, normalizarla con min-max sobre el histórico antes de combinar. [Decidir en evaluación]

### Por qué no un LLM en el ranking

| Enfoque | Entiende el texto | Etiquetas | Explicable | Rol |
|---|---|---|---|---|
| Filtro por categoría o palabra clave | No | No requiere | Sí | Línea base |
| TF-IDF + regresión logística | Por vocabulario | Sí (≈1,000) | Sí: peso por palabra | Modelo principal |
| Embeddings preentrenados | Por significado | No requiere | Parcial: proyecto similar | Segunda señal |
| LLM zero-shot vía API | Sí | No requiere | Texto libre, poco auditable | Pre-etiquetado y línea base |

El costo de un LLM sobre ~1,000 títulos al día sería bajo; las razones para no usarlo en el ranking son otras: respuestas que cambian entre versiones del modelo, ausencia de una probabilidad calibrada para ordenar, dependencia externa en el ciclo diario y que no mejora con la retroalimentación del usuario. El curso además evalúa una técnica de PLN aplicada, no una llamada a una API.

## 2. Señal 1: clasificador

Configuración inicial (ajustar con validación):

```python
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizador = FeatureUnion([
    ("palabras", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("caracteres", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
])

clasificador = Pipeline([
    ("tfidf", vectorizador),
    ("rl", LogisticRegression(class_weight="balanced", max_iter=2000)),
])
```

- Entrada: `procesos.texto_tfidf`.
- Los n-gramas de caracteres compensan textos cortos, abreviaturas y errores de escritura.
- Hiperparámetros a ajustar: `C`, `min_df`, rangos de n-gramas.
- Explicación por concurso: los 5 términos con mayor contribución positiva (`coeficiente × valor TF-IDF` del documento). Requiere mapear los índices de la `FeatureUnion` a sus nombres.
- Se guarda con `joblib` en `modelos/clasificador.joblib` junto con una cadena de versión.

## 3. Señal 2: similitud con el portafolio

- Modelo: `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers), en CPU.
- Entrada: `procesos.texto_embedding`.
- Embeddings normalizados; similitud coseno = producto punto.
- `similitud_maxima` = máximo sobre los proyectos del portafolio; `proyecto_similar` = el que lo alcanza.
- Los embeddings del portafolio se calculan una vez y se cachean; se recalculan solo si cambia el CSV.

### Portafolio (`portafolio/proyectos_albran.csv`)

Columnas: `proyecto`, `descripcion`, `linea_negocio`.

La descripción debe estar redactada como lo haría una entidad pública en un concurso (qué se compra o contrata), no como descripción técnica interna.

Candidatos a incluir:

| Proyecto | Descripción breve | Estado |
|---|---|---|
| ManoCard | Plataforma B2B de gestión de eventos: credenciales digitales, control de acceso y captura de leads | [PENDIENTE: confirmar inclusión y redacción] |
| EstacionFlow | Sistema de control de despacho de combustible | [PENDIENTE: confirmar inclusión y redacción] |
| CONSTRUBEL OS / Capataz | ERP para empresa constructora | [PENDIENTE: confirmar inclusión y redacción] |
| Integración JCB LiveLink → HubSpot | Integración de telemetría de flotas de maquinaria con CRM | [PENDIENTE: confirmar inclusión y redacción] |
| ERP de restaurante | ERP para restaurante de comida rápida con integración a Odoo | [PENDIENTE: confirmar inclusión y redacción] |
| [PENDIENTE: otros proyectos o servicios que ALBRAN quiere ofrecer al Estado] | | |

Proyectos en cotización (por ejemplo IC360) quedan fuera hasta que estén ejecutados. [PENDIENTE: confirmar]

## 4. Etiquetado

### Criterio de relevancia

[PENDIENTE: definir y confirmar. Borrador para discusión:]

Un concurso es **relevante** si ALBRAN podría presentarse como proveedor con su oferta actual:

- Desarrollo de software a medida (sistemas, plataformas web, aplicaciones móviles).
- Implementación o personalización de productos SaaS.
- Modernización de procesos, integraciones entre sistemas o consultoría de infraestructura.

Casos a decidir explícitamente:

- Compra de licencias de software de terceros sin servicios asociados. [PENDIENTE]
- Compra de equipo de cómputo. [PENDIENTE]
- Soporte técnico o mantenimiento de equipo. [PENDIENTE]
- Software como accesorio de un equipo (por ejemplo, equipo médico con software incluido): no relevante.

El criterio final se documenta en `notebooks/02_etiquetado.ipynb` con ejemplos positivos, negativos y límite.

### Procedimiento

1. **Muestreo** (`src/etiquetado/muestreo.py`) sobre 2024–2025:
   - Pre-filtro amplio: palabras clave (software, sistema, plataforma, aplicación, desarrollo, licencia, informático, tecnología, digital, web) y categorías de tecnología.
   - Muestra aleatoria del resto para no heredar el sesgo del pre-filtro.
   - Proporción: [PENDIENTE: definir; punto de partida 70% pre-filtro / 30% aleatoria].
2. **Pre-etiquetado** (`src/etiquetado/preetiquetar.py`): el LLM recibe el criterio y el texto del concurso y devuelve JSON. Se guarda en `etiqueta_llm` y `justificacion_llm`.

   ```json
   {"relevante": true, "justificacion": "Contrata el desarrollo de un sistema de gestión"}
   ```

3. **Revisión humana**: vista de etiquetado en el panel Streamlit que muestra el concurso, la propuesta del LLM y botones de confirmar o corregir. Se guarda en `etiqueta_humana`.
4. **Exportación**: `etiquetas/etiquetas.csv` versionado en git.
5. Meta: ≈1,000 concursos con etiqueta humana.

Reportar en el notebook: distribución de clases, tasa de acuerdo LLM-humano y ejemplos de desacuerdo.

## 5. Evaluación

### División

División temporal, para simular el uso real:

| Conjunto | Periodo |
|---|---|
| Entrenamiento | Enero 2024 – junio 2025 |
| Validación | Julio – diciembre 2025 |
| Prueba | Enero 2026 en adelante |

[PENDIENTE: ajustar según la cantidad de positivos por periodo]

### Métricas

| Métrica | Qué mide | Por qué |
|---|---|---|
| Precisión@10 por día | Cuántos de los 10 del día son relevantes | Es lo que ve el usuario |
| Recall | Qué fracción de los relevantes aparece | Costo de perder oportunidades |
| AUC-PR | Calidad del ordenamiento en todo el rango | Adecuada para clases desbalanceadas |

Accuracy no se reporta como métrica principal: con ~1% de positivos, predecir siempre «no relevante» da ~99%.

### Etiquetas para precisión@10 (pooling)

La muestra de etiquetado no garantiza que los 10 primeros de cada día de prueba estén etiquetados. Para cada día del conjunto de prueba:

1. Correr todos los métodos comparados.
2. Unir sus top 10.
3. Etiquetar a mano los concursos de esa unión que no tengan etiqueta (`origen_muestra = 'pooling'`).

Así se comparan todos los métodos sobre el mismo conjunto etiquetado.

### Comparaciones

1. Filtro por palabra clave (línea base).
2. Filtro por categoría (línea base).
3. Solo clasificador TF-IDF + RL.
4. Solo similitud con embeddings.
5. Score combinado.
6. LLM zero-shot.

Valor agregado de la propuesta: comparar TF-IDF contra embeddings como medida de similitud con el portafolio (coseno sobre vectores TF-IDF frente a coseno sobre embeddings).

Cerrar con análisis de errores: ejemplos de falsos positivos y falsos negativos del modelo combinado, con su explicación.
