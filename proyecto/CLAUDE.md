# Radar de Licitaciones
> Contexto para Claude Code · Proyecto final de Procesamiento de Lenguaje Natural (UFM) · ALBRAN Industries
> Estado: captura, exploración y pruebas hechas, criterio de relevancia cerrado; sigue la parte 3 (ruff, muestreo) · Actualizado: 14/11/2026

## Contexto

Herramienta que revisa a diario los procesos publicados en Guatecompras (portal de compras públicas de Guatemala) y entrega a una PyME de software una lista corta de concursos relevantes, ordenados por relevancia y con la razón de cada recomendación. Primer usuario: ALBRAN Industries.

Es el proyecto final del curso de PLN. La rúbrica exige un producto end-to-end: captura de datos reales, limpieza documentada, al menos una técnica de PLN del curso y un producto usable por alguien fuera del equipo.

Existe una sesión previa de Claude Code sobre este proyecto en otro equipo. Puede haber código o notas anteriores en el repositorio.

## Documentos de referencia

Viven en `devdocs/proyecto/` (raíz del repo, ignorada por git): son documentos de desarrollo y no se versionan. Dentro de ellos y de este archivo, `docs/0N` es la forma corta de `devdocs/proyecto/0N-…`. Leer según la tarea. No duplicar su contenido en el código ni en otros documentos, y no citarlos desde lo que sí se entrega (notebooks, bitácora, README).

| Archivo | Contenido |
|---|---|
| `../devdocs/proyecto/01-proyecto-y-rubrica.md` | Objetivo, usuario, alcance, rúbrica y criterios de evaluación |
| `../devdocs/proyecto/02-fuente-de-datos-ocds.md` | API OCDS de Guatecompras: endpoints verificados, formato, campos, pendientes |
| `../devdocs/proyecto/03-arquitectura-y-pipeline.md` | Ciclos offline/online, estructura del repo, módulos, esquema de datos, limpieza |
| `../devdocs/proyecto/04-modelado-y-evaluacion.md` | Clasificador, embeddings, score, etiquetado asistido, métricas |
| `../devdocs/proyecto/05-plan-de-trabajo.md` | Fases, tareas, criterios de terminado y decisiones abiertas |

## Stack

| Capa | Tecnología | Notas |
|---|---|---|
| Lenguaje | Python 3.13 | Todo el pipeline |
| Gestión de entorno | `.venv` de la raíz del repo | `requirements.txt` de la raíz; `pyproject.toml` solo configura pytest y ruff (D-01) |
| Captura | requests | API OCDS de Minfin, sin scraping |
| Almacenamiento | DuckDB | Un archivo: `datos/radar.duckdb` |
| Limpieza | pandas, spaCy (`es_core_news_sm`) | Tokenización y lematización |
| Clasificador | scikit-learn | TF-IDF + regresión logística |
| Embeddings | sentence-transformers | `paraphrase-multilingual-MiniLM-L12-v2`, CPU |
| Etiquetado | Aprendizaje activo con el propio clasificador | Sin LLM (D-08) |
| Producto | Streamlit + correo por SMTP | Panel web, resumen diario y alertas (D-09) |
| Programación | launchd (macOS) o cron | Corrida diaria 05:00, resumen 07:00 |
| Calidad | ruff, pytest | Formato, lint y pruebas |

## Convenciones

- Desarrollo modular e incremental: cada etapa del pipeline queda funcional y probada antes de empezar la siguiente.
- Carpetas, módulos, identificadores, docstrings y commits en español, siguiendo la estructura de `docs/03`.
- La lógica reutilizable vive en `src/`. Los notebooks importan desde `src/` y documentan decisiones; no contienen lógica que el pipeline diario necesite.
- Cada transformación de limpieza se documenta (qué se hace y por qué) en `notebooks/01_exploracion.ipynb`. La rúbrica lo evalúa.
- Type hints en funciones públicas. Formato y lint con `ruff`.
- Pruebas con `pytest` como mínimo para captura (parseo del índice y del record package) y limpieza (normalización).
- Secretos solo en `.env` (no versionado). Mantener `.env.example` con los nombres de variables: `SMTP_SERVIDOR`, `SMTP_PUERTO`, `SMTP_USUARIO`, `SMTP_CLAVE_APP`, `CORREO_DESTINO`.
- Correr pruebas desde `proyecto/` con `../.venv/bin/python -m pytest`.
- No versionar datos crudos, la base DuckDB ni modelos entrenados: `datos/crudos/`, `*.duckdb`, `modelos/*.joblib` van en `.gitignore`.
- Todo script del ciclo diario es idempotente: correrlo dos veces el mismo día no duplica registros.

## Cómo trabajar conmigo

- Si una tarea requiere una decisión no documentada en `docs/`, preguntar antes de implementarla.
- Cuando tomes una decisión menor por tu cuenta, decirlo explícitamente al terminar, no presentarla como ya acordada.
- No asumir criterios comerciales de ALBRAN (qué concursos le interesan, montos, modalidades). Están marcados como `[PENDIENTE]` hasta que yo los confirme.
- Preferir pasos concretos y verificables sobre explicaciones conceptuales.

## Lo que está lockeado

No cambiar sin confirmación explícita:

- **Fuente de datos:** API OCDS en `https://ocds.guatecompras.gt/files`. No se hace scraping de guatecompras.gt para datos que ya están en el OCDS.
- **Técnicas de PLN:** clasificador TF-IDF + regresión logística (modelo principal) y similitud con embeddings preentrenados (segunda señal).
- **Sin LLM (D-08, 13/11/2026):** ni en el ranking, ni en el etiquetado, ni como línea base. Las propuestas de etiqueta salen del clasificador por aprendizaje activo.
- **Anclaje:** solo entrenamiento se etiqueta con propuesta del modelo; validación, prueba y pooling se etiquetan a ciegas.
- **Score:** `score = α · P(relevante) + (1 − α) · similitud_máxima`, con α ajustado en validación.
- **Ciclos:** entrenamiento offline una vez con datos históricos etiquetados; el ciclo diario solo hace inferencia.
- **Evaluación:** división temporal; métricas principales precisión@10 por día, recall y AUC-PR. Accuracy no se usa.
- **Estructura del repositorio:** la definida en `docs/03-arquitectura-y-pipeline.md`.

## Qué NO hacer

- No reentrenar el modelo dentro de la corrida diaria.
- No descargar ni procesar los PDF de bases en el MVP.
- No generar etiquetas finales automáticamente: la etiqueta final siempre es una decisión humana; el modelo solo propone.
- No commitear datos, llaves ni la base DuckDB.
- No reestructurar código existente de la sesión anterior sin mostrarme primero qué hay y qué cambiarías.

## Fase actual

**Cierre de la Fase 2 y preparación de la Fase 3.** Entrega final el **domingo 15/11/2026**. Detalle y criterios de terminado en `docs/05-plan-de-trabajo.md` (el plan del fin de semana está al final). Lo hecho y lo que falta, con fechas, está en `reporte/bitacora.md`.

Se trabaja **por partes**: cerrar una, reportar y pedir confirmación antes de la siguiente.

### Hecho en la sesión del 13/11

- Pruebas en `tests/` (33, todas pasan): `conftest.py`, `test_captura.py`, `test_normalizar.py` y fixtures sin datos de oferentes. Se corren con `../.venv/bin/python -m pytest -q -p no:anyio` (el plugin anyio da una advertencia).
- `.env.example` con las variables SMTP y `pyproject.toml` (solo pytest por ahora, falta ruff).
- Correcciones que atraparon las pruebas: P-03 (`fetchone` en DuckDB 2), P-04 (id de `corridas`) y P-05 (regex de IGSS).
- Decisiones D-08 (sin LLM, aprendizaje activo) y D-09 (correo en vez de Slack), con los docs 01, 03, 04 y 05 actualizados.
- Criterio de relevancia cerrado caso por caso, en `docs/04`, sección 4. Hallazgos H-11 (ruido del pre-filtro) y H-12 (tasa de positivos ~2 %).
- Nada de esto está commiteado. Avisar a Milton que es buen momento para un commit si no lo hizo.

### Lo siguiente, en orden (parte 3)

1. ~~Ruff en `pyproject.toml` y `requirements.txt` de la raíz~~ (hecho el 14/11; `ruff check src tests` pasa, el notebook 01 tiene 7 avisos pendientes).
2. ~~Esquema de `etiquetas`~~ (hecho el 14/11; `src/db.py::migrar` recreó la tabla vacía; `ocid` es llave y `etiqueta_humana` NULL marca lo muestreado pendiente).
3. ~~`src/etiquetado/muestreo.py`~~ (hecho el 14/11, D-11; 70/30 confirmado). Correr `../.venv/bin/python -m src.etiquetado.muestreo` con el kernel del notebook cerrado: DuckDB no deja escribir si otro proceso tiene la base abierta.
4. ~~Vista de etiquetado~~ (hecho el 14/11): `../.venv/bin/streamlit run src/producto/app.py`. Atajos 1/2/3/u; la propuesta solo aparece en rondas ≥ 1 de entrenamiento.
5. ~~Embeddings del histórico~~: se calculan bajo demanda (D-10).
6. Después: `activo.py`, clasificador, similitud, ranking, evaluación, `notificar.py` y simulación de la demo.

### Bloqueado por Milton

- Contenido del portafolio (`portafolio/proyectos_albran.csv`, con proyectos de hardware). Hace falta el sábado temprano.
- Proporción incertidumbre/aleatoria de los lotes, cuenta de correo y formato de la entrega.
- Las 4 celdas ✍️ del notebook 01.

### Detalles técnicos que conviene recordar

- DuckDB 2: usar `con.execute(sql).fetchall()`; `fetchone()` deja el resultado abierto y bloquea el `register` siguiente. `USING SAMPLE` se aplica antes del `WHERE`; para muestrear de forma reproducible, `ORDER BY hash(ocid || ':' || semilla) LIMIT n` (D-11).
- Ruido que la limpieza todavía no quita (P-07): «MSPAS 2024 GUA-CD-167», «EPQ-GSI-135-2024 SC.20243105», «FYB-M 2830-2026».
- La API del OCDS estuvo en mantenimiento el 13/11 a las 21:30 (P-06).
