# Radar de Licitaciones — Plan de trabajo
> Documento de seguimiento · Proyecto final PLN, UFM
> Estado: Fase 0 · Actualizado: octubre 2026

Cada fase termina funcional y probada antes de empezar la siguiente. Al cerrar una fase, actualizar la línea "Estado" del encabezado y la de `CLAUDE.md` (sección Fase actual).

## Fase 0 — Inventario y preparación

1. Revisar el repositorio: código, notebooks, datos o notas de la sesión anterior. Reportar qué hay antes de crear o mover archivos.
2. Crear la estructura de `docs/03`, sección 3, sin borrar lo existente.
3. `uv init` con Python 3.12; dependencias: `requests`, `duckdb`, `pandas`, `spacy`, `scikit-learn`, `sentence-transformers`, `streamlit`, `anthropic`, `python-dotenv`, `joblib`; desarrollo: `ruff`, `pytest`, `ipykernel`.
4. `.gitignore` (datos crudos, `*.duckdb`, `modelos/*.joblib`, `.env`) y `.env.example`.
5. `src/db.py` con la creación del esquema de `docs/03`, sección 4.

**Terminado cuando:** `uv run pytest` corre (aunque no haya pruebas aún), `uv run ruff check .` pasa y `src/db.py` crea `datos/radar.duckdb` con las tablas.

## Fase 1 — Captura

1. Verificaciones de `docs/02`, sección 7, puntos 1 a 3 y 6. Actualizar `docs/02` con los resultados.
2. `src/captura/ocds_client.py`: índice, SHA, descarga con reintentos, guardado en `datos/crudos/`.
3. `src/captura/actualizar.py`: corrida idempotente que inserta o actualiza `procesos` y registra en `corridas`. Parámetro para procesar un mes específico.
4. Carga histórica: enero 2024 al mes en curso.
5. Pruebas: parseo del índice y de un record de ejemplo guardado en `tests/fixtures/`.
6. Iniciar la prueba de rezago (`docs/02`, sección 7, punto 4); toma varios días.

**Terminado cuando:** correr `actualizar` dos veces seguidas no duplica registros, la segunda termina con `sin_cambios`, y la tabla `procesos` tiene el histórico completo.

## Fase 2 — Limpieza

1. `notebooks/01_exploracion.ipynb`: longitud de textos, campos vacíos, n-gramas frecuentes, frases plantilla, distribución por modalidad y entidad.
2. `src/limpieza/normalizar.py`: texto de entrada y sus dos representaciones (`docs/03`, sección 5).
3. Poblar `texto_tfidf` y `texto_embedding` para todo el histórico.
4. Pruebas de normalización con casos concretos.

**Terminado cuando:** cada decisión de la tabla de limpieza tiene evidencia en el notebook y las columnas de texto están pobladas.

## Fase 3 — Etiquetado

1. Confirmar conmigo el criterio de relevancia (`docs/04`, sección 4). No avanzar sin esto.
2. `src/etiquetado/muestreo.py`.
3. `src/etiquetado/preetiquetar.py` con el modelo `claude-haiku-5-5`.
4. Vista de etiquetado en `src/producto/app.py`.
5. Etiquetar hasta la meta y exportar `etiquetas/etiquetas.csv`.
6. `notebooks/02_etiquetado.ipynb`: distribución, acuerdo LLM-humano, desacuerdos.

**Terminado cuando:** hay ≈1,000 etiquetas humanas distribuidas en los tres periodos de la división temporal.

## Fase 4 — Modelado

1. Confirmar conmigo el contenido de `portafolio/proyectos_albran.csv`.
2. `src/modelo/clasificador.py`: entrenamiento, guardado, inferencia, explicación por términos.
3. `src/modelo/similitud.py`: embeddings del portafolio en caché, similitud máxima y proyecto más parecido.
4. `src/modelo/ranking.py`: filtros de negocio, score, top 10, escritura en `predicciones`.
5. Ajuste de hiperparámetros y de α en validación.

**Terminado cuando:** una función recibe una fecha y devuelve los 10 concursos del día con score, explicación y proyecto similar.

## Fase 5 — Evaluación

1. Líneas base: palabra clave, categoría, LLM zero-shot.
2. Pooling de etiquetas sobre los días de prueba.
3. Métricas de `docs/04`, sección 5, para los seis métodos.
4. Comparación TF-IDF contra embeddings como medida de similitud.
5. `notebooks/03_evaluacion.ipynb` con tablas, curvas PR y análisis de errores.

**Terminado cuando:** el notebook corre de inicio a fin y muestra la tabla comparativa completa.

## Fase 6 — Producto

1. Panel Streamlit: lista del día, filtros (modalidad, monto, fecha de cierre), explicación, enlace al concurso, botones de retroalimentación. Paleta de `docs/01`, sección 7.
2. `src/producto/notificar.py`: resumen de los 10 mejores por webhook de Slack.
3. Guardado de retroalimentación en `retroalimentacion`.

**Terminado cuando:** una persona que no conoce el proyecto puede abrir el panel, entender por qué se recomienda cada concurso y marcar su decisión.

## Fase 7 — Operación y demo

1. Programar corrida 05:00 y resumen 07:00 en el host definido.
2. Alertas a Slack por error o SHA sin cambios en 48 h.
3. Script de simulación día por día sobre un periodo reciente para la demo.

**Terminado cuando:** el pipeline corre sin intervención al menos 5 días seguidos y la simulación reproduce la demo sin red.

## Fase 8 — Entrega

1. `README.md`: qué es, cómo instalar, cómo correr cada etapa, fuente de datos y licencia (citar al Ministerio de Finanzas Públicas, CC BY 4.0).
2. Notebooks limpios y ejecutados de inicio a fin.
3. Actualizar la presentación con resultados reales.

**Terminado cuando:** [PENDIENTE: según formato de entrega final del curso].

## Decisiones abiertas

| Decisión | Fase que bloquea | Responsable |
|---|---|---|
| Criterio de relevancia y casos límite | 3 | Milton |
| Contenido del portafolio | 4 | Milton |
| Proporción pre-filtro / aleatoria en el muestreo | 3 | Milton, con propuesta de Claude Code |
| Rangos de filtros de negocio (montos, modalidades) | 4 | Milton |
| Host de ejecución | 7 | Milton |
| Fecha y formato de la entrega final | 8 | Curso |
| Rezago real del OCDS y cobertura NPG | 1 (medición), 8 (limitaciones) | Medición |
