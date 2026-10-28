# Radar de Licitaciones
> Contexto para Claude Code · Proyecto final de Procesamiento de Lenguaje Natural (UFM) · ALBRAN Industries
> Estado: propuesta presentada, implementación por iniciar · Actualizado: octubre 2026

## Contexto

Herramienta que revisa a diario los procesos publicados en Guatecompras (portal de compras públicas de Guatemala) y entrega a una PyME de software una lista corta de concursos relevantes, ordenados por relevancia y con la razón de cada recomendación. Primer usuario: ALBRAN Industries.

Es el proyecto final del curso de PLN. La rúbrica exige un producto end-to-end: captura de datos reales, limpieza documentada, al menos una técnica de PLN del curso y un producto usable por alguien fuera del equipo.

Existe una sesión previa de Claude Code sobre este proyecto en otro equipo. Puede haber código o notas anteriores en el repositorio.

## Documentos de referencia

Leer según la tarea. No duplicar su contenido en el código ni en otros documentos.

| Archivo | Contenido |
|---|---|
| `docs/01-proyecto-y-rubrica.md` | Objetivo, usuario, alcance, rúbrica y criterios de evaluación |
| `docs/02-fuente-de-datos-ocds.md` | API OCDS de Guatecompras: endpoints verificados, formato, campos, pendientes |
| `docs/03-arquitectura-y-pipeline.md` | Ciclos offline/online, estructura del repo, módulos, esquema de datos, limpieza |
| `docs/04-modelado-y-evaluacion.md` | Clasificador, embeddings, score, etiquetado asistido, métricas |
| `docs/05-plan-de-trabajo.md` | Fases, tareas, criterios de terminado y decisiones abiertas |

## Stack

| Capa | Tecnología | Notas |
|---|---|---|
| Lenguaje | Python 3.12 | Todo el pipeline |
| Gestión de entorno | uv | `pyproject.toml` + `uv.lock` |
| Captura | requests | API OCDS de Minfin, sin scraping |
| Almacenamiento | DuckDB | Un archivo: `datos/radar.duckdb` |
| Limpieza | pandas, spaCy (`es_core_news_sm`) | Tokenización y lematización |
| Clasificador | scikit-learn | TF-IDF + regresión logística |
| Embeddings | sentence-transformers | `paraphrase-multilingual-MiniLM-L12-v2`, CPU |
| Pre-etiquetado | API de Anthropic | Modelo `claude-haiku-5-5`; solo para proponer etiquetas |
| Producto | Streamlit + webhook de Slack | Panel web y resumen diario |
| Programación | launchd (macOS) o cron | Corrida diaria 05:00, resumen 07:00 |
| Calidad | ruff, pytest | Formato, lint y pruebas |

## Convenciones

- Desarrollo modular e incremental: cada etapa del pipeline queda funcional y probada antes de empezar la siguiente.
- Carpetas, módulos, identificadores, docstrings y commits en español, siguiendo la estructura de `docs/03`.
- La lógica reutilizable vive en `src/`. Los notebooks importan desde `src/` y documentan decisiones; no contienen lógica que el pipeline diario necesite.
- Cada transformación de limpieza se documenta (qué se hace y por qué) en `notebooks/01_exploracion.ipynb`. La rúbrica lo evalúa.
- Type hints en funciones públicas. Formato y lint con `ruff`.
- Pruebas con `pytest` como mínimo para captura (parseo del índice y del record package) y limpieza (normalización).
- Secretos solo en `.env` (no versionado). Mantener `.env.example` con los nombres de variables: `ANTHROPIC_API_KEY`, `SLACK_WEBHOOK_URL`.
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
- **Rol del LLM:** solo pre-etiquetado y línea base de comparación. No participa en el ranking diario.
- **Score:** `score = α · P(relevante) + (1 − α) · similitud_máxima`, con α ajustado en validación.
- **Ciclos:** entrenamiento offline una vez con datos históricos etiquetados; el ciclo diario solo hace inferencia.
- **Evaluación:** división temporal; métricas principales precisión@10 por día, recall y AUC-PR. Accuracy no se usa.
- **Estructura del repositorio:** la definida en `docs/03-arquitectura-y-pipeline.md`.

## Qué NO hacer

- No reentrenar el modelo dentro de la corrida diaria.
- No descargar ni procesar los PDF de bases en el MVP.
- No generar etiquetas finales automáticamente: la etiqueta final siempre es una decisión humana; el LLM solo propone.
- No commitear datos, llaves ni la base DuckDB.
- No reestructurar código existente de la sesión anterior sin mostrarme primero qué hay y qué cambiarías.

## Fase actual

**Fase 0: inventario y preparación del repositorio.** Detalle y criterios de terminado en `docs/05-plan-de-trabajo.md`.

1. Revisar si el repositorio ya contiene código, notebooks o datos de la sesión anterior y reportar qué hay antes de crear nada.
2. Crear o ajustar la estructura de carpetas de `docs/03`.
3. Configurar `uv`, `ruff`, `pytest`, `.gitignore` y `.env.example`.
