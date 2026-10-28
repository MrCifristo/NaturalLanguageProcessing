# Radar de Licitaciones — Arquitectura y pipeline
> Documento técnico · Proyecto final PLN, UFM
> Estado: diseño propuesto · Actualizado: octubre 2026

## 1. Dos ciclos

| Ciclo | Frecuencia | Qué hace | Entradas | Salidas |
|---|---|---|---|---|
| Offline (entrenamiento) | Una vez, y reentrenos manuales | Etiquetado, entrenamiento, ajuste de α, evaluación | Histórico 2024–2026, etiquetas, portafolio | `modelos/clasificador.joblib`, métricas |
| Online (operación) | Diario, 05:00 | Captura, limpieza, inferencia, ranking, notificación | Archivo OCDS del mes, modelo entrenado, portafolio | Lista del día en DuckDB, panel, resumen 07:00 |

El ciclo diario nunca entrena. Solo hace inferencia sobre procesos nuevos o actualizados.

## 2. Pipeline

```
API OCDS ──► captura ──► limpieza ──► modelo ──► ranking ──► producto
 /files       JSON del     texto        P(relev.)   score +      panel +
              mes, diff    normalizado  + similitud filtros +    resumen
              por ocid                              explicación
```

| Etapa | Módulo | Responsabilidad |
|---|---|---|
| Captura | `src/captura/` | Consultar índice, comparar SHA, descargar, insertar o actualizar procesos |
| Limpieza | `src/limpieza/` | Construir el texto de entrada y sus dos representaciones |
| Modelo | `src/modelo/clasificador.py`, `similitud.py` | Probabilidad de relevancia y similitud máxima con el portafolio |
| Ranking | `src/modelo/ranking.py` | Filtros de negocio, score combinado, explicación, top 10 |
| Producto | `src/producto/` | Panel Streamlit y notificación diaria |

## 3. Estructura del repositorio

```
radar-licitaciones/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── docs/
│   ├── 01-proyecto-y-rubrica.md
│   ├── 02-fuente-de-datos-ocds.md
│   ├── 03-arquitectura-y-pipeline.md
│   ├── 04-modelado-y-evaluacion.md
│   └── 05-plan-de-trabajo.md
├── src/
│   ├── captura/
│   │   ├── ocds_client.py      # índice, SHA, descarga, reintentos
│   │   └── actualizar.py       # punto de entrada de la corrida diaria
│   ├── limpieza/
│   │   └── normalizar.py       # texto de entrada, TF-IDF y embeddings
│   ├── modelo/
│   │   ├── clasificador.py     # entrenamiento, carga e inferencia
│   │   ├── similitud.py        # embeddings del portafolio y similitud
│   │   └── ranking.py          # filtros, score, explicación
│   ├── etiquetado/
│   │   ├── muestreo.py         # pre-filtro + muestra aleatoria
│   │   └── preetiquetar.py     # propuesta de etiqueta con LLM
│   ├── producto/
│   │   ├── app.py              # panel Streamlit (incluye vista de etiquetado)
│   │   └── notificar.py        # resumen diario por Slack
│   └── db.py                   # conexión y esquema DuckDB
├── notebooks/
│   ├── 01_exploracion.ipynb    # exploración y decisiones de limpieza
│   ├── 02_etiquetado.ipynb     # distribución de etiquetas, acuerdo LLM-humano
│   └── 03_evaluacion.ipynb     # métricas, líneas base, análisis de errores
├── tests/
├── datos/
│   ├── crudos/                 # archivos OCDS descargados (no versionado)
│   └── radar.duckdb            # base única (no versionada)
├── etiquetas/
│   └── etiquetas.csv           # exportación versionada de etiquetas humanas
├── portafolio/
│   └── proyectos_albran.csv    # perfil de la empresa
└── modelos/
    └── clasificador.joblib     # no versionado
```

Respecto al deck de la propuesta se agregan `src/etiquetado/`, `src/db.py` y `tests/`. [PENDIENTE: confirmar]

## 4. Esquema de datos (DuckDB)

Propuesta inicial. Ajustar las columnas de `procesos` cuando se verifiquen las rutas OCDS (`docs/02`, sección 4).

```sql
CREATE TABLE procesos (
    ocid                 VARCHAR PRIMARY KEY,
    nog                  VARCHAR,
    titulo               VARCHAR,
    descripcion          VARCHAR,
    items_texto          VARCHAR,
    entidad              VARCHAR,
    modalidad            VARCHAR,
    categoria            VARCHAR,
    monto                DOUBLE,
    moneda               VARCHAR,
    estado               VARCHAR,
    fecha_inicio         TIMESTAMP,
    fecha_cierre         TIMESTAMP,
    fecha_release        TIMESTAMP,
    texto_tfidf          VARCHAR,
    texto_embedding      VARCHAR,
    mes_archivo          VARCHAR,     -- 'gc-2026-9'
    primera_vez_visto    TIMESTAMP,
    ultima_actualizacion TIMESTAMP
);

CREATE TABLE etiquetas (
    ocid               VARCHAR,
    origen_muestra     VARCHAR,       -- 'prefiltro' | 'aleatoria' | 'pooling'
    etiqueta_llm       BOOLEAN,
    justificacion_llm  VARCHAR,
    etiqueta_humana    BOOLEAN,       -- etiqueta final
    fecha              TIMESTAMP
);

CREATE TABLE predicciones (
    ocid               VARCHAR,
    fecha_corrida      DATE,
    version_modelo     VARCHAR,
    p_relevante        DOUBLE,
    similitud_max      DOUBLE,
    proyecto_similar   VARCHAR,
    score              DOUBLE,
    terminos_top       VARCHAR,       -- JSON con los 5 términos de más peso
    rango              INTEGER
);

CREATE TABLE retroalimentacion (
    ocid               VARCHAR,
    fecha              TIMESTAMP,
    decision           VARCHAR        -- 'me_interesa' | 'no_aplica'
);

CREATE TABLE corridas (
    id                 VARCHAR PRIMARY KEY,
    fecha              TIMESTAMP,
    mes_archivo        VARCHAR,
    sha                VARCHAR,
    estado             VARCHAR,       -- 'ok' | 'sin_cambios' | 'error'
    nuevos             INTEGER,
    actualizados       INTEGER,
    error              VARCHAR
);
```

## 5. Limpieza y preparación

Cada decisión se documenta en `notebooks/01_exploracion.ipynb` con evidencia (ejemplos, conteos).

| Decisión | Qué se hace | Por qué |
|---|---|---|
| Texto de entrada | Concatenar título + descripción + descripciones de ítems | Un título tiene ~16 palabras; los ítems aportan contexto |
| Representación TF-IDF | Minúsculas, sin tildes, tokenización y lematización con spaCy, sin stopwords | Que las variantes de una palabra coincidan en el vocabulario |
| Representación embeddings | Texto casi original: solo espacios y caracteres de control normalizados | El modelo preentrenado aprovecha tildes, mayúsculas y contexto |
| Ruido | Eliminar frases plantilla («según especificaciones técnicas», «ver bases») y códigos internos | No aportan significado y dominan el vocabulario |
| Duplicados | Un registro por `ocid`; las actualizaciones sobrescriben estado y fechas | Los archivos se regeneran y un proceso aparece en varias corridas |
| Faltantes | Sin descripción: solo título. Sin fecha de cierre: fuera del ranking diario | No se puede calcular vigencia |
| Vigencia | Solo procesos con periodo de ofertas abierto entran al ranking | Un concurso cerrado no es accionable |

La lista de frases plantilla se construye en la exploración a partir de los n-gramas más frecuentes. [PENDIENTE]

## 6. Operación

| Elemento | Valor |
|---|---|
| Corrida diaria | 05:00, `uv run python -m src.captura.actualizar` |
| Resumen | 07:00, `uv run python -m src.producto.notificar` |
| Panel | `uv run streamlit run src/producto/app.py` |
| Programador | launchd en macOS o cron en Linux |
| Host | [PENDIENTE: confirmar. Candidato: servidor homelab siempre encendido] |
| Logs | Tabla `corridas` + archivo de log rotado [PENDIENTE: ruta] |
| Alertas | Error de corrida o SHA sin cambios en 48 h → mensaje a Slack |

Para la demo de la entrega: simular el ciclo diario día por día sobre un periodo reciente ya descargado, para no depender de la red el día de la presentación. El script programado real queda funcional, pero no es requisito de la demo.
