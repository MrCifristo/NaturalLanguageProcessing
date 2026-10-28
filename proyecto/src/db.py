"""Conexión y esquema de la base DuckDB del radar."""

from pathlib import Path

import duckdb

RUTA_DB = Path(__file__).resolve().parents[1] / "datos" / "radar.duckdb"

# procesos ajustado a lo que trae el OCDS real: sin descripcion ni monto (no vienen),
# con caracteristicas, unspsc y fecha_publicacion
ESQUEMA = """
CREATE TABLE IF NOT EXISTS procesos (
    ocid                 VARCHAR PRIMARY KEY,
    nog                  VARCHAR,
    titulo               VARCHAR,
    items_texto          VARCHAR,
    caracteristicas      VARCHAR,
    n_items              INTEGER,
    unspsc               VARCHAR,
    entidad              VARCHAR,
    entidad_id           VARCHAR,
    modalidad            VARCHAR,
    categoria            VARCHAR,
    estado               VARCHAR,
    fecha_publicacion    TIMESTAMP,
    fecha_inicio         TIMESTAMP,
    fecha_cierre         TIMESTAMP,
    fecha_release        TIMESTAMP,
    mes_archivo          VARCHAR,
    texto_tfidf          VARCHAR,
    texto_embedding      VARCHAR,
    primera_vez_visto    TIMESTAMP,
    ultima_actualizacion TIMESTAMP
);

CREATE TABLE IF NOT EXISTS etiquetas (
    ocid               VARCHAR,
    origen_muestra     VARCHAR,
    etiqueta_llm       BOOLEAN,
    justificacion_llm  VARCHAR,
    etiqueta_humana    BOOLEAN,
    fecha              TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predicciones (
    ocid               VARCHAR,
    fecha_corrida      DATE,
    version_modelo     VARCHAR,
    p_relevante        DOUBLE,
    similitud_max      DOUBLE,
    proyecto_similar   VARCHAR,
    score              DOUBLE,
    terminos_top       VARCHAR,
    rango              INTEGER
);

CREATE TABLE IF NOT EXISTS retroalimentacion (
    ocid               VARCHAR,
    fecha              TIMESTAMP,
    decision           VARCHAR
);

CREATE TABLE IF NOT EXISTS corridas (
    id                 VARCHAR PRIMARY KEY,
    fecha              TIMESTAMP,
    mes_archivo        VARCHAR,
    sha                VARCHAR,
    estado             VARCHAR,
    nuevos             INTEGER,
    actualizados       INTEGER,
    error              VARCHAR
);
"""


def conectar(solo_lectura: bool = False) -> duckdb.DuckDBPyConnection:
    RUTA_DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(RUTA_DB), read_only=solo_lectura)
    if not solo_lectura:
        con.execute(ESQUEMA)
    return con
