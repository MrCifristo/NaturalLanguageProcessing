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

-- Una fila por proceso muestreado; etiqueta_humana queda NULL hasta que se etiqueta (D-08)
CREATE TABLE IF NOT EXISTS etiquetas (
    ocid               VARCHAR PRIMARY KEY,
    conjunto           VARCHAR,       -- 'entrenamiento' | 'validacion' | 'prueba'
    ronda              INTEGER,       -- 0 = semilla; NULL en validación y prueba
    origen_muestra     VARCHAR,       -- 'prefiltro' | 'aleatoria' | 'incertidumbre' | 'pooling'
    p_propuesta        DOUBLE,        -- P mostrada al etiquetar; NULL si fue a ciegas
    terminos_propuesta VARCHAR,
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


def migrar(con: duckdb.DuckDBPyConnection) -> None:
    """Quita la tabla etiquetas con columnas del LLM (antes de D-08), solo si está vacía."""
    viejas = con.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'etiquetas' AND column_name = 'etiqueta_llm'"
    ).fetchall()
    if not viejas:
        return
    n = con.execute("SELECT count(*) FROM etiquetas").fetchall()[0][0]
    if n:
        raise RuntimeError(f"etiquetas tiene el esquema viejo y {n} filas; migrar a mano")
    con.execute("DROP TABLE etiquetas")


def conectar(solo_lectura: bool = False) -> duckdb.DuckDBPyConnection:
    RUTA_DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(RUTA_DB), read_only=solo_lectura)
    if not solo_lectura:
        migrar(con)
        con.execute(ESQUEMA)
    return con
