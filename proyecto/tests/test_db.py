import duckdb
import pytest

from src.db import ESQUEMA, migrar

ETIQUETAS_VIEJA = """
CREATE TABLE etiquetas (
    ocid VARCHAR, origen_muestra VARCHAR, etiqueta_llm BOOLEAN,
    justificacion_llm VARCHAR, etiqueta_humana BOOLEAN, fecha TIMESTAMP
);
"""


def columnas(con, tabla: str) -> list[str]:
    filas = con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position",
        [tabla],
    ).fetchall()
    return [f[0] for f in filas]


def test_esquema_nuevo_de_etiquetas(con):
    assert columnas(con, "etiquetas") == [
        "ocid",
        "conjunto",
        "ronda",
        "origen_muestra",
        "p_propuesta",
        "terminos_propuesta",
        "etiqueta_humana",
        "fecha",
    ]


def test_migrar_recrea_etiquetas_vieja_vacia():
    c = duckdb.connect(":memory:")
    c.execute(ETIQUETAS_VIEJA)
    migrar(c)
    c.execute(ESQUEMA)
    assert "etiqueta_llm" not in columnas(c, "etiquetas")
    assert "conjunto" in columnas(c, "etiquetas")


def test_migrar_no_borra_etiquetas_con_datos():
    c = duckdb.connect(":memory:")
    c.execute(ETIQUETAS_VIEJA)
    c.execute("INSERT INTO etiquetas VALUES ('ocds-xqjsxa-1', 'aleatoria', NULL, NULL, TRUE, NULL)")
    with pytest.raises(RuntimeError):
        migrar(c)


def test_migrar_es_idempotente(con):
    migrar(con)
    migrar(con)
    assert "conjunto" in columnas(con, "etiquetas")


def test_ocid_no_se_repite_en_etiquetas(con):
    con.execute("INSERT INTO etiquetas (ocid, conjunto) VALUES ('ocds-xqjsxa-1', 'validacion')")
    with pytest.raises(duckdb.ConstraintException):
        con.execute("INSERT INTO etiquetas (ocid, conjunto) VALUES ('ocds-xqjsxa-1', 'validacion')")
