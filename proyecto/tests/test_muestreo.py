from datetime import datetime

import pytest

from src.etiquetado.muestreo import PATRON_FRASES, muestrear, tamanos


def agregar(con, ocid: str, titulo: str, fecha: str, unspsc: str | None = None) -> None:
    con.execute(
        "INSERT INTO procesos (ocid, titulo, items_texto, unspsc, fecha_publicacion) "
        "VALUES (?, ?, '', ?, ?)",
        [ocid, titulo, unspsc, datetime.fromisoformat(fecha)],
    )


@pytest.fixture
def con_procesos(con):
    # 20 del pre-filtro y 20 fuera, en entrenamiento; 10 y 10 en validación; 5 de prueba
    for i in range(20):
        agregar(con, f"e-pre-{i}", "Desarrollo de software de planillas", "2024-03-01")
        agregar(con, f"e-otro-{i}", "Compra de jeringas", "2024-03-01")
    for i in range(10):
        agregar(con, f"v-pre-{i}", "Compra de toner", "2025-08-01", unspsc="78181501,43212105")
        agregar(con, f"v-otro-{i}", "Compra de frijol", "2025-08-01")
    for i in range(5):
        agregar(con, f"p-{i}", "Desarrollo de software", "2026-02-01")
    return con


@pytest.mark.parametrize(
    "texto",
    [
        "desarrollo de software de gestion",
        "mantenimiento del sistema informatico",
        "adquisicion de computadoras portatiles",
        "instalacion de cableado estructurado",
        "unidad de poder ininterrumpido",
    ],
)
def test_frases_que_entran(con, texto):
    assert con.execute("SELECT regexp_matches(?, ?)", [texto, PATRON_FRASES]).fetchall()[0][0]


@pytest.mark.parametrize(
    "texto",
    [
        "ministerio de desarrollo social",  # H-11: nombre de institución
        "mantenimiento del sistema de frenos",  # H-11: sistema que no es TI
        "aplicacion de pintura",
        "mantenimiento correctivo a pick ups",
    ],
)
def test_frases_que_no_entran(con, texto):
    assert not con.execute("SELECT regexp_matches(?, ?)", [texto, PATRON_FRASES]).fetchall()[0][0]


def test_proporcion_y_periodo(con_procesos):
    agregados = muestrear(con_procesos, "entrenamiento", n=10)
    assert agregados == {"prefiltro": 7, "aleatoria": 3}
    filas = con_procesos.execute(
        "SELECT ocid, conjunto, ronda, origen_muestra, etiqueta_humana FROM etiquetas"
    ).fetchall()
    assert all(o.startswith("e-") for o, *_ in filas)
    assert all(c == "entrenamiento" and r == 0 and e is None for _, c, r, _o, e in filas)
    assert sum(o.startswith("e-pre") for o, *_ in filas) == 7


def test_validacion_sin_ronda_y_por_unspsc(con_procesos):
    muestrear(con_procesos, "validacion", n=10)
    filas = con_procesos.execute("SELECT ocid, ronda, origen_muestra FROM etiquetas").fetchall()
    assert all(r is None for _, r, _o in filas)
    pre = [o for o, _, origen in filas if origen == "prefiltro"]
    assert len(pre) == 7 and all(o.startswith("v-pre") for o in pre)


def test_repetir_no_duplica(con_procesos):
    muestrear(con_procesos, "entrenamiento", n=10)
    assert muestrear(con_procesos, "entrenamiento", n=10) == {"prefiltro": 0, "aleatoria": 0}
    assert con_procesos.execute("SELECT count(*) FROM etiquetas").fetchall()[0][0] == 10


def test_subir_n_completa_lo_que_falta(con_procesos):
    muestrear(con_procesos, "entrenamiento", n=10)
    assert muestrear(con_procesos, "entrenamiento", n=20) == {"prefiltro": 7, "aleatoria": 3}


def test_misma_semilla_misma_muestra(con_procesos):
    muestrear(con_procesos, "entrenamiento", n=10)
    primera = con_procesos.execute("SELECT ocid FROM etiquetas ORDER BY ocid").fetchall()
    con_procesos.execute("DELETE FROM etiquetas")
    muestrear(con_procesos, "entrenamiento", n=10)
    assert con_procesos.execute("SELECT ocid FROM etiquetas ORDER BY ocid").fetchall() == primera


def test_tamanos(con_procesos):
    assert tamanos(con_procesos, "entrenamiento") == {"prefiltro": 20, "aleatoria": 20}
