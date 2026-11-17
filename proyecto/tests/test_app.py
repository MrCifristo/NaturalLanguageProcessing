from datetime import datetime

import pytest

from src.producto.app import guardar, lotes, muestra_propuesta, nombre_lote, siguiente


@pytest.fixture
def con_lote(con):
    for i in range(3):
        con.execute(
            "INSERT INTO procesos (ocid, nog, titulo, fecha_publicacion) VALUES (?, ?, ?, ?)",
            [f"e-{i}", str(i), f"Proceso {i}", datetime(2024, 3, 1)],
        )
        con.execute(
            "INSERT INTO etiquetas (ocid, conjunto, ronda, origen_muestra) "
            "VALUES (?, 'entrenamiento', 0, 'prefiltro')",
            [f"e-{i}"],
        )
    con.execute(
        "INSERT INTO procesos (ocid, nog, titulo, fecha_publicacion) VALUES ('v-0', '9', 'V', ?)",
        [datetime(2025, 8, 1)],
    )
    con.execute(
        "INSERT INTO etiquetas (ocid, conjunto, origen_muestra) VALUES ('v-0', 'validacion', 'aleatoria')"
    )
    return con


def test_lotes_con_avance(con_lote):
    guardar(con_lote, "e-0", True)
    assert lotes(con_lote) == [
        {"conjunto": "entrenamiento", "ronda": 0, "total": 3, "hechas": 1},
        {"conjunto": "validacion", "ronda": None, "total": 1, "hechas": 0},
    ]


def test_siguiente_salta_etiquetados_y_saltados(con_lote):
    guardar(con_lote, "e-0", False)
    primero = siguiente(con_lote, "entrenamiento", 0, [])
    assert primero["ocid"] in {"e-1", "e-2"}
    otro = siguiente(con_lote, "entrenamiento", 0, [primero["ocid"]])
    assert otro["ocid"] not in {"e-0", primero["ocid"]}
    assert siguiente(con_lote, "entrenamiento", 0, ["e-1", "e-2"]) is None


def test_validacion_con_ronda_nula(con_lote):
    assert siguiente(con_lote, "validacion", None, [])["ocid"] == "v-0"


def test_deshacer_borra_etiqueta_y_fecha(con_lote):
    guardar(con_lote, "e-0", True)
    guardar(con_lote, "e-0", None)
    assert con_lote.execute(
        "SELECT etiqueta_humana, fecha FROM etiquetas WHERE ocid = 'e-0'"
    ).fetchall() == [(None, None)]


@pytest.mark.parametrize(
    "conjunto, ronda, p, esperado",
    [
        ("entrenamiento", 2, 0.48, True),
        ("entrenamiento", 0, 0.48, False),  # semilla: a ciegas
        ("validacion", None, 0.48, False),  # anclaje: nunca en validación
        ("prueba", None, 0.48, False),
        ("entrenamiento", 2, None, False),  # lote aleatorio sin propuesta
    ],
)
def test_regla_de_anclaje(conjunto, ronda, p, esperado):
    assert muestra_propuesta({"conjunto": conjunto, "ronda": ronda, "p_propuesta": p}) is esperado


def test_nombre_de_lote_no_cambia_al_etiquetar(con_lote):
    # Regresión: st.radio guarda la opción por su texto en el navegador. Si el texto llevaba
    # el avance («3/150»), al etiquetar cambiaba y el radio volvía a la Semilla.
    def nombres():
        return [nombre_lote(d["conjunto"], d["ronda"]) for d in lotes(con_lote)]

    antes = nombres()
    guardar(con_lote, "v-0", False)
    guardar(con_lote, "e-0", True)
    assert nombres() == antes == ["Semilla", "Validación"]
