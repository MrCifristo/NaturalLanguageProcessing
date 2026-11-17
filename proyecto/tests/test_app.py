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


def test_titulos_y_entidades_legibles():
    from src.producto.app import entidad_legible, titulo_legible

    assert titulo_legible("-COMPRA DIRECTA- SUMINISTRO DE CPU Y UPS PARA EL IGSS") == (
        "Compra directa- suministro de CPU y UPS para el IGSS"
    )
    assert titulo_legible("SERVIDORES EN RED, SON DOS") == "Servidores en red, son dos"
    assert titulo_legible("Compra Directa por Recarga") == "Compra Directa por Recarga"
    largo = titulo_legible("palabra " * 40, maximo=30)
    assert largo.endswith("…") and len(largo) <= 31
    assert entidad_legible("INSTITUTO NACIONAL DE ELECTRIFICACIÓN -INDE-") == (
        "Instituto Nacional de Electrificación -INDE-"
    )


def test_plazo_y_modalidad():
    from datetime import date, datetime

    from src.producto.app import a_favor, modalidad_corta, plazo

    dia = date(2026, 10, 9)
    assert plazo(dia, None) == ("—", "sin fecha de cierre", False)
    assert plazo(dia, datetime(2026, 10, 10, 9)) == ("1", "día para ofertar", True)
    assert plazo(dia, datetime(2026, 10, 22, 14)) == ("13", "días para ofertar", False)
    assert plazo(dia, datetime(2026, 10, 9, 18)) == ("0", "cierra hoy", True)
    assert modalidad_corta("Compra Directa con Oferta Electrónica (Art. 43 LCE Inciso b)") == (
        "Compra directa"
    )
    assert modalidad_corta("Procedimientos regulados por el artículo 54 LCE") == ""
    assert a_favor("a favor: computadora, pantalla · en contra: frijol") == "computadora, pantalla"
    assert a_favor("a favor: — · en contra: frijol") == ""


def test_lista_guardada_y_decision(con):
    from datetime import date

    from src.producto.app import decidir, dias_disponibles, lista_guardada

    con_lote = con
    con_lote.execute(
        "INSERT INTO procesos (ocid, nog, titulo, fecha_publicacion) VALUES ('a', '1', 'A', ?)",
        [datetime(2026, 10, 9, 8)],
    )
    con_lote.execute(
        "INSERT INTO predicciones (ocid, fecha_corrida, score, rango, seccion) "
        "VALUES ('a', '2026-10-09', 0.9, 1, 'principal')"
    )
    assert dias_disponibles(con_lote) == [
        {"dia": date(2026, 10, 9), "recomendados": 1, "publicados": 1}
    ]
    decidir(con_lote, "a", "no_aplica")
    decidir(con_lote, "a", "me_interesa")
    assert lista_guardada(con_lote, date(2026, 10, 9))[0]["decision"] == "me_interesa"
