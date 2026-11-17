from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from src.etiquetado.pooling import elegir_dias, insertar, union_top
from src.modelo.clasificador import construir
from src.modelo.ranking import (
    ajustar_alfa,
    candidatos,
    puntaje_categoria,
    puntaje_palabra_clave,
    puntajes,
    top,
)
from src.modelo.similitud import similitud


@pytest.fixture
def modelo():
    textos = ["software sistema web", "computadora portatil", "jeringa medicamento", "frijol"] * 3
    return construir().fit(textos, [1, 1, 0, 0] * 3)


def procesos(n_rel: int = 3, n_otros: int = 15) -> pd.DataFrame:
    filas = [
        (f"r-{i}", "Desarrollo de software web", "", "43232300", "software sistema web")
        for i in range(n_rel)
    ] + [(f"o-{i}", "Compra de frijol", "", "50221101", "frijol") for i in range(n_otros)]
    return pd.DataFrame(filas, columns=["ocid", "titulo", "items_texto", "unspsc", "texto_tfidf"])


def test_lineas_base():
    datos = pd.DataFrame(
        {
            "titulo": ["Desarrollo de SOFTWARE y plataforma web", "Compra de frijol", "Licencias"],
            "items_texto": ["sistema informático", None, ""],
            "unspsc": ["43232300,50221101", "50221101", None],
        }
    )
    assert puntaje_palabra_clave(datos).tolist() == [6, 0, 1]
    assert puntaje_categoria(datos).tolist() == [0.5, 0, 0]


def test_top_respeta_filtro_y_orden(modelo):
    puntuados = puntajes(procesos(), modelo, alfa=0.5)
    assert "similitud" not in puntuados and "combinado" not in puntuados
    assert set(top(puntuados, "palabra_clave")["ocid"]) == {"r-0", "r-1", "r-2"}
    clasif = top(puntuados, "clasificador")
    assert len(clasif) == 10 and clasif["ocid"].iloc[:3].tolist() == ["r-0", "r-1", "r-2"]
    assert top(puntuados, "similitud").empty


def test_combinado_con_portafolio(modelo):
    datos = procesos(2, 2).assign(texto_embedding=["a", "b", "c", "d"])
    proyectos = pd.DataFrame({"proyecto": ["ERP", "Cómputo"]})
    emb_port = np.eye(2)
    import src.modelo.similitud as sim_mod

    falsos = {"a": [1, 0], "b": [0.6, 0.8], "c": [0, 1], "d": [0.8, 0.6]}
    original = sim_mod.embeber
    sim_mod.embeber = lambda textos: np.array([falsos[t] for t in textos], dtype=float)
    try:
        puntuados = puntajes(datos, modelo, alfa=0.0, portafolio=(proyectos, emb_port))
    finally:
        sim_mod.embeber = original
    assert puntuados["proyecto_similar"].tolist() == ["ERP", "Cómputo", "Cómputo", "ERP"]
    np.testing.assert_allclose(puntuados["combinado"], [1, 0.8, 1, 0.8])


def test_similitud_maxima():
    sim, similar = similitud(
        np.array([[1.0, 0.0], [0.0, 1.0]]), pd.DataFrame({"proyecto": ["A", "B"]}), np.eye(2)
    )
    assert sim.tolist() == [1.0, 1.0] and similar == ["A", "B"]


def test_ajustar_alfa_elige_la_senal_buena():
    y = np.array([1, 1, 0, 0, 0])
    p_buena = np.array([0.9, 0.8, 0.1, 0.2, 0.3])
    sim_mala = np.array([0.1, 0.2, 0.9, 0.8, 0.7])
    alfa, auc = ajustar_alfa(p_buena, sim_mala, y)
    assert alfa == 1.0 and auc == pytest.approx(1.0)


def agregar(con, ocid, publicado, cierre, titulo="Compra de frijol"):
    con.execute(
        "INSERT INTO procesos (ocid, titulo, items_texto, unspsc, fecha_publicacion, fecha_cierre,"
        " texto_tfidf, texto_embedding) VALUES (?, ?, '', '', ?, ?, 'frijol', ?)",
        [ocid, titulo, publicado, cierre, titulo],
    )


def test_candidatos_solo_vigentes_del_dia(con):
    agregar(con, "ok", datetime(2026, 2, 3, 9), datetime(2026, 2, 13))
    agregar(con, "sin-cierre", datetime(2026, 2, 3, 10), None)
    agregar(con, "cerrado", datetime(2026, 2, 3, 11), datetime(2026, 2, 3, 8))
    agregar(con, "otro-dia", datetime(2026, 2, 4, 9), datetime(2026, 2, 13))
    assert candidatos(con, date(2026, 2, 3))["ocid"].tolist() == ["ok"]


def test_elegir_dias_habiles_y_reproducible(con):
    # Lunes 2 a viernes 27 de febrero, 10 procesos por día; un sábado y Semana Santa fuera
    for d in pd.bdate_range("2026-02-02", "2026-02-27"):
        for i in range(10):
            agregar(con, f"{d:%m%d}-{i}", d.to_pydatetime(), None)
    agregar(con, "sabado", datetime(2026, 2, 7), None)
    for i in range(10):
        agregar(con, f"ss-{i}", datetime(2026, 3, 31), None)
    dias = elegir_dias(con, n=4)
    assert len(dias) == 4 and dias == sorted(dias) == elegir_dias(con, n=4)
    assert all(d.weekday() < 5 and d.month == 2 for d in dias)


def test_insertar_pooling_idempotente(con):
    agregar(con, "a", datetime(2026, 2, 3), None)
    agregar(con, "b", datetime(2026, 2, 3), None)
    con.execute("INSERT INTO etiquetas (ocid, conjunto) VALUES ('b', 'validacion')")
    assert insertar(con, ["a", "b", "a"]) == 1
    assert insertar(con, ["a", "b"]) == 0
    fila = con.execute(
        "SELECT conjunto, origen_muestra, ronda, p_propuesta FROM etiquetas WHERE ocid = 'a'"
    ).fetchall()
    assert fila == [("prueba", "pooling", None, None)]


def test_union_top_solo_metodos_disponibles(modelo):
    tops = union_top(puntajes(procesos(), modelo, alfa=0.5))
    assert set(tops) == {"palabra_clave", "categoria", "clasificador"}


def test_union_top_incluye_combinado_sin_tope():
    puntuados = pd.DataFrame(
        {
            "ocid": [f"c-{i}" for i in range(12)],
            "titulo": ["Computadora portátil"] * 12,
            "unspsc": [""] * 12,
            "combinado": np.linspace(1, 0, 12),
        }
    )
    tops = union_top(puntuados)
    assert len(tops["combinado"]) == 5 and len(tops["combinado_sin_tope"]) == 10


def test_tope_de_computadoras_en_el_combinado():
    titulos = [f"Compra de {i} computadoras portátiles" for i in range(8)] + [
        "Desarrollo de sistema web",
        "Servidor rack 2U",
        "Licencia de firewall",
        "Compra de frijol",
        "Compra de jeringas",
    ]
    puntuados = pd.DataFrame(
        {
            "ocid": [f"p-{i:02d}" for i in range(len(titulos))],
            "titulo": titulos,
            "unspsc": ["43211503"] * 8 + ["", "43211501", "", "", ""],
            "combinado": np.linspace(1, 0.1, len(titulos)),
        }
    )
    seleccion = top(puntuados, "combinado")
    computo = seleccion["titulo"].str.contains("computadoras")
    assert computo.sum() == 5 and len(seleccion) == 10
    # Los 5 de cómputo son los de mayor score y el resto sigue el orden
    assert seleccion["ocid"].tolist()[:5] == [f"p-{i:02d}" for i in range(5)]
    assert "Servidor rack 2U" in seleccion["titulo"].tolist()
    # Sin tope, el top se llena de computadoras
    assert (
        top(puntuados, "combinado", max_computo=None)["titulo"].str.contains("computadoras").sum()
        == 8
    )
    # Los otros métodos no llevan tope
    assert (
        top(puntuados.rename(columns={"combinado": "clasificador"}), "clasificador")["titulo"]
        .str.contains("computadoras")
        .sum()
        == 8
    )


def test_candidatos_sin_cierre_solo_en_la_lista_diaria(con):
    agregar(con, "ok", datetime(2026, 2, 3, 9), datetime(2026, 2, 13))
    agregar(con, "sin-cierre", datetime(2026, 2, 3, 10), None)
    agregar(con, "cerrado", datetime(2026, 2, 3, 11), datetime(2026, 2, 3, 8))
    assert candidatos(con, date(2026, 2, 3))["ocid"].tolist() == ["ok"]
    assert candidatos(con, date(2026, 2, 3), sin_cierre=True)["ocid"].tolist() == [
        "ok",
        "sin-cierre",
    ]


def test_venta_computo_reconoce_plurales_y_deja_fuera_servidores():
    from src.modelo.ranking import es_venta_computo

    datos = pd.DataFrame(
        {
            "titulo": [
                "La Subgerencia solicita 4 Computadoras de escritorio",
                "COMPRA DE COMPUTADORES PORTÁTILES",
                "Adquisición de estaciones de trabajo",
                "Servidor de almacenamiento",
                "Desarrollo de sistema web",
            ],
            "unspsc": ["", "", "", "43211501", ""],
        }
    )
    assert es_venta_computo(datos).tolist() == [True, True, True, False, False]


def test_lista_del_dia_umbral_y_seccion_de_computo():
    from src.modelo.ranking import lista_del_dia

    titulos = [f"Compra de {i} computadoras" for i in range(8)] + ["Sistema web", "Frijol"]
    puntuados = pd.DataFrame(
        {
            "ocid": [f"p-{i}" for i in range(10)],
            "titulo": titulos,
            "unspsc": [""] * 10,
            "clasificador": [0.9] * 9 + [0.1],
            "combinado": np.linspace(1, 0.1, 10),
        }
    )
    lista = lista_del_dia(puntuados)
    principal = lista[lista["seccion"] == "principal"]
    assert len(principal) == 6  # 5 de cómputo + el sistema web; el frijol no pasa el umbral
    assert lista[lista["seccion"] == "computo"]["ocid"].tolist() == ["p-5", "p-6", "p-7"]
