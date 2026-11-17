from datetime import datetime

import pytest

from src.etiquetado.activo import armar_lote, curva, entrenar, estado, evaluar
from src.modelo.clasificador import terminos

POSITIVOS = [
    "desarrollo software sistema web planilla",
    "software gestion expediente plataforma web",
    "sistema informatico web desarrollo modulo",
    "plataforma web software portal ciudadano",
]
NEGATIVOS = [
    "compra jeringa medicamento hospital",
    "adquisicion frijol arroz alimento",
    "medicamento tableta compra farmacia",
    "alimento racion frijol escuela",
]


def agregar(con, ocid, titulo, texto, fecha, etiqueta=None, conjunto=None, ronda=None, origen=None):
    # texto_tfidf ya poblado: así las pruebas no cargan spaCy
    con.execute(
        "INSERT INTO procesos (ocid, titulo, items_texto, fecha_publicacion, texto_tfidf) "
        "VALUES (?, ?, '', ?, ?)",
        [ocid, titulo, datetime.fromisoformat(fecha), texto],
    )
    if conjunto:
        con.execute(
            "INSERT INTO etiquetas (ocid, conjunto, ronda, origen_muestra, etiqueta_humana) "
            "VALUES (?, ?, ?, ?, ?)",
            [ocid, conjunto, ronda, origen, etiqueta],
        )


@pytest.fixture
def con_semilla(con):
    for i in range(12):
        txt = POSITIVOS[i % 4]
        agregar(
            con,
            f"s-pos-{i}",
            "Desarrollo de software",
            txt,
            "2024-02-01",
            True,
            "entrenamiento",
            0,
            "prefiltro",
        )
        txt = NEGATIVOS[i % 4]
        agregar(
            con,
            f"s-neg-{i}",
            "Compra de insumos",
            txt,
            "2024-02-01",
            False,
            "entrenamiento",
            0,
            "aleatoria",
        )
    for i in range(4):
        agregar(
            con,
            f"v-pos-{i}",
            "Software",
            POSITIVOS[i],
            "2025-08-01",
            True,
            "validacion",
            None,
            "prefiltro",
        )
        agregar(
            con,
            f"v-neg-{i}",
            "Insumos",
            NEGATIVOS[i],
            "2025-08-01",
            False,
            "validacion",
            None,
            "aleatoria",
        )
    # Sin etiquetar en entrenamiento: 60 del pre-filtro (mezcla de vocabulario) y 60 fuera
    for i in range(60):
        mezcla = f"{POSITIVOS[i % 4]} {NEGATIVOS[i % 3]}" if i % 2 else POSITIVOS[i % 4]
        agregar(con, f"pool-pre-{i}", "Software y equipo", mezcla, "2024-05-01")
        agregar(con, f"pool-otro-{i}", "Compra de frijol", NEGATIVOS[i % 4], "2024-05-01")
    return con


def test_lote_70_30_con_propuesta(con_semilla):
    modelo = entrenar(con_semilla)
    agregados = armar_lote(con_semilla, modelo, ronda=1, n=20)
    assert agregados == {"incertidumbre": 14, "aleatoria": 6}

    filas = con_semilla.execute(
        "SELECT conjunto, p_propuesta, terminos_propuesta, etiqueta_humana FROM etiquetas "
        "WHERE ronda = 1"
    ).fetchall()
    assert len(filas) == 20
    assert all(c == "entrenamiento" and h is None for c, _, _, h in filas)
    assert all(0 <= p <= 1 and t.startswith("a favor:") for _, p, t, _ in filas)


def test_lote_no_repite_ni_sale_del_periodo(con_semilla):
    armar_lote(con_semilla, entrenar(con_semilla), ronda=1, n=20)
    armar_lote(con_semilla, entrenar(con_semilla), ronda=2, n=20)
    repetidos = con_semilla.execute(
        "SELECT count(*) - count(DISTINCT ocid) FROM etiquetas"
    ).fetchall()[0][0]
    fuera = con_semilla.execute(
        "SELECT count(*) FROM etiquetas e JOIN procesos p USING (ocid) "
        "WHERE e.ronda >= 1 AND p.fecha_publicacion >= '2025-07-01'"
    ).fetchall()[0][0]
    assert repetidos == 0 and fuera == 0


def test_estado_detecta_pendientes(con_semilla):
    assert estado(con_semilla) == (0, 0)
    armar_lote(con_semilla, entrenar(con_semilla), ronda=1, n=10)
    assert estado(con_semilla) == (1, 10)


def test_entrenar_exige_las_dos_clases(con):
    for i in range(5):
        agregar(
            con,
            f"n-{i}",
            "Frijol",
            NEGATIVOS[i % 4],
            "2024-02-01",
            False,
            "entrenamiento",
            0,
            "aleatoria",
        )
    agregar(
        con, "p-0", "Software", POSITIVOS[0], "2024-02-01", True, "entrenamiento", 0, "prefiltro"
    )
    with pytest.raises(ValueError, match="al menos 2"):
        entrenar(con)


def test_evaluar_validacion(con_semilla):
    metricas = evaluar(con_semilla, entrenar(con_semilla))
    assert metricas["n"] == 8 and metricas["positivos"] == 4
    assert metricas["auc_pr"] == pytest.approx(1.0)
    assert 0 <= metricas["auc_pr_reponderado"] <= 1


def test_terminos_a_favor_y_en_contra(con_semilla):
    modelo = entrenar(con_semilla)
    favor, contra = terminos(modelo, ["software web jeringa medicamento"])[0]
    assert "software" in favor or "web" in favor
    assert "jeringa" in contra or "medicamento" in contra


def test_curva_una_fila_por_ronda(con_semilla):
    armar_lote(con_semilla, entrenar(con_semilla), ronda=1, n=10)
    con_semilla.execute("UPDATE etiquetas SET etiqueta_humana = false WHERE ronda = 1")
    filas = curva(con_semilla)
    assert [f["ronda"] for f in filas] == [0, 1]
    assert filas[1]["n_entrenamiento"] == filas[0]["n_entrenamiento"] + 10


def test_poblar_textos_solo_lo_que_falta(con):
    from src.limpieza.normalizar import poblar_textos

    agregar(con, "ya", "Compra de frijol", "frijol", "2024-02-01")
    con.execute(
        "INSERT INTO procesos (ocid, titulo, items_texto, fecha_publicacion) "
        "VALUES ('nuevo', 'DESARROLLO DE SISTEMAS INFORMÁTICOS <br />', '', '2024-02-01')"
    )
    assert poblar_textos(con, ["ya", "nuevo"]) == 1
    assert poblar_textos(con, ["ya", "nuevo"]) == 0
    tfidf, emb = con.execute(
        "SELECT texto_tfidf, texto_embedding FROM procesos WHERE ocid = 'nuevo'"
    ).fetchall()[0]
    assert "sistema" in tfidf and "<br" not in emb


def test_exportar_solo_etiquetadas(con_semilla, tmp_path):
    import pandas as pd

    from src.etiquetado.exportar import exportar

    con_semilla.execute("UPDATE etiquetas SET etiqueta_humana = NULL WHERE ocid = 's-pos-0'")
    ruta = tmp_path / "etiquetas.csv"
    assert exportar(con_semilla, ruta) == 31
    datos = pd.read_csv(ruta)
    assert "s-pos-0" not in set(datos["ocid"])
    assert datos["conjunto"].tolist() == sorted(datos["conjunto"])
