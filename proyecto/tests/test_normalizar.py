import pytest

from src.limpieza.normalizar import (
    MAX_CARACTERES,
    para_embedding,
    para_tfidf,
    quitar_ruido,
    quitar_tildes,
    texto_entrada,
)

# --- Ruido en títulos (H-07, D-04) ---


@pytest.mark.parametrize(
    "sucio, limpio",
    [
        ("CODIGO IGSS: 227281 GUANTES DE LATEX", "GUANTES DE LATEX"),
        ("CÓDIGO IGSS. 5882. BRAZALETE PARA PACIENTE", "BRAZALETE PARA PACIENTE"),
        ("PEDIDO No. 38811/2026 GUANTES", "GUANTES"),
        ("EVENTO 1438/2026 PAPEL BOND", "PAPEL BOND"),
        ("AMPARO No. 02075-2026-00197 INSULINA", "INSULINA"),
        ("[DA-DAC-A18][CD-271] SERVICIO DE INTERNET", "SERVICIO DE INTERNET"),
        ("P-11111 TONER", "TONER"),
        ("NOG: 31859046 sillas", "sillas"),
        ("CODIGO 77091; GUSELKUMAB INYECTABLE", "GUSELKUMAB INYECTABLE"),
    ],
)
def test_quita_codigos_internos(sucio, limpio):
    assert quitar_ruido(sucio) == limpio


def test_quita_html_y_decodifica_entidades():
    assert quitar_ruido("Licencias &amp; soporte<br />anual") == "Licencias & soporte anual"


def test_quita_frases_plantilla():
    assert (
        quitar_ruido("Compra de sillas según especificaciones técnicas adjuntas")
        == "Compra de sillas"
    )


def test_no_deja_separadores_huerfanos():
    assert quitar_ruido("; ; mesas; ; sillas") == "mesas; sillas"


def test_conserva_cantidades_que_describen_la_compra():
    # Los números del producto no son ruido; solo los códigos con su prefijo
    assert quitar_ruido("JERINGA PRECARGADA 100 MG/ML") == "JERINGA PRECARGADA 100 MG/ML"


# --- Texto de entrada (D-03, D-07) ---


def test_entrada_omite_fragmentos_que_ya_estan_en_el_titulo():
    texto = texto_entrada(
        "BRAZALETE PARA PACIENTE", "Brazalete", "BRAZALETE PARA PACIENTE | Color: Rosado;"
    )
    assert texto == "BRAZALETE PARA PACIENTE | Color: Rosado;"


def test_entrada_omite_campos_de_formulario():
    texto = texto_entrada(
        "GUSELKUMAB", None, "Vía: Subcutánea; | TELEFONO | TIEMPO DE ENTREGA | REGIMEN TRIBUTARIO"
    )
    assert texto == "GUSELKUMAB | Vía: Subcutánea;"


def test_entrada_respeta_el_tope_de_caracteres():
    items = " | ".join(f"articulo {i}" for i in range(500))
    assert len(texto_entrada("Compra", items, None)) == MAX_CARACTERES


# --- Representaciones ---


def test_quitar_tildes_conserva_la_enie():
    assert quitar_tildes("Adquisición de año médico") == "Adquisicion de año medico"


def test_embedding_conserva_mayusculas_y_tildes():
    assert para_embedding("CODIGO IGSS: 1 Solución Inyectable") == "Solución Inyectable"


@pytest.fixture(scope="module")
def spacy_disponible():
    pytest.importorskip("spacy")
    import spacy.util

    if not spacy.util.is_package("es_core_news_sm"):
        pytest.skip("falta es_core_news_sm")


def test_tfidf_lematiza_sin_stopwords_numeros_ni_tildes(spacy_disponible):
    [salida] = para_tfidf(["ADQUISICIÓN DE 25 CAFETERAS Y TABLETAS DE METFORMINA"])
    assert salida == "adquisicion cafetera tableta metformina"


def test_tfidf_en_minusculas_evita_errores_del_lematizador(spacy_disponible):
    # P-02: en mayúsculas el modelo pequeño daba "metodologiar"
    [salida] = para_tfidf(["METODOLOGÍA DE DESARROLLO DE SOFTWARE"])
    assert "metodologia" in salida.split()
    assert "software" in salida.split()
