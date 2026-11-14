import json

import pandas as pd
import pytest

from src.captura import actualizar, ocds_client
from src.captura.extraer import leer_mes
from tests.conftest import FIXTURES


class RespuestaFalsa:
    def __init__(self, texto: str):
        self.text = texto

    def raise_for_status(self):
        pass

    def json(self):
        return json.loads(self.text)


@pytest.fixture
def indice_falso(monkeypatch):
    texto = (FIXTURES / "indice.json").read_text(encoding="utf-8")
    monkeypatch.setattr(ocds_client.requests, "get", lambda url, timeout: RespuestaFalsa(texto))


# --- Índice ---


def test_indice_devuelve_un_registro_por_mes(indice_falso):
    meses = ocds_client.indice()
    assert len(meses) == 4
    assert {"year", "month", "files"} <= meses[0].keys()


def test_meses_sin_desde_devuelve_solo_el_mas_reciente(indice_falso):
    # El índice no viene ordenado; el mes en curso es el máximo (año, mes)
    assert actualizar.meses(None) == [(2026, 10)]


def test_meses_desde_filtra_y_ordena(indice_falso):
    assert actualizar.meses("2026-01") == [(2026, 1), (2026, 9), (2026, 10)]


def test_get_reintenta_y_luego_falla(monkeypatch):
    llamadas = []

    def falla(url, timeout):
        llamadas.append(url)
        raise ocds_client.requests.ConnectionError("503")

    monkeypatch.setattr(ocds_client.requests, "get", falla)
    monkeypatch.setattr(ocds_client.time, "sleep", lambda s: None)
    with pytest.raises(ocds_client.requests.ConnectionError):
        ocds_client.indice()
    assert len(llamadas) == 4


# --- Record package ---


@pytest.fixture
def df(zip_mes) -> pd.DataFrame:
    return leer_mes(zip_mes).set_index("ocid")


def test_una_fila_por_record(df):
    assert len(df) == 2
    assert df.loc["ocds-xqjsxa-31859046", "mes_archivo"] == "gc-2026-9"


def test_nog_sin_prefijo(df):
    assert df.loc["ocds-xqjsxa-31859046", "nog"] == "31859046"


def test_items_repetidos_se_unen_una_vez(df):
    # Dos ítems "Brazalete" con distinta característica
    fila = df.loc["ocds-xqjsxa-31859046"]
    assert fila["n_items"] == 2
    assert fila["items_texto"] == "Brazalete"
    assert fila["unspsc"] == "42261609"


def test_caracteristicas_sin_tipo_de_obligatoriedad(df):
    carac = df.loc["ocds-xqjsxa-31859046", "caracteristicas"]
    assert "Color: Rosado" in carac and "Color: Celeste" in carac
    assert "Obligatorio salvo excepciones" not in carac


def test_fechas_en_hora_de_guatemala_sin_zona(df):
    fila = df.loc["ocds-xqjsxa-31859046"]
    assert fila["fecha_publicacion"] == pd.Timestamp("2026-09-07 08:56:49")
    assert fila["fecha_publicacion"].tzinfo is None


def test_proceso_sin_fecha_de_cierre(df):
    assert pd.isna(df.loc["ocds-xqjsxa-31859046", "fecha_cierre"])
    assert df.loc["ocds-xqjsxa-31772102", "fecha_cierre"] == pd.Timestamp("2026-09-02 14:44:45")


# --- Idempotencia de la corrida ---


@pytest.fixture
def corrida(monkeypatch, zip_mes):
    estado = {"sha": "aaa", "forzar": []}
    monkeypatch.setattr(ocds_client, "sha_mes", lambda anio, mes: estado["sha"])

    def descargar(anio, mes, forzar=False):
        estado["forzar"].append(forzar)
        return zip_mes

    monkeypatch.setattr(ocds_client, "descargar_mes", descargar)
    return estado


def _contar(con, tabla: str) -> int:
    return con.execute(f"SELECT count(*) FROM {tabla}").fetchone()[0]


def test_segunda_corrida_sin_cambios_no_duplica(con, corrida):
    assert actualizar.procesar_mes(con, 2026, 9) == "gc-2026-9: ok, 2 nuevos, 0 actualizados"
    assert (
        actualizar.procesar_mes(con, 2026, 9) == "gc-2026-9: sin_cambios, 0 nuevos, 0 actualizados"
    )
    assert _contar(con, "procesos") == 2
    assert _contar(con, "corridas") == 2


def test_sha_nuevo_actualiza_sin_duplicar(con, corrida):
    actualizar.procesar_mes(con, 2026, 9)
    primera = con.execute("SELECT primera_vez_visto FROM procesos ORDER BY ocid").fetchall()

    corrida["sha"] = "bbb"
    assert actualizar.procesar_mes(con, 2026, 9) == "gc-2026-9: ok, 0 nuevos, 2 actualizados"
    assert _contar(con, "procesos") == 2
    # Conserva cuándo se vio por primera vez y vuelve a descargar el ZIP
    assert con.execute("SELECT primera_vez_visto FROM procesos ORDER BY ocid").fetchall() == primera
    assert corrida["forzar"] == [False, True]


def test_upsert_conserva_texto_calculado(con, corrida):
    actualizar.procesar_mes(con, 2026, 9)
    con.execute("UPDATE procesos SET texto_embedding = 'calculado'")
    corrida["sha"] = "bbb"
    actualizar.procesar_mes(con, 2026, 9)
    assert con.execute("SELECT DISTINCT texto_embedding FROM procesos").fetchall() == [
        ("calculado",)
    ]
