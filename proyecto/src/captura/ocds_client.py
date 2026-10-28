"""Cliente de la API OCDS de Guatecompras: índice, SHA y descarga de archivos mensuales."""

import time
from pathlib import Path

import requests

BASE = "https://ocds.guatecompras.gt"
CRUDOS = Path(__file__).resolve().parents[2] / "datos" / "crudos"


def _get(url: str, timeout: int, intentos: int = 4) -> requests.Response:
    # Reintentos con backoff: 5, 10, 20 s. timeout es por lectura: una conexión
    # colgada espera como máximo 2 min antes de reintentar (P-01)
    for i in range(intentos):
        try:
            r = requests.get(url, timeout=(30, min(timeout, 120)))
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == intentos - 1:
                raise
            time.sleep(5 * 2**i)


def indice() -> list[dict]:
    """Un registro por mes publicado (gc-2020-1 … mes en curso)."""
    return _get(f"{BASE}/files", 60).json()["result"]


def sha_mes(anio: int, mes: int) -> str:
    return _get(f"{BASE}/file/sha/{anio}/{mes}", 60).text.strip()


def ruta_mes(anio: int, mes: int) -> Path:
    return CRUDOS / f"{anio}-{mes:02d}.zip"


def descargar_mes(anio: int, mes: int, forzar: bool = False) -> Path:
    """Descarga el JSON del mes. La API lo entrega comprimido en ZIP (~37 MB)."""
    destino = ruta_mes(anio, mes)
    if destino.exists() and not forzar:
        return destino
    CRUDOS.mkdir(parents=True, exist_ok=True)
    r = _get(f"{BASE}/file/json/{anio}/{mes}", 600)
    # Se escribe a un temporal para no dejar un ZIP a medias si la descarga falla
    tmp = destino.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.rename(destino)
    return destino
