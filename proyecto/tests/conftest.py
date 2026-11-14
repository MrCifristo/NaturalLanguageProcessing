import json
import zipfile
from pathlib import Path

import duckdb
import pytest

from src.db import ESQUEMA

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def paquete() -> dict:
    # Dos records reales de 2026-09, recortados a los campos que lee extraer.py
    return json.loads((FIXTURES / "record_package.json").read_text(encoding="utf-8"))


@pytest.fixture
def zip_mes(tmp_path, paquete) -> Path:
    # Mismo formato que entrega la API: un ZIP con un solo JSON adentro
    ruta = tmp_path / "2026-09.zip"
    with zipfile.ZipFile(ruta, "w") as z:
        z.writestr("2026-09_Guatecompras.json", json.dumps(paquete))
    return ruta


@pytest.fixture
def con():
    c = duckdb.connect(":memory:")
    c.execute(ESQUEMA)
    yield c
    c.close()
