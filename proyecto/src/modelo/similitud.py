"""Señal 2: similitud coseno entre el proceso y los proyectos del portafolio de ALBRAN.

Embeddings de `paraphrase-multilingual-MiniLM-L12-v2` sobre `texto_embedding`, normalizados,
así que el coseno es el producto punto. Los del portafolio se guardan en caché y se recalculan
solo si cambia el CSV.
"""

import hashlib
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RUTA_PORTAFOLIO = RAIZ / "portafolio" / "proyectos_albran.csv"
RUTA_CACHE = RAIZ / "modelos" / "portafolio_embeddings.npz"
NOMBRE_MODELO = "paraphrase-multilingual-MiniLM-L12-v2"
COLUMNAS = ["proyecto", "descripcion", "linea_negocio"]


@cache
def _modelo():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(NOMBRE_MODELO, device="cpu")


def embeber(textos: list[str], batch_size: int = 64) -> np.ndarray:
    """Embeddings normalizados (norma 1), uno por fila."""
    return _modelo().encode(
        list(textos), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )


def cargar_portafolio(
    ruta: Path = RUTA_PORTAFOLIO, ruta_cache: Path = RUTA_CACHE
) -> tuple[pd.DataFrame, np.ndarray]:
    """Proyectos del portafolio y sus embeddings (de la caché si el CSV no cambió)."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"Falta {ruta}. Columnas: {', '.join(COLUMNAS)}; la descripción redactada como "
            "la escribiría la entidad en un concurso."
        )
    contenido = ruta.read_bytes()
    huella = hashlib.sha256(contenido).hexdigest()
    proyectos = pd.read_csv(ruta)
    faltan = set(COLUMNAS) - set(proyectos.columns)
    if faltan:
        raise ValueError(f"Al portafolio le faltan las columnas {sorted(faltan)}")

    if ruta_cache.exists():
        cache_npz = np.load(ruta_cache)
        if str(cache_npz["huella"]) == huella:
            return proyectos, cache_npz["embeddings"]
    embeddings = embeber(proyectos["descripcion"].tolist())
    ruta_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ruta_cache, embeddings=embeddings, huella=huella)
    return proyectos, embeddings


def similitud(
    embeddings: np.ndarray, proyectos: pd.DataFrame, emb_portafolio: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Similitud máxima de cada proceso contra el portafolio y el proyecto que la alcanza."""
    sims = embeddings @ emb_portafolio.T
    mejor = sims.argmax(axis=1)
    return sims.max(axis=1), proyectos["proyecto"].iloc[mejor].tolist()
