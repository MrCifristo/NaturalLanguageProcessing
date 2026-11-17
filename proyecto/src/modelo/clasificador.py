"""Señal 1: TF-IDF (palabras + caracteres) y regresión logística sobre `texto_tfidf`."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

RUTA_MODELO = Path(__file__).resolve().parents[2] / "modelos" / "clasificador.joblib"


def construir(c: float = 1.0) -> Pipeline:
    """Configuración inicial de docs/04, sección 2."""
    vectorizador = FeatureUnion(
        [
            ("palabras", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            (
                "caracteres",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("tfidf", vectorizador),
            ("rl", LogisticRegression(C=c, class_weight="balanced", max_iter=2000)),
        ]
    )


def terminos(modelo: Pipeline, textos: list[str], n: int = 5) -> list[tuple[list[str], list[str]]]:
    """Por documento, los n términos que más empujan a favor y en contra (coeficiente × tf-idf).

    Solo se usan los n-gramas de palabras: los de caracteres («ftwa», « sof») no se leen
    como explicación, aunque sí cuentan en la P.
    """
    union = modelo.named_steps["tfidf"]
    palabras = dict(union.transformer_list)["palabras"]
    nombres = palabras.get_feature_names_out()
    coef = modelo.named_steps["rl"].coef_[0][: len(nombres)]  # palabras va primero en la unión

    x = palabras.transform(textos).tocsr()
    salida = []
    for i in range(x.shape[0]):
        fila = x.getrow(i)
        aporte = fila.data * coef[fila.indices]
        orden = np.argsort(aporte)
        favor = [nombres[fila.indices[j]] for j in orden[::-1][:n] if aporte[j] > 0]
        contra = [nombres[fila.indices[j]] for j in orden[:n] if aporte[j] < 0]
        salida.append((favor, contra))
    return salida


def formato_terminos(favor: list[str], contra: list[str]) -> str:
    return f"a favor: {', '.join(favor) or '—'} · en contra: {', '.join(contra) or '—'}"


def guardar(modelo: Pipeline, version: str, ruta: Path = RUTA_MODELO) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo, "version": version}, ruta)


def cargar(ruta: Path = RUTA_MODELO) -> tuple[Pipeline, str]:
    # joblib usa pickle: solo cargar modelos que generó este mismo pipeline
    datos = joblib.load(ruta)
    return datos["modelo"], datos["version"]


def puntuar(modelo: Pipeline, textos: list[str]) -> np.ndarray:
    """P(relevante) de cada texto (representación texto_tfidf)."""
    return modelo.predict_proba(list(textos))[:, 1]


def main() -> None:
    """Entrena con todo lo etiquetado de entrenamiento y guarda en modelos/clasificador.joblib."""
    from datetime import date

    from src.db import conectar
    from src.etiquetado.activo import entrenar, etiquetados, evaluar

    con = conectar()
    try:
        modelo = entrenar(con)
        n = len(etiquetados(con, "entrenamiento"))
        version = f"tfidf-rl-{n}et-{date.today():%Y%m%d}"
        guardar(modelo, version)
        metricas = evaluar(con, modelo) or {}
        print(f"{version} guardado en {RUTA_MODELO} · validación {metricas}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
