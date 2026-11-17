"""Rondas de aprendizaje activo sobre el periodo de entrenamiento (D-08).

Entrena TF-IDF + RL con lo etiquetado de entrenamiento, mide el AUC-PR en validación y arma el
lote de la ronda siguiente: 70 % con la P más cercana a 0.5 y 30 % aleatorio del periodo, cada
uno con la propuesta del modelo (P y términos). Validación nunca recibe propuesta (anclaje).

    python -m src.etiquetado.activo              # nueva ronda si la anterior está completa
    python -m src.etiquetado.activo --curva      # AUC-PR en validación por ronda

DuckDB no deja escribir si otro proceso tiene la base abierta: correrlo con el kernel de los
notebooks cerrado. La vista de etiquetado abre conexiones cortas y no estorba.
"""

import argparse

import duckdb
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from src.db import conectar
from src.etiquetado.muestreo import (
    PATRON_FRASES,
    PATRON_UNSPSC,
    PERIODOS,
    PREFILTRO,
    tamanos,
)
from src.limpieza.normalizar import poblar_textos
from src.modelo.clasificador import construir, formato_terminos, terminos


def etiquetados(
    con: duckdb.DuckDBPyConnection, conjunto: str, hasta_ronda: int | None = None
) -> pd.DataFrame:
    """Procesos con etiqueta humana del conjunto (ocid, origen_muestra, texto_tfidf, y)."""
    filtro_ronda = "AND e.ronda <= $ronda" if hasta_ronda is not None else ""
    ocids = [
        f[0]
        for f in con.execute(
            f"SELECT ocid FROM etiquetas e WHERE conjunto = $conjunto "
            f"AND etiqueta_humana IS NOT NULL {filtro_ronda}",
            {"conjunto": conjunto, **({"ronda": hasta_ronda} if filtro_ronda else {})},
        ).fetchall()
    ]
    poblar_textos(con, ocids)
    return con.execute(
        f"""
        SELECT e.ocid, e.origen_muestra, p.texto_tfidf, e.etiqueta_humana AS y
        FROM etiquetas e JOIN procesos p USING (ocid)
        WHERE e.conjunto = $conjunto AND e.etiqueta_humana IS NOT NULL {filtro_ronda}
        ORDER BY e.ocid
        """,
        {"conjunto": conjunto, **({"ronda": hasta_ronda} if filtro_ronda else {})},
    ).df()


def entrenar(
    con: duckdb.DuckDBPyConnection, hasta_ronda: int | None = None, c: float = 1.0
) -> Pipeline:
    datos = etiquetados(con, "entrenamiento", hasta_ronda)
    positivos = int(datos["y"].sum())
    if positivos < 2 or len(datos) - positivos < 2:
        raise ValueError(
            f"Hacen falta al menos 2 relevantes y 2 no relevantes; hay {positivos} de {len(datos)}"
        )
    return construir(c).fit(datos["texto_tfidf"], datos["y"].astype(int))


def evaluar(con: duckdb.DuckDBPyConnection, modelo: Pipeline) -> dict | None:
    """AUC-PR en validación, sin reponderar (criterio de parada) y reponderado por estrato (D-11)."""
    datos = etiquetados(con, "validacion")
    y = datos["y"].astype(int)
    if y.nunique() < 2:
        return None
    p = modelo.predict_proba(datos["texto_tfidf"])[:, 1]
    # Cada etiquetado representa a (población del estrato / etiquetados del estrato) procesos
    poblacion = tamanos(con, "validacion")
    muestra = datos["origen_muestra"].value_counts()
    pesos = datos["origen_muestra"].map(lambda o: poblacion[o] / muestra[o])
    return {
        "n": len(datos),
        "positivos": int(y.sum()),
        "auc_pr": average_precision_score(y, p),
        "auc_pr_reponderado": average_precision_score(y, p, sample_weight=pesos),
    }


def candidatos(
    con: duckdb.DuckDBPyConnection, n_resto: int = 20_000, semilla: int = 42
) -> list[str]:
    """Sin etiqueta en entrenamiento: todo el pre-filtro más n_resto al azar de fuera de él.

    Puntuar los ~450 mil del periodo obligaría a lematizarlos todos (cerca de una hora, D-05).
    Fuera del pre-filtro casi nada es relevante (D-11), así que una muestra basta para que la
    zona de duda también incluya esos casos.
    """
    desde, hasta = PERIODOS["entrenamiento"]
    params = {"frases": PATRON_FRASES, "unspsc": PATRON_UNSPSC, "desde": desde, "hasta": hasta}
    base = """
        FROM procesos p
        WHERE p.fecha_publicacion >= $desde AND p.fecha_publicacion < $hasta
          AND p.ocid NOT IN (SELECT ocid FROM etiquetas)
    """
    pre = con.execute(f"SELECT p.ocid {base} AND {PREFILTRO}", params).fetchall()
    resto = con.execute(
        f"SELECT p.ocid {base} AND NOT {PREFILTRO} "
        "ORDER BY hash(p.ocid || ':resto:' || $semilla) LIMIT $n",
        {**params, "semilla": str(semilla), "n": n_resto},
    ).fetchall()
    return [f[0] for f in pre + resto]


def aleatorios(
    con: duckdb.DuckDBPyConnection, n: int, excluir: list[str], ronda: int, semilla: int = 42
) -> list[str]:
    """n procesos al azar de todo el periodo de entrenamiento, distintos en cada ronda."""
    desde, hasta = PERIODOS["entrenamiento"]
    return [
        f[0]
        for f in con.execute(
            """
            SELECT ocid FROM procesos
            WHERE fecha_publicacion >= $desde AND fecha_publicacion < $hasta
              AND ocid NOT IN (SELECT ocid FROM etiquetas) AND NOT list_contains($excluir, ocid)
            ORDER BY hash(ocid || ':ronda' || $ronda || ':' || $semilla) LIMIT $n
            """,
            {
                "desde": desde,
                "hasta": hasta,
                "excluir": excluir,
                "ronda": str(ronda),
                "semilla": str(semilla),
                "n": n,
            },
        ).fetchall()
    ]


def textos(con: duckdb.DuckDBPyConnection, ocids: list[str]) -> pd.DataFrame:
    poblar_textos(con, ocids)
    con.register("_pedidos", pd.DataFrame({"ocid": ocids}))
    datos = con.execute(
        "SELECT p.ocid, p.texto_tfidf FROM procesos p JOIN _pedidos USING (ocid) ORDER BY p.ocid"
    ).df()
    con.unregister("_pedidos")
    return datos


def armar_lote(
    con: duckdb.DuckDBPyConnection,
    modelo: Pipeline,
    ronda: int,
    n: int = 100,
    prop_incertidumbre: float = 0.7,
    n_resto: int = 20_000,
    semilla: int = 42,
) -> dict[str, int]:
    """Inserta en `etiquetas` el lote de la ronda con su propuesta; devuelve cuántos por origen."""
    n_incierto = round(n * prop_incertidumbre)

    pool = textos(con, candidatos(con, n_resto, semilla))
    pool["p"] = modelo.predict_proba(pool["texto_tfidf"])[:, 1]
    # Empates en |P − 0.5| se rompen por ocid para que el lote sea reproducible
    pool["duda"] = (pool["p"] - 0.5).abs()
    inciertos = pool.sort_values(["duda", "ocid"]).head(n_incierto).assign(origen="incertidumbre")

    azar = textos(
        con, aleatorios(con, n - len(inciertos), inciertos["ocid"].tolist(), ronda, semilla)
    )
    azar["p"] = modelo.predict_proba(azar["texto_tfidf"])[:, 1] if len(azar) else []
    azar["origen"] = "aleatoria"

    lote = pd.concat([inciertos, azar], ignore_index=True)
    explicacion = terminos(modelo, lote["texto_tfidf"].tolist())
    lote["terminos"] = [formato_terminos(f, c) for f, c in explicacion]
    lote["ronda"] = ronda

    con.register("_lote", lote[["ocid", "ronda", "origen", "p", "terminos"]])
    con.execute(
        """
        INSERT INTO etiquetas (ocid, conjunto, ronda, origen_muestra, p_propuesta, terminos_propuesta)
        SELECT ocid, 'entrenamiento', ronda, origen, p, terminos FROM _lote
        WHERE ocid NOT IN (SELECT ocid FROM etiquetas)
        """
    )
    con.unregister("_lote")
    return lote["origen"].value_counts().to_dict()


def estado(con: duckdb.DuckDBPyConnection) -> tuple[int | None, int]:
    """Última ronda de entrenamiento y cuántos de ella siguen sin etiqueta."""
    ultima = con.execute(
        "SELECT max(ronda) FROM etiquetas WHERE conjunto = 'entrenamiento'"
    ).fetchall()[0][0]
    if ultima is None:
        return None, 0
    pendientes = con.execute(
        "SELECT count(*) FROM etiquetas "
        "WHERE conjunto = 'entrenamiento' AND ronda = ? AND etiqueta_humana IS NULL",
        [ultima],
    ).fetchall()[0][0]
    return ultima, pendientes


def curva(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """AUC-PR en validación entrenando con las rondas 0..r, para cada r."""
    ultima, _ = estado(con)
    salida = []
    for r in range(0 if ultima is None else ultima + 1):
        modelo = entrenar(con, hasta_ronda=r)
        metricas = evaluar(con, modelo) or {}
        n_train = len(etiquetados(con, "entrenamiento", r))
        salida.append({"ronda": r, "n_entrenamiento": n_train, **metricas})
    return salida


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--prop-incertidumbre", type=float, default=0.7)
    p.add_argument("--curva", action="store_true", help="solo mostrar el AUC-PR por ronda")
    p.add_argument("--forzar", action="store_true", help="armar el lote aunque queden pendientes")
    args = p.parse_args()

    con = conectar()
    try:
        if args.curva:
            print(pd.DataFrame(curva(con)).to_string(index=False))
            return
        ultima, pendientes = estado(con)
        if ultima is None:
            print("No hay semilla. Correr primero `python -m src.etiquetado.muestreo`.")
            return
        if pendientes and not args.forzar:
            print(f"La ronda {ultima} tiene {pendientes} sin etiquetar; terminarla primero.")
            return

        modelo = entrenar(con)
        metricas = evaluar(con, modelo)
        if metricas:
            print(
                f"Validación ({metricas['n']} etiquetados, {metricas['positivos']} relevantes): "
                f"AUC-PR {metricas['auc_pr']:.3f} · reponderado {metricas['auc_pr_reponderado']:.3f}"
            )
        else:
            print("Validación sin etiquetas suficientes (hacen falta las dos clases).")

        agregados = armar_lote(con, modelo, ultima + 1, args.n, args.prop_incertidumbre)
        print(f"Ronda {ultima + 1}: {agregados}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
