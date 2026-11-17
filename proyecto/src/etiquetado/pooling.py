"""Pooling de los días de prueba (docs/04, sección 5; D-13).

Para cada día de prueba se corren los cinco métodos, se unen sus top 10 y los procesos de la
unión que no tengan etiqueta entran a `etiquetas` como conjunto 'prueba', origen 'pooling',
sin propuesta (a ciegas). Así todos los métodos se comparan sobre el mismo conjunto etiquetado.

    python -m src.etiquetado.pooling              # 4 días al azar con semilla fija

Es idempotente y acumulativo: sin portafolio corre los tres métodos que no lo necesitan; al
volver a correrlo con el portafolio agrega solo lo que traen la similitud y el combinado.
"""

import argparse
from datetime import date

import duckdb
import pandas as pd

from src.db import conectar
from src.modelo.clasificador import cargar
from src.modelo.ranking import METODOS, alfa_validacion, candidatos, puntajes, top

# Semana Santa 2026 (Domingo de Ramos a Domingo de Resurrección): volumen atípico
SEMANA_SANTA = (date(2026, 3, 29), date(2026, 4, 5))
INICIO_PRUEBA = date(2026, 1, 1)


def elegir_dias(con: duckdb.DuckDBPyConnection, n: int = 4, semilla: int = 42) -> list[date]:
    """n días hábiles de prueba con volumen normal (entre los cuartiles 1 y 3), al azar.

    El orden sale de un hash de la fecha con la semilla, igual que el muestreo (D-11).
    """
    filas = con.execute(
        """
        WITH dias AS (
            SELECT fecha_publicacion::DATE AS dia, count(*) AS n FROM procesos
            WHERE fecha_publicacion >= $inicio AND dayofweek(fecha_publicacion) BETWEEN 1 AND 5
              AND fecha_publicacion::DATE NOT BETWEEN $ss_desde AND $ss_hasta
            GROUP BY 1
        ), limites AS (
            SELECT quantile_cont(n, 0.25) AS q1, quantile_cont(n, 0.75) AS q3 FROM dias
        )
        SELECT dia FROM dias, limites WHERE n BETWEEN q1 AND q3
        ORDER BY hash(dia::VARCHAR || ':' || $semilla) LIMIT $n
        """,
        {
            "inicio": INICIO_PRUEBA,
            "ss_desde": SEMANA_SANTA[0],
            "ss_hasta": SEMANA_SANTA[1],
            "semilla": str(semilla),
            "n": n,
        },
    ).fetchall()
    return sorted(f[0] for f in filas)


def union_top(puntuados: pd.DataFrame) -> dict[str, list[str]]:
    """ocids del top 10 de cada método disponible."""
    return {m: top(puntuados, m)["ocid"].tolist() for m in METODOS if m in puntuados}


def insertar(con: duckdb.DuckDBPyConnection, ocids: list[str]) -> int:
    """Agrega a `etiquetas` los que no estén ya etiquetados o muestreados; devuelve cuántos."""
    if not ocids:
        return 0
    con.register("_pool", pd.DataFrame({"ocid": sorted(set(ocids))}))
    n = con.execute(
        """
        INSERT INTO etiquetas (ocid, conjunto, origen_muestra)
        SELECT ocid, 'prueba', 'pooling' FROM _pool
        WHERE ocid NOT IN (SELECT ocid FROM etiquetas)
        """
    ).fetchall()[0][0]
    con.unregister("_pool")
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dias", type=int, default=4)
    p.add_argument("--semilla", type=int, default=42)
    args = p.parse_args()

    modelo, version = cargar()
    try:
        from src.modelo.similitud import cargar_portafolio

        portafolio = cargar_portafolio()
    except FileNotFoundError as e:
        print(f"Sin portafolio ({e}); se omiten similitud y combinado.")
        portafolio = None

    con = conectar()
    try:
        alfa = 0.5  # sin portafolio no hay combinado y α no se usa
        if portafolio is not None:
            alfa, auc = alfa_validacion(con, modelo, portafolio)
            print(f"α = {alfa} (AUC-PR del combinado en validación: {auc:.3f})")
        dias = elegir_dias(con, args.dias, args.semilla)
        print(f"Modelo {version} · días de prueba: {', '.join(f'{d:%a %d/%m/%Y}' for d in dias)}")
        for dia in dias:
            tops = union_top(puntajes(candidatos(con, dia), modelo, alfa, portafolio))
            union = {o for ocids in tops.values() for o in ocids}
            nuevos = insertar(con, list(union))
            detalle = " · ".join(f"{m} {len(o)}" for m, o in tops.items())
            print(f"{dia:%d/%m}: {detalle} → unión {len(union)}, nuevos {nuevos}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
