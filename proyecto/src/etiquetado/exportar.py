"""Exporta las etiquetas humanas a etiquetas/etiquetas.csv (versionado; la base no se sube) y
las vuelve a cargar en una base nueva, por ejemplo en otra computadora.

    python -m src.etiquetado.exportar             # base → CSV
    python -m src.etiquetado.exportar --importar  # CSV → base (después de cargar procesos)
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from src.db import conectar

RUTA_CSV = Path(__file__).resolve().parents[2] / "etiquetas" / "etiquetas.csv"


def exportar(con: duckdb.DuckDBPyConnection, ruta: Path = RUTA_CSV) -> int:
    """Escribe las filas etiquetadas, ordenadas por conjunto y ocid; devuelve cuántas."""
    datos = con.execute(
        """
        SELECT e.ocid, p.nog, e.conjunto, e.ronda, e.origen_muestra, e.p_propuesta,
               e.etiqueta_humana, e.fecha
        FROM etiquetas e JOIN procesos p USING (ocid)
        WHERE e.etiqueta_humana IS NOT NULL
        ORDER BY e.conjunto, e.ocid
        """
    ).df()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    datos.to_csv(ruta, index=False)
    return len(datos)


def importar(con: duckdb.DuckDBPyConnection, ruta: Path = RUTA_CSV) -> tuple[int, int]:
    """Carga el CSV en `etiquetas`; la etiqueta del CSV gana si el ocid ya estaba.

    Devuelve (cargadas, omitidas): se omiten los ocid que no están en `procesos`, porque sin
    el proceso no hay texto para entrenar ni evaluar.
    """
    datos = pd.read_csv(ruta, dtype={"nog": str})
    datos["ronda"] = datos["ronda"].astype("Int64")
    datos["fecha"] = pd.to_datetime(datos["fecha"])
    con.register("_csv", datos)
    existentes = con.execute(
        "SELECT count(*) FROM _csv WHERE ocid IN (SELECT ocid FROM procesos)"
    ).fetchall()[0][0]
    con.execute(
        """
        INSERT INTO etiquetas (ocid, conjunto, ronda, origen_muestra, p_propuesta,
                               etiqueta_humana, fecha)
        SELECT ocid, conjunto, ronda, origen_muestra, p_propuesta, etiqueta_humana, fecha
        FROM _csv WHERE ocid IN (SELECT ocid FROM procesos)
        ON CONFLICT (ocid) DO UPDATE SET
            conjunto = excluded.conjunto, ronda = excluded.ronda,
            origen_muestra = excluded.origen_muestra, p_propuesta = excluded.p_propuesta,
            etiqueta_humana = excluded.etiqueta_humana, fecha = excluded.fecha
        """
    )
    con.unregister("_csv")
    return existentes, len(datos) - existentes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--importar", action="store_true", help="cargar el CSV en la base")
    args = p.parse_args()
    if args.importar:
        con = conectar()
        cargadas, omitidas = importar(con)
        print(f"{cargadas} etiquetas cargadas desde {RUTA_CSV}; {omitidas} sin proceso en la base")
    else:
        con = conectar(solo_lectura=True)
        print(f"{exportar(con)} etiquetas en {RUTA_CSV}")
    con.close()


if __name__ == "__main__":
    main()
