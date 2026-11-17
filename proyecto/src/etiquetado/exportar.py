"""Exporta las etiquetas humanas a etiquetas/etiquetas.csv (versionado; la base no se sube).

python -m src.etiquetado.exportar
"""

from pathlib import Path

import duckdb

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


if __name__ == "__main__":
    con = conectar(solo_lectura=True)
    print(f"{exportar(con)} etiquetas en {RUTA_CSV}")
    con.close()
