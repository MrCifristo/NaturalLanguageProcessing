"""Corrida de captura: descarga meses del OCDS y los inserta o actualiza en procesos.

Uso (desde proyecto/):
    python -m src.captura.actualizar                      # mes en curso
    python -m src.captura.actualizar --desde 2024-01      # histórico hasta el mes en curso
"""

import argparse
from datetime import datetime

from src.captura import ocds_client
from src.captura.extraer import leer_mes
from src.db import conectar

COLUMNAS = [
    "ocid", "nog", "titulo", "items_texto", "caracteristicas", "n_items", "unspsc",
    "entidad", "entidad_id", "modalidad", "categoria", "estado",
    "fecha_publicacion", "fecha_inicio", "fecha_cierre", "fecha_release", "mes_archivo",
]


def procesar_mes(con, anio: int, mes: int) -> str:
    mes_archivo = f"gc-{anio}-{mes}"
    ahora = datetime.now()
    sha = ocds_client.sha_mes(anio, mes)
    previo = con.execute(
        "SELECT sha FROM corridas WHERE mes_archivo = ? AND estado = 'ok' ORDER BY fecha DESC LIMIT 1",
        [mes_archivo],
    ).fetchone()
    if previo and previo[0] == sha:
        estado, nuevos, actualizados = "sin_cambios", 0, 0
    else:
        # SHA distinto: el ZIP local está desactualizado
        ruta = ocds_client.descargar_mes(anio, mes, forzar=previo is not None)
        df = leer_mes(ruta)
        con.register("nuevo", df)
        nuevos = con.execute("SELECT count(*) FROM nuevo WHERE ocid NOT IN (SELECT ocid FROM procesos)").fetchone()[0]
        actualizados = len(df) - nuevos
        cols = ", ".join(COLUMNAS)
        # Upsert por ocid: conserva primera_vez_visto y las columnas de texto ya calculadas
        con.execute(f"""
            INSERT INTO procesos ({cols}, primera_vez_visto, ultima_actualizacion)
            SELECT {cols}, ?, ? FROM nuevo
            ON CONFLICT (ocid) DO UPDATE SET
                {", ".join(f"{c} = excluded.{c}" for c in COLUMNAS if c != "ocid")},
                ultima_actualizacion = excluded.ultima_actualizacion
        """, [ahora, ahora])
        con.unregister("nuevo")
        estado = "ok"

    con.execute(
        "INSERT INTO corridas VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
        [f"{mes_archivo}@{ahora:%Y%m%dT%H%M%S}", ahora, mes_archivo, sha, estado, nuevos, actualizados],
    )
    return f"{mes_archivo}: {estado}, {nuevos} nuevos, {actualizados} actualizados"


def meses(desde: str | None) -> list[tuple[int, int]]:
    disponibles = sorted((r["year"], r["month"]) for r in ocds_client.indice())
    if desde is None:
        return disponibles[-1:]
    anio, mes = map(int, desde.split("-"))
    return [m for m in disponibles if m >= (anio, mes)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--desde", help="AAAA-MM; sin él solo procesa el mes en curso")
    args = p.parse_args()
    con = conectar()
    for anio, mes in meses(args.desde):
        try:
            print(procesar_mes(con, anio, mes), flush=True)
        except Exception as e:  # un mes con error no detiene el histórico
            con.execute(
                "INSERT INTO corridas VALUES (?, ?, ?, NULL, 'error', 0, 0, ?)",
                [f"gc-{anio}-{mes}@{datetime.now():%Y%m%dT%H%M%S}", datetime.now(), f"gc-{anio}-{mes}", str(e)],
            )
            print(f"gc-{anio}-{mes}: error {e}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
