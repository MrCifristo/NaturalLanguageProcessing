"""Muestras a ciegas para etiquetar: semilla de entrenamiento (ronda 0) y validación.

Cada muestra mezcla procesos del pre-filtro con procesos aleatorios del resto del periodo,
para no heredar el sesgo del pre-filtro (D-08). Las filas entran a `etiquetas` con
`etiqueta_humana` en NULL, que la vista de etiquetado completa después.

    python -m src.etiquetado.muestreo                      # semilla y validación, 150 c/u
    python -m src.etiquetado.muestreo --conjunto validacion --n 200
"""

import argparse

import duckdb

from src.db import conectar

# Fin exclusivo; prueba (2026 en adelante) se etiqueta por pooling, no aquí
PERIODOS = {
    "entrenamiento": ("2024-01-01", "2025-07-01"),
    "validacion": ("2025-07-01", "2026-01-01"),
}

# Frases y no palabras sueltas: «desarrollo» o «sistema» solos traen sobre todo ruido (H-11).
# Se buscan en minúsculas y sin tildes sobre título + ítems.
FRASES = [
    r"software",
    r"desarrollo del? (un |una )?(sistema|software|aplicaci|plataforma|portal|sitio)",
    r"sistemas? (informatic|de informacion|computarizad|en linea|web)",
    r"plataforma (web|digital|tecnologic|informatic|virtual|en linea|electronic)",
    r"aplicacion(es)? (web|movil|informatic|para dispositivos)",
    r"(pagina|sitio|portal) (web|electronic|de internet)",
    r"hosting",
    r"servidor(es)? (dedicad|virtual|web)",
    r"base de datos",
    r"ciberseguridad",
    r"seguridad informatica",
    r"auditoria (de sistemas|informatica)",
    r"transformacion digital",
    r"digitalizacion",
    r"firma electronica",
    r"equipo (de computo|informatico)",
    r"computadora",
    r"laptop",
    r"impresora",
    r"escaner",
    r"poder ininterrumpid",
    r"cableado (estructurado|de red|de datos)",
    r"redes? de datos",
    r"centro de datos",
    r"data ?center",
    r"cctv",
    r"videovigilancia",
    r"camaras? de (seguridad|vigilancia|video)",
    r"control de asistencia",
    r"biometric",
    r"kiosco",
    r"proyector",
    r"pantalla (led|interactiva|inteligente|tactil)",
]
PATRON_FRASES = "|".join(f"({f})" for f in FRASES)
# UNSPSC 43 (TI) y 8111 (servicios informáticos); unspsc trae varios códigos separados por coma
PATRON_UNSPSC = r"(^|,)(43|8111)"

PREFILTRO = """(
    regexp_matches(strip_accents(lower(p.titulo || ' ' || coalesce(p.items_texto, ''))), $frases)
    OR regexp_matches(coalesce(p.unspsc, ''), $unspsc)
)"""


def tamanos(con: duckdb.DuckDBPyConnection, conjunto: str) -> dict[str, int]:
    """Procesos del periodo dentro y fuera del pre-filtro (para reponderar métricas)."""
    desde, hasta = PERIODOS[conjunto]
    filas = con.execute(
        f"""
        SELECT {PREFILTRO} AS pre, count(*) FROM procesos p
        WHERE p.fecha_publicacion >= $desde AND p.fecha_publicacion < $hasta
        GROUP BY pre
        """,
        {"frases": PATRON_FRASES, "unspsc": PATRON_UNSPSC, "desde": desde, "hasta": hasta},
    ).fetchall()
    conteo = dict(filas)
    return {"prefiltro": conteo.get(True, 0), "aleatoria": conteo.get(False, 0)}


def muestrear(
    con: duckdb.DuckDBPyConnection,
    conjunto: str,
    n: int = 150,
    prop_prefiltro: float = 0.7,
    semilla: int = 42,
) -> dict[str, int]:
    """Agrega a `etiquetas` lo que falte para tener n procesos del conjunto; devuelve lo agregado.

    Idempotente: lo ya muestreado cuenta para la meta, así que repetir la llamada no agrega nada.
    El orden sale de un hash del ocid con la semilla, no de random(), para que no dependa de
    cómo DuckDB reparte el escaneo entre hilos.
    """
    desde, hasta = PERIODOS[conjunto]
    ronda = 0 if conjunto == "entrenamiento" else None
    meta = {"prefiltro": round(n * prop_prefiltro)}
    meta["aleatoria"] = n - meta["prefiltro"]

    agregados = {}
    for origen, objetivo in meta.items():
        ya = con.execute(
            "SELECT count(*) FROM etiquetas WHERE conjunto = ? AND origen_muestra = ?",
            [conjunto, origen],
        ).fetchall()[0][0]
        faltan = max(objetivo - ya, 0)
        condicion = PREFILTRO if origen == "prefiltro" else f"NOT {PREFILTRO}"
        insertados = con.execute(
            f"""
            INSERT INTO etiquetas (ocid, conjunto, ronda, origen_muestra)
            SELECT p.ocid, $conjunto, $ronda, $origen FROM procesos p
            WHERE p.fecha_publicacion >= $desde AND p.fecha_publicacion < $hasta
              AND {condicion}
              AND p.ocid NOT IN (SELECT ocid FROM etiquetas)
            ORDER BY hash(p.ocid || ':' || $semilla)
            LIMIT $faltan
            """,
            {
                "frases": PATRON_FRASES,
                "unspsc": PATRON_UNSPSC,
                "conjunto": conjunto,
                "ronda": ronda,
                "origen": origen,
                "desde": desde,
                "hasta": hasta,
                "semilla": str(semilla),
                "faltan": faltan,
            },
        ).fetchall()[0][0]
        agregados[origen] = insertados
    return agregados


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--conjunto", choices=[*PERIODOS, "ambos"], default="ambos")
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--prop-prefiltro", type=float, default=0.7)
    args = p.parse_args()
    conjuntos = list(PERIODOS) if args.conjunto == "ambos" else [args.conjunto]
    con = conectar()
    for conjunto in conjuntos:
        agregados = muestrear(con, conjunto, args.n, args.prop_prefiltro)
        print(f"{conjunto}: agregados {agregados} · población {tamanos(con, conjunto)}")
    con.close()


if __name__ == "__main__":
    main()
