"""Panel del radar. Por ahora solo la vista de etiquetado.

    ../.venv/bin/streamlit run src/producto/app.py

La propuesta del modelo (P y términos) se muestra solo en las rondas de entrenamiento que la
traen; validación, prueba y la semilla se etiquetan a ciegas (regla de anclaje, D-08).
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import duckdb

# streamlit run ejecuta este archivo como script, sin proyecto/ en el path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.db import conectar  # noqa: E402

URL_CONCURSO = "https://www.guatecompras.gt/concursos/consultaConcurso.aspx?nog={nog}&o=5"

CAMPOS = """
    e.ocid, e.conjunto, e.ronda, e.p_propuesta, e.terminos_propuesta,
    p.nog, p.titulo, p.items_texto, p.caracteristicas, p.n_items, p.unspsc, p.entidad,
    p.modalidad, p.fecha_publicacion, p.fecha_cierre
"""


def lotes(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Conjuntos y rondas con su avance."""
    filas = con.execute(
        """
        SELECT conjunto, ronda, count(*) AS total, count(etiqueta_humana) AS hechas
        FROM etiquetas GROUP BY conjunto, ronda
        ORDER BY conjunto = 'validacion', conjunto, ronda
        """
    ).fetchall()
    return [dict(zip(("conjunto", "ronda", "total", "hechas"), f, strict=True)) for f in filas]


def siguiente(
    con: duckdb.DuckDBPyConnection, conjunto: str, ronda: int | None, saltados: list[str]
) -> dict | None:
    """Próximo pendiente del lote. El orden mezcla pre-filtro y aleatoria sin mostrar el origen."""
    cur = con.execute(
        f"""
        SELECT {CAMPOS} FROM etiquetas e JOIN procesos p USING (ocid)
        WHERE e.conjunto = ? AND e.ronda IS NOT DISTINCT FROM ? AND e.etiqueta_humana IS NULL
          AND NOT list_contains(?, e.ocid)
        ORDER BY hash(e.ocid) LIMIT 1
        """,
        [conjunto, ronda, saltados],
    )
    nombres = [d[0] for d in cur.description]
    filas = cur.fetchall()
    return dict(zip(nombres, filas[0], strict=True)) if filas else None


def guardar(con: duckdb.DuckDBPyConnection, ocid: str, relevante: bool | None) -> None:
    """Pone la etiqueta humana; None la borra (deshacer)."""
    fecha = datetime.now() if relevante is not None else None
    con.execute(
        "UPDATE etiquetas SET etiqueta_humana = ?, fecha = ? WHERE ocid = ?",
        [relevante, fecha, ocid],
    )


def muestra_propuesta(fila: dict) -> bool:
    """Regla de anclaje: solo entrenamiento después de la semilla, y solo si hay propuesta."""
    return (
        fila["conjunto"] == "entrenamiento"
        and (fila["ronda"] or 0) > 0
        and (fila["p_propuesta"] is not None)
    )


def limpiar(texto: str | None) -> str:
    return re.sub(r"<br\s*/?>", "\n", texto or "").strip()


def nombre_lote(lote: dict) -> str:
    if lote["conjunto"] == "entrenamiento":
        nombre = "Semilla" if lote["ronda"] == 0 else f"Ronda {lote['ronda']}"
    else:
        nombre = {"validacion": "Validación", "prueba": "Prueba"}[lote["conjunto"]]
    return f"{nombre} · {lote['hechas']}/{lote['total']}"


def vista_etiquetado() -> None:
    import streamlit as st

    st.session_state.setdefault("saltados", [])
    st.session_state.setdefault("ultimo", None)

    # Conexión corta por acción: así la base no queda bloqueada para los notebooks
    con = conectar()
    try:
        disponibles = lotes(con)
        if not disponibles:
            st.info("No hay lotes. Correr `python -m src.etiquetado.muestreo`.")
            return
        # Opciones estables (conjunto, ronda): si fueran los dicts, el avance cambiaría la
        # opción y Streamlit volvería al primer lote después de cada etiqueta
        por_clave = {(d["conjunto"], d["ronda"]): d for d in disponibles}
        clave = st.sidebar.radio(
            "Lote", list(por_clave), format_func=lambda k: nombre_lote(por_clave[k]), key="lote"
        )
        lote = por_clave[clave]
        st.sidebar.progress(lote["hechas"] / lote["total"])

        ultimo = st.session_state["ultimo"]
        if ultimo and st.sidebar.button("Deshacer la última", shortcut="u"):
            guardar(con, ultimo, None)
            st.session_state["ultimo"] = None
            st.rerun()
        if st.session_state["saltados"] and st.sidebar.button("Volver a los saltados"):
            st.session_state["saltados"] = []
            st.rerun()

        fila = siguiente(con, lote["conjunto"], lote["ronda"], st.session_state["saltados"])
        if fila is None:
            st.success("Lote terminado.")
            return

        st.subheader(limpiar(fila["titulo"]))
        cierre = f"{fila['fecha_cierre']:%d/%m/%Y}" if fila["fecha_cierre"] else "sin fecha"
        st.caption(
            f"{fila['entidad']} · {fila['modalidad']} · publicado "
            f"{fila['fecha_publicacion']:%d/%m/%Y} · cierre {cierre} · "
            f"UNSPSC {fila['unspsc'] or '—'} · [ver en Guatecompras]"
            f"({URL_CONCURSO.format(nog=fila['nog'])})"
        )
        st.text(limpiar(fila["items_texto"])[:2000] or "(sin ítems)")
        if fila["caracteristicas"]:
            with st.expander("Características"):
                st.text(limpiar(fila["caracteristicas"])[:3000])

        if muestra_propuesta(fila):
            st.info(
                f"Propuesta del modelo: P = {fila['p_propuesta']:.2f} · "
                f"{fila['terminos_propuesta'] or ''}"
            )

        a, b, c = st.columns(3)
        decision = None
        if a.button("Relevante", type="primary", shortcut="1", width="stretch"):
            decision = True
        if b.button("No relevante", shortcut="2", width="stretch"):
            decision = False
        if c.button("Saltar", shortcut="3", width="stretch"):
            st.session_state["saltados"].append(fila["ocid"])
            st.rerun()
        if decision is not None:
            guardar(con, fila["ocid"], decision)
            st.session_state["ultimo"] = fila["ocid"]
            st.rerun()
        st.caption("Atajos: 1 relevante · 2 no relevante · 3 saltar · u deshacer")
    finally:
        con.close()


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Radar de Licitaciones", layout="wide")
    vista_etiquetado()


if __name__ == "__main__":
    main()
