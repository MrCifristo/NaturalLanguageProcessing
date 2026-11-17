"""Panel del radar: recomendaciones de la semana y vista de etiquetado.

    ../.venv/bin/streamlit run src/producto/app.py

Las recomendaciones salen de `predicciones`, que llena `python -m src.modelo.ranking`.
La propuesta del modelo (P y términos) en el etiquetado se muestra solo en las rondas de
entrenamiento que la traen; validación, prueba y la semilla van a ciegas (anclaje, D-08).
"""

import html
import re
import sys
from datetime import date, datetime
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


def nombre_lote(conjunto: str, ronda: int | None) -> str:
    """Nombre fijo del lote, sin el avance.

    st.radio guarda la opción elegida por su texto: si el texto incluyera «3/150», al etiquetar
    pasaría a «4/150», la opción guardada dejaría de existir y el radio volvería al primer lote.
    """
    if conjunto == "entrenamiento":
        return "Semilla" if ronda == 0 else f"Ronda {ronda}"
    return {"validacion": "Validación", "prueba": "Prueba"}[conjunto]


def vista_etiquetado() -> None:
    import streamlit as st

    st.markdown(ESTILO, unsafe_allow_html=True)
    cabecera(st, "Etiquetado")
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
            "Lote", list(por_clave), format_func=lambda k: nombre_lote(*k), key="lote"
        )
        lote = por_clave[clave]
        st.sidebar.progress(
            lote["hechas"] / lote["total"], text=f"{lote['hechas']}/{lote['total']} etiquetados"
        )

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


# --- Recomendaciones ---------------------------------------------------------------------

DIAS_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]  # fmt: skip


def dias_disponibles(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Días con lista guardada, del más reciente al más viejo, con lo publicado ese día."""
    filas = con.execute(
        """
        SELECT pr.fecha_corrida,
               count(*) FILTER (WHERE pr.seccion = 'principal') AS recomendados,
               (SELECT count(*) FROM procesos p
                WHERE p.fecha_publicacion::DATE = pr.fecha_corrida) AS publicados
        FROM predicciones pr GROUP BY pr.fecha_corrida ORDER BY pr.fecha_corrida DESC
        """
    ).fetchall()
    return [dict(zip(("dia", "recomendados", "publicados"), f, strict=True)) for f in filas]


def lista_guardada(con: duckdb.DuckDBPyConnection, dia: date) -> list[dict]:
    """La lista del día con los datos del proceso y la última decisión de ALBRAN."""
    cur = con.execute(
        """
        SELECT pr.ocid, pr.seccion, pr.rango, pr.score, pr.p_relevante, pr.similitud_max,
               pr.proyecto_similar, pr.terminos_top, p.nog, p.titulo, p.entidad, p.modalidad,
               p.fecha_cierre,
               (SELECT r.decision FROM retroalimentacion r WHERE r.ocid = pr.ocid
                ORDER BY r.fecha DESC LIMIT 1) AS decision
        FROM predicciones pr JOIN procesos p USING (ocid)
        WHERE pr.fecha_corrida = ? ORDER BY pr.rango
        """,
        [dia],
    )
    nombres = [d[0] for d in cur.description]
    return [dict(zip(nombres, f, strict=True)) for f in cur.fetchall()]


def decidir(con: duckdb.DuckDBPyConnection, ocid: str, decision: str) -> None:
    """Guarda la decisión de ALBRAN; la última es la que vale."""
    con.execute("INSERT INTO retroalimentacion VALUES (?, ?, ?)", [ocid, datetime.now(), decision])


# Palabras que nunca son siglas aunque tengan pocas vocales («DE», «LA», «EN»)
MINUSCULAS = set(
    "a al con de del e el en la las lo los o para por que se sin su sus u un una y".split()
)
# Siglas que se dejan en mayúsculas. Una lista y no una regla: «SON», «SAN», «DOS» o «RED»
# tienen la misma forma que «IGSS» o «CPU».
SIGLAS = set(
    """IGSS MSPAS SAT INE INDE MAGA MINFIN MP PNC OJ TSE RENAP CHN USAC SIE SIAF LCE PPR NOG
    CPU RAM SSD HDD USB LED LCD UPS NAS PC TI IA WAF VPN CCTV HDMI ERP POS GPS ISO SMS
    API CRM BI QR IP TV DVD CD AS400 SQL""".split()
)


def _legible(texto: str | None, cada_palabra: bool) -> str:
    """Pasa a minúsculas los textos del portal escritos en mayúsculas y respeta las siglas.

    cada_palabra=True deja en mayúscula inicial cada palabra (nombres de entidades); si no, solo
    la primera (títulos).
    """
    texto = re.sub(r"\s+", " ", limpiar(texto)).strip(" .;")
    if not cada_palabra:  # «-COMPRA DIRECTA-» en títulos; en entidades el guion cierra la sigla
        texto = texto.strip(" -")
    letras = [c for c in texto if c.isalpha()]
    if not letras or sum(c.isupper() for c in letras) / len(letras) < 0.6:
        return texto

    def palabra(p: str, primera: bool) -> str:
        minuscula = p.lower()
        es_conector = re.sub(r"[^a-záéíóúñ]", "", minuscula) in MINUSCULAS
        if re.sub(r"[^A-Z0-9]", "", p) in SIGLAS and p.upper() == p:
            return p
        # Siglas entre guiones al final del nombre de una entidad («-INDE-», «-CDAG-»)
        if re.fullmatch(r"-[A-ZÁÉÍÓÚÑ]{2,8}-?", p):
            return p
        if primera or (cada_palabra and not es_conector):
            return minuscula[:1].upper() + minuscula[1:]
        return minuscula

    return " ".join(palabra(p, i == 0) for i, p in enumerate(texto.split(" ")))


def titulo_legible(titulo: str | None, maximo: int = 150) -> str:
    """Título en oración y cortado en una palabra completa si es muy largo."""
    texto = _legible(titulo, cada_palabra=False)
    if len(texto) <= maximo:
        return texto
    return texto[:maximo].rsplit(" ", 1)[0].rstrip(" ,;:(") + "…"


def entidad_legible(entidad: str | None) -> str:
    return _legible(entidad, cada_palabra=True)


def fecha_larga(momento: date | datetime) -> str:
    texto = f"{DIAS_SEMANA[momento.weekday()]} {momento.day} de {MESES[momento.month - 1]}"
    if isinstance(momento, datetime) and (momento.hour or momento.minute):
        texto += f" a las {momento:%H:%M}"
    return texto


def plazo(dia: date, cierre: datetime | None) -> tuple[str, str, bool]:
    """(cifra, texto, urgente) del plazo para ofertar contado desde el día de la lista."""
    if cierre is None:
        return "—", "sin fecha de cierre", False
    dias = (cierre.date() - dia).days
    if dias <= 0:
        return "0", "cierra hoy", True
    return str(dias), "día para ofertar" if dias == 1 else "días para ofertar", dias <= 3


def a_favor(terminos: str | None) -> str:
    """Los términos a favor de `formato_terminos` («a favor: x, y · en contra: z»)."""
    m = re.match(r"a favor: (.*?) · en contra", terminos or "")
    return "" if not m or m.group(1) == "—" else m.group(1)


def modalidad_corta(modalidad: str | None) -> str:
    """Nombre corto de la modalidad; las que solo citan un artículo de la ley no se muestran."""
    texto = (modalidad or "").lower()
    for clave, nombre in [
        ("compra directa", "Compra directa"),
        ("licitaci", "Licitación"),
        ("cotizaci", "Cotización"),
        ("subasta", "Subasta electrónica"),
        ("arrendamiento", "Arrendamiento"),
        ("contrato abierto", "Contrato abierto"),
    ]:
        if clave in texto:
            return nombre
    return ""


ESTILO = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
.stApp, .stApp p, .stApp h1, .stApp button p, .stApp label, .stApp summary p {
  font-family: 'IBM Plex Sans', sans-serif; }
.block-container { max-width: 1040px; padding-top: 2.2rem; }
h1.radar { font-size: 1.55rem; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 .2rem; }
p.resumen { color: #5C636B; font-size: .98rem; max-width: 62ch; margin: .2rem 0 1.4rem; }
.plazo { line-height: 1; padding-top: .15rem; }
.plazo .cifra { display: block; font-size: 2.6rem; font-weight: 600; color: #1E2329;
  font-variant-numeric: tabular-nums; letter-spacing: -0.03em; }
.plazo .texto { display: block; font-size: .78rem; color: #5C636B; margin-top: .35rem; max-width: 10ch; }
.marca { font-size: .9rem; font-weight: 600; color: #1E2329; letter-spacing: -0.005em; padding: .45rem 0 .9rem; }
.marca span { color: #5C636B; font-weight: 400; }
.plazo.urgente .cifra { color: #5543D1; }
.plazo.urgente .texto { color: #5543D1; font-weight: 500; }
a.titulo { color: #1E2329; font-size: 1.05rem; font-weight: 500; line-height: 1.35;
  text-decoration: none; }
a.titulo:hover { text-decoration: underline; text-decoration-color: #5543D1; text-underline-offset: 3px; }
a.titulo:focus-visible { outline: 2px solid #5543D1; outline-offset: 2px; }
p.entidad { color: #5C636B; font-size: .9rem; margin: .25rem 0 .55rem; }
p.porque { font-size: .9rem; margin: 0 0 .2rem; max-width: 72ch; }
p.porque b { font-weight: 500; }
p.porque i { font-style: normal; color: #5543D1; }
p.cierre { color: #5C636B; font-size: .82rem; margin: 0; }
/* Botones de decisión: compactos, para que no pesen más que el concurso */
.stApp [data-testid="stHorizontalBlock"] .stButton button,
.stApp [data-testid="stLayoutWrapper"] .stButton button {
  min-height: 0; padding: .28rem .85rem; font-size: .84rem; border-radius: 6px; }
hr.fila { border: none; border-top: 1px solid #DEDBD3; margin: 1.1rem 0; }
.extra a { color: #1E2329; text-decoration: none; font-weight: 500; }
.extra a:hover { text-decoration: underline; text-decoration-color: #5543D1; text-underline-offset: 3px; }
.extra p { margin: 0 0 .8rem; font-size: .92rem; line-height: 1.4; }
p.resumen b { color: #1E2329; font-weight: 500; }
.extra span { color: #5C636B; font-size: .84rem; }
</style>
"""


def lineas_de_negocio() -> dict[str, str]:
    """Proyecto del portafolio → línea de negocio de ALBRAN."""
    from src.modelo.similitud import RUTA_PORTAFOLIO

    try:
        import pandas as pd

        datos = pd.read_csv(RUTA_PORTAFOLIO)
    except (FileNotFoundError, ValueError):
        return {}
    return dict(zip(datos["proyecto"], datos["linea_negocio"], strict=True))


def explicacion(rec: dict, lineas: dict[str, str]) -> str:
    """Por qué se recomienda: el proyecto parecido, su línea de negocio y los términos."""
    proyecto = rec["proyecto_similar"] or ""
    linea = lineas.get(proyecto, "")
    if proyecto == linea or not proyecto:
        texto = f"Encaja en la línea de <i>{html.escape(linea.lower())}</i>." if linea else ""
    else:
        texto = f"Se parece a <b>{html.escape(proyecto)}</b>"
        texto += f", de <i>{html.escape(linea.lower())}</i>." if linea else "."
    terminos = a_favor(rec["terminos_top"])
    if terminos:
        texto += f" Palabras que más pesaron: {html.escape(terminos)}."
    return texto.strip()


def fila_recomendacion(st, con, dia: date, rec: dict, lineas: dict[str, str]) -> None:
    cifra, texto, urgente = plazo(dia, rec["fecha_cierre"])
    url = URL_CONCURSO.format(nog=rec["nog"])
    cierre = f"Cierra el {fecha_larga(rec['fecha_cierre'])}." if rec["fecha_cierre"] else ""
    modalidad = modalidad_corta(rec["modalidad"])
    pie = " ".join(x for x in (f"{modalidad}." if modalidad else "", cierre) if x)

    izq, centro = st.columns([1, 8.2], gap="medium")
    izq.markdown(
        f'<div class="plazo{" urgente" if urgente else ""}"><span class="cifra">{cifra}</span>'
        f'<span class="texto">{texto}</span></div>',
        unsafe_allow_html=True,
    )
    centro.markdown(
        f'<a class="titulo" href="{url}" target="_blank" rel="noopener">'
        f"{html.escape(titulo_legible(rec['titulo']))}</a>"
        f'<p class="entidad">{html.escape(entidad_legible(rec["entidad"]))}</p>'
        f'<p class="porque">{explicacion(rec, lineas)}</p>'
        f'<p class="cierre">{pie}</p>',
        unsafe_allow_html=True,
    )
    # La decisión elegida queda resaltada; cambiarla guarda una nueva (vale la última)
    with centro.container(horizontal=True, gap="small"):
        for decision, etiqueta in (("me_interesa", "Me interesa"), ("no_aplica", "No aplica")):
            elegido = rec["decision"] == decision
            if (
                st.button(
                    etiqueta,
                    key=f"{decision}-{rec['ocid']}",
                    type="primary" if elegido else "secondary",
                )
                and not elegido
            ):
                decidir(con, rec["ocid"], decision)
                st.rerun()


def vista_recomendaciones() -> None:
    import streamlit as st

    st.markdown(ESTILO, unsafe_allow_html=True)
    cabecera(st, "Recomendaciones")
    con = conectar()
    try:
        dias = dias_disponibles(con)
        if not dias:
            st.markdown('<h1 class="radar">Recomendaciones</h1>', unsafe_allow_html=True)
            st.info(
                "Todavía no hay listas guardadas. Generalas con "
                "`python -m src.modelo.ranking --hasta AAAA-MM-DD`."
            )
            return
        por_dia = {d["dia"]: d for d in dias}
        dia = (
            st.segmented_control(
                "Día",
                list(por_dia),
                default=dias[0]["dia"],
                format_func=lambda d: f"{DIAS_SEMANA[d.weekday()][:3]} {d.day}/{d.month}",
                label_visibility="collapsed",
                key="dia",
            )
            or dias[0]["dia"]
        )
        info = por_dia[dia]
        lista = lista_guardada(con, dia)
        principal = [r for r in lista if r["seccion"] == "principal"]
        computo = [r for r in lista if r["seccion"] == "computo"]

        publicados = f"{info['publicados']:,}".replace(",", "\u202f")  # miles con espacio fino
        urgentes = sum(plazo(dia, r["fecha_cierre"])[2] for r in principal)
        resumen = f"Ese día se publicaron {publicados} procesos en Guatecompras."
        if principal:
            resumen += (
                f" Estos son los {len(principal)} que más se parecen a lo que hace ALBRAN,"
                " con el plazo que queda para ofertar."
            )
        if urgentes:
            resumen += (
                f" <b>{urgentes} {'cierra' if urgentes == 1 else 'cierran'} en tres días o"
                " menos.</b>"
            )
        st.markdown(
            f'<h1 class="radar">Concursos para el {fecha_larga(dia)}</h1>'
            f'<p class="resumen">{resumen}</p>',
            unsafe_allow_html=True,
        )
        lineas = lineas_de_negocio()
        if not principal:
            st.markdown(
                '<p class="resumen">Ese día no hubo concursos que el modelo considere '
                "relevantes para ALBRAN.</p>",
                unsafe_allow_html=True,
            )
        for i, rec in enumerate(principal):
            if i:
                st.markdown('<hr class="fila">', unsafe_allow_html=True)
            fila_recomendacion(st, con, dia, rec, lineas)

        if computo:
            st.markdown('<hr class="fila">', unsafe_allow_html=True)
            with st.expander(f"Otras {len(computo)} compras de computadoras de ese día"):
                partes = []
                for rec in computo:
                    cifra, texto, _ = plazo(dia, rec["fecha_cierre"])
                    cuando = f"{cifra} {texto}" if cifra != "—" else texto
                    partes.append(
                        f'<p><a href="{URL_CONCURSO.format(nog=rec["nog"])}" target="_blank" '
                        f'rel="noopener">{html.escape(titulo_legible(rec["titulo"]))}</a><br>'
                        f"<span>{html.escape(entidad_legible(rec['entidad']))}, {cuando}</span></p>"
                    )
                st.markdown(f'<div class="extra">{"".join(partes)}</div>', unsafe_allow_html=True)
    finally:
        con.close()


PAGINAS: dict = {}


def cabecera(st, actual: str) -> None:
    """Nombre del radar a la izquierda y el enlace a la otra vista a la derecha."""
    marca, enlace = st.columns([5, 1.3], vertical_alignment="center")
    marca.markdown(
        '<div class="marca">Radar de Licitaciones <span>para ALBRAN Industries</span></div>',
        unsafe_allow_html=True,
    )
    otra = "Etiquetado" if actual == "Recomendaciones" else "Recomendaciones"
    enlace.page_link(PAGINAS[otra], label=otra)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Radar de Licitaciones", layout="wide")
    PAGINAS["Recomendaciones"] = st.Page(
        vista_recomendaciones, title="Recomendaciones", default=True
    )
    PAGINAS["Etiquetado"] = st.Page(vista_etiquetado, title="Etiquetado", url_path="etiquetado")
    st.navigation(list(PAGINAS.values()), position="hidden").run()


if __name__ == "__main__":
    main()
