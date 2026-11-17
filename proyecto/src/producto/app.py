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
    banda(st, "Etiquetado", "Etiquetado")
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
.stApp, .stApp p, .stApp h1, .stApp h2, .stApp button p, .stApp label, .stApp summary p,
.stApp a, .stApp span:not([data-testid="stIconMaterial"]) {
  font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #F3F2EE; overflow-x: hidden; }
header[data-testid="stHeader"] { display: none; }
.block-container, [data-testid="stMainBlockContainer"] { max-width: 1120px;
  padding-top: 0 !important; padding-bottom: 4rem; }

/* Banda oscura de arriba: fondo a todo el ancho con el truco de la sombra recortada */
.banda { background: #14171B; color: #F3F2EE; box-shadow: 0 0 0 100vmax #14171B;
  clip-path: inset(-100vmax -100vmax 0); padding: 26px 0 30px; margin-bottom: 30px; }
.banda .arriba { display: flex; justify-content: space-between; align-items: baseline; }
.banda .marca { font-size: .95rem; font-weight: 600; letter-spacing: -0.005em; }
.banda .marca span { color: #8E959E; font-weight: 400; }
.banda a.vista { color: #B4BBC4; font-size: .9rem; text-decoration: none; }
.banda a.vista:hover { color: #F3F2EE; }
.banda h1 { font-size: 1.9rem; font-weight: 600; letter-spacing: -0.02em; margin: 22px 0 18px;
  color: #F3F2EE; }
.dias { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
a.dia { display: block; background: #1E2329; border-radius: 10px; padding: 14px 16px;
  text-decoration: none; color: #F3F2EE; border: 1px solid #2A3038; }
a.dia:hover { border-color: #5C636B; }
a.dia .nombre { font-size: .8rem; color: #8E959E; }
a.dia .numero { font-size: 1.7rem; font-weight: 600; line-height: 1.1; margin: 2px 0 6px;
  font-variant-numeric: tabular-nums; }
a.dia .cuenta { font-size: .8rem; color: #B4BBC4; line-height: 1.35; }
a.dia.activo { background: #9A8CFF; border-color: #9A8CFF; color: #14171B; }
a.dia.activo .nombre, a.dia.activo .cuenta { color: #2B2470; }
a.dia:focus-visible { outline: 2px solid #9A8CFF; outline-offset: 2px; }

p.resumen { color: #4A4F57; font-size: 1rem; max-width: 70ch; margin: 0 0 6px; }
p.resumen b { color: #1E2329; font-weight: 600; }

/* Tarjeta de cada concurso */
[class*="st-key-rec-"] { background: #FBFBF8; border: 1px solid #E2DFD7; border-radius: 14px;
  padding: 20px 22px 16px; }
.plazo { border-radius: 10px; padding: 14px 8px 12px; text-align: center; line-height: 1; }
.plazo .cifra { display: block; font-size: 2.1rem; font-weight: 600;
  font-variant-numeric: tabular-nums; letter-spacing: -0.03em; }
.plazo .texto { display: block; font-size: .72rem; margin-top: 6px; line-height: 1.25; }
.plazo.urgente { background: #5543D1; color: #F3F2EE; }
.plazo.fecha { background: #E4E0FA; color: #2B2470; }
.plazo.sin { background: #ECEAE4; color: #5C636B; }
.cabeza { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 6px; }
span.linea { background: #E4E0FA; color: #3B2FA0; font-size: .74rem; font-weight: 500;
  padding: 3px 10px; border-radius: 999px; }
span.linea.computo { background: #ECEAE4; color: #4A4F57; }
span.entidad { color: #5C636B; font-size: .85rem; }
a.titulo { color: #1E2329; font-size: 1.08rem; font-weight: 600; line-height: 1.35;
  text-decoration: none; letter-spacing: -0.005em; }
a.titulo:hover { color: #5543D1; }
a.titulo:focus-visible { outline: 2px solid #5543D1; outline-offset: 2px; }
p.porque { font-size: .9rem; color: #2E333A; margin: 8px 0 4px; max-width: 78ch; }
p.porque b { font-weight: 600; }
p.porque i { font-style: normal; color: #5543D1; }
p.cierre { color: #5C636B; font-size: .8rem; margin: 0; }
[class*="st-key-rec-"] .stButton button { min-height: 0; padding: .25rem .9rem;
  font-size: .82rem; border-radius: 999px; }

.extra a { color: #1E2329; text-decoration: none; font-weight: 500; }
.extra a:hover { color: #5543D1; }
.extra p { margin: 0 0 .8rem; font-size: .92rem; line-height: 1.4; }
.extra span { color: #5C636B; font-size: .84rem; }
p.pie { color: #8A8F96; font-size: .78rem; margin-top: 2.5rem; }
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


def clase_plazo(dia: date, cierre: datetime | None) -> str:
    if cierre is None:
        return "sin"
    return "urgente" if plazo(dia, cierre)[2] else "fecha"


def ordenar(lista: list[dict], criterio: str) -> list[dict]:
    """Relevancia (el orden del ranking) o cierre más próximo; sin fecha van al final."""
    if criterio == "Cierre más próximo":
        return sorted(lista, key=lambda r: (r["fecha_cierre"] is None, r["fecha_cierre"] or 0))
    return sorted(lista, key=lambda r: r["rango"])


def banda(st, actual: str, titulo: str = "", dias_html: str = "") -> None:
    """Cabecera oscura: el nombre, el enlace a la otra vista y, si se da, la semana."""
    otra, ruta = (
        ("Etiquetado", "etiquetado")
        if actual == "Recomendaciones"
        else (
            "Recomendaciones",
            "",
        )
    )
    st.markdown(
        '<div class="banda"><div class="arriba">'
        '<div class="marca">Radar de Licitaciones <span>para ALBRAN Industries</span></div>'
        f'<a class="vista" href="/{ruta}" target="_self">{otra}</a></div>'
        + (f"<h1>{titulo}</h1>" if titulo else "")
        + dias_html
        + "</div>",
        unsafe_allow_html=True,
    )


def tira_de_dias(dias: list[dict], elegido: date, urgentes: dict[date, int]) -> str:
    tarjetas = []
    for d in sorted(dias, key=lambda x: x["dia"]):
        dia = d["dia"]
        n = d["recomendados"]
        cuenta = f"{n} en la lista" if n else "sin concursos"
        if urgentes.get(dia):
            n_urg = urgentes[dia]
            cuenta += f"<br>{n_urg} {'cierra' if n_urg == 1 else 'cierran'} pronto"
        tarjetas.append(
            f'<a class="dia{" activo" if dia == elegido else ""}" href="?dia={dia:%Y-%m-%d}" '
            f'target="_self"><div class="nombre">{DIAS_SEMANA[dia.weekday()]}</div>'
            f'<div class="numero">{dia.day} {MESES[dia.month - 1][:3]}</div>'
            f'<div class="cuenta">{cuenta}</div></a>'
        )
    return f'<div class="dias">{"".join(tarjetas)}</div>'


def tarjeta(st, con, dia: date, rec: dict, lineas: dict[str, str]) -> None:
    cifra, texto, _ = plazo(dia, rec["fecha_cierre"])
    url = URL_CONCURSO.format(nog=rec["nog"])
    cierre = f"Cierra el {fecha_larga(rec['fecha_cierre'])}." if rec["fecha_cierre"] else ""
    modalidad = modalidad_corta(rec["modalidad"])
    pie = " ".join(x for x in (f"{modalidad}." if modalidad else "", cierre) if x)
    linea = lineas.get(rec["proyecto_similar"] or "", "")
    clase_linea = "linea computo" if "cómputo" in linea else "linea"

    with st.container(key=f"rec-{rec['ocid']}"):
        izq, centro = st.columns([1.05, 8], gap="medium")
        izq.markdown(
            f'<div class="plazo {clase_plazo(dia, rec["fecha_cierre"])}">'
            f'<span class="cifra">{cifra}</span><span class="texto">{texto}</span></div>',
            unsafe_allow_html=True,
        )
        with centro:
            st.markdown(
                '<div class="cabeza">'
                + (f'<span class="{clase_linea}">{html.escape(linea)}</span>' if linea else "")
                + f'<span class="entidad">{html.escape(entidad_legible(rec["entidad"]))}</span>'
                "</div>"
                f'<a class="titulo" href="{url}" target="_blank" rel="noopener">'
                f"{html.escape(titulo_legible(rec['titulo']))}</a>"
                f'<p class="porque">{explicacion(rec, lineas)}</p>'
                f'<p class="cierre">{pie}</p>',
                unsafe_allow_html=True,
            )
            # La decisión elegida queda resaltada; cambiarla guarda una nueva (vale la última)
            with st.container(horizontal=True, gap="small"):
                for decision, etiqueta in (
                    ("me_interesa", "Me interesa"),
                    ("no_aplica", "No aplica"),
                ):
                    elegido = rec["decision"] == decision
                    pulsado = st.button(
                        etiqueta,
                        key=f"{decision}-{rec['ocid']}",
                        type="primary" if elegido else "secondary",
                    )
                    if pulsado and not elegido:
                        decidir(con, rec["ocid"], decision)
                        st.rerun()


def vista_recomendaciones() -> None:
    import streamlit as st

    st.markdown(ESTILO, unsafe_allow_html=True)
    con = conectar()
    try:
        dias = dias_disponibles(con)
        if not dias:
            banda(st, "Recomendaciones", "Recomendaciones")
            st.info(
                "Todavía no hay listas guardadas. Generalas con "
                "`python -m src.modelo.ranking --hasta AAAA-MM-DD`."
            )
            return
        por_dia = {d["dia"]: d for d in dias}
        pedido = st.query_params.get("dia")
        try:
            dia = date.fromisoformat(pedido) if pedido else dias[0]["dia"]
        except ValueError:
            dia = dias[0]["dia"]
        if dia not in por_dia:
            dia = dias[0]["dia"]

        listas = {d: lista_guardada(con, d) for d in por_dia}
        urgentes = {
            d: sum(r["seccion"] == "principal" and plazo(d, r["fecha_cierre"])[2] for r in lista)
            for d, lista in listas.items()
        }
        lista = listas[dia]
        principal = [r for r in lista if r["seccion"] == "principal"]
        computo = [r for r in lista if r["seccion"] == "computo"]
        lineas = lineas_de_negocio()

        primero, ultimo = min(por_dia), max(por_dia)
        semana = (
            f"Semana del {primero.day} al {ultimo.day} de {MESES[ultimo.month - 1]}"
            if primero.month == ultimo.month
            else f"Del {fecha_larga(primero)} al {fecha_larga(ultimo)}"
        )
        banda(st, "Recomendaciones", semana, tira_de_dias(dias, dia, urgentes))

        publicados = f"{por_dia[dia]['publicados']:,}".replace(",", " ")
        marcados = sum(r["decision"] is not None for r in principal)
        resumen = (
            f"El <b>{fecha_larga(dia)}</b> se publicaron {publicados} procesos en Guatecompras."
        )
        if principal:
            resumen += (
                f" Estos son los {len(principal)} que más se parecen a lo que hace ALBRAN."
                f" Ya decidiste {marcados} de {len(principal)}."
            )
        else:
            resumen += " Ninguno parece relevante para ALBRAN."
        st.markdown(f'<p class="resumen">{resumen}</p>', unsafe_allow_html=True)

        if principal:
            presentes = sorted(
                {lineas.get(r["proyecto_similar"] or "", "") for r in principal} - {""}
            )
            elegidas = st.pills(
                "Línea de negocio", presentes, selection_mode="multi", key=f"lineas-{dia}"
            )
            criterio = (
                st.segmented_control(
                    "Ordenar por",
                    ["Relevancia", "Cierre más próximo"],
                    default="Relevancia",
                    key="orden",
                )
                or "Relevancia"
            )
            visibles = [
                r
                for r in ordenar(principal, criterio)
                if not elegidas or lineas.get(r["proyecto_similar"] or "", "") in elegidas
            ]
            for rec in visibles:
                tarjeta(st, con, dia, rec, lineas)
            if not visibles:
                st.caption("Ningún concurso de ese día es de las líneas elegidas.")

        if computo:
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

        st.markdown(
            '<p class="pie">Datos abiertos del OCDS de Guatecompras, Ministerio de Finanzas '
            "Públicas (CC BY 4.0). El título de cada concurso lleva a su página en el portal.</p>",
            unsafe_allow_html=True,
        )
    finally:
        con.close()


PAGINAS: dict = {}


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
