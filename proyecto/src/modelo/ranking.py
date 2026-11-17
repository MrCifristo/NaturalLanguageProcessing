"""Ranking del día: candidatos vigentes, cinco métodos y top 10.

Métodos (docs/04, sección 5): las dos líneas base (palabra clave y categoría), solo el
clasificador, solo la similitud y el score combinado α · P + (1 − α) · similitud.

Como el score solo se usa para ordenar, no hace falta normalizar la similitud: escalarla
equivale a mover α, y sumarle una constante no cambia el orden.
"""

import re
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from src.limpieza.normalizar import poblar_textos, quitar_tildes
from src.modelo.clasificador import formato_terminos, puntuar, terminos

TOP = 10

# Pre-filtro original del deck (notebook 01, celda 3.9) y categorías de TI
CLAVES = re.compile(
    r"\b(?:software|sistemas?|plataformas?|aplicacion(?:es)?|desarrollo|licencias?"
    r"|informatic\w*|tecnologi\w*|digital\w*|web)\b"
)
CATEGORIA = re.compile(r"^(?:43|8111)")

METODOS = ["palabra_clave", "categoria", "clasificador", "similitud", "combinado"]


def candidatos(con: duckdb.DuckDBPyConnection, dia: date) -> pd.DataFrame:
    """Publicados ese día y con el periodo de ofertas abierto; sin fecha de cierre no entran."""
    ocids = [
        f[0]
        for f in con.execute(
            "SELECT ocid FROM procesos WHERE fecha_publicacion >= ? AND fecha_publicacion < ? "
            "AND fecha_cierre > fecha_publicacion",
            [dia, dia + timedelta(days=1)],
        ).fetchall()
    ]
    poblar_textos(con, ocids)
    con.register("_dia", pd.DataFrame({"ocid": ocids}))
    datos = con.execute(
        "SELECT p.ocid, p.nog, p.titulo, p.items_texto, p.unspsc, p.entidad, p.fecha_cierre, "
        "p.texto_tfidf, p.texto_embedding FROM procesos p JOIN _dia USING (ocid) ORDER BY p.ocid"
    ).df()
    con.unregister("_dia")
    return datos


def puntaje_palabra_clave(datos: pd.DataFrame) -> np.ndarray:
    """Cantidad de palabras clave distintas en título e ítems; 0 = no pasa el filtro."""
    textos = (datos["titulo"].fillna("") + " " + datos["items_texto"].fillna("")).str.lower()
    return np.array([len(set(CLAVES.findall(quitar_tildes(t)))) for t in textos], dtype=float)


def puntaje_categoria(datos: pd.DataFrame) -> np.ndarray:
    """Fracción de códigos UNSPSC del proceso que son de TI; 0 = no pasa el filtro."""

    def fraccion(unspsc: str | None) -> float:
        codigos = [c for c in (unspsc or "").split(",") if c]
        return sum(bool(CATEGORIA.match(c)) for c in codigos) / len(codigos) if codigos else 0.0

    # fillna: un NULL de DuckDB llega como NaN, que es verdadero y no se puede partir
    return np.array([fraccion(u) for u in datos["unspsc"].fillna("")], dtype=float)


def puntajes(
    datos: pd.DataFrame,
    modelo: Pipeline,
    alfa: float,
    portafolio: tuple[pd.DataFrame, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Agrega una columna por método (y proyecto_similar). Sin portafolio, sin similitud."""
    salida = datos.copy()
    salida["palabra_clave"] = puntaje_palabra_clave(datos)
    salida["categoria"] = puntaje_categoria(datos)
    salida["clasificador"] = puntuar(modelo, datos["texto_tfidf"]) if len(datos) else []
    if portafolio is not None and len(datos):
        from src.modelo.similitud import embeber, similitud

        proyectos, emb_portafolio = portafolio
        sim, similar = similitud(embeber(datos["texto_embedding"]), proyectos, emb_portafolio)
        salida["similitud"] = sim
        salida["proyecto_similar"] = similar
        salida["combinado"] = alfa * salida["clasificador"] + (1 - alfa) * salida["similitud"]
    return salida


# Venta de computadoras: título con el equipo o UNSPSC de la familia 432115 (computadores),
# sin 43211501 (servidores), que no cuentan para el tope
COMPUTO = re.compile(
    r"\b(?:computadora|computador|portatil(?:es)?|laptops?|estacion(?:es)? de trabajo|cpu"
    r"|unidad(?:es)? de procesamiento|todo en uno|all in one)\b"
)
UNSPSC_COMPUTO = re.compile(r"(?:^|,)432115(?!01)")

# Tope de computadoras en la lista del día (D-16): solo en el combinado, que es el producto
MAX_COMPUTO = {"combinado": 5}


def es_venta_computo(datos: pd.DataFrame) -> np.ndarray:
    titulos = datos["titulo"].fillna("").str.lower().map(quitar_tildes)
    por_titulo = titulos.str.contains(COMPUTO)
    por_unspsc = datos["unspsc"].fillna("").str.contains(UNSPSC_COMPUTO)
    return (por_titulo | por_unspsc).to_numpy()


def top(
    puntuados: pd.DataFrame, metodo: str, n: int = TOP, max_computo: int | None = -1
) -> pd.DataFrame:
    """Los n de mayor puntaje; en las líneas base, solo los que pasan el filtro (puntaje > 0).

    max_computo limita cuántas ventas de computadoras entran; los lugares que sobran pasan a
    los siguientes por puntaje. -1 usa el tope del método (MAX_COMPUTO) y None no limita.
    """
    if metodo not in puntuados:
        return puntuados.iloc[0:0]
    datos = puntuados
    if metodo in ("palabra_clave", "categoria"):
        datos = datos[datos[metodo] > 0]
    # Empates: por ocid, para que el top sea reproducible
    datos = datos.sort_values([metodo, "ocid"], ascending=[False, True])
    if max_computo == -1:
        max_computo = MAX_COMPUTO.get(metodo)
    if max_computo is not None:
        computo = es_venta_computo(datos)
        datos = datos[~computo | (np.cumsum(computo) <= max_computo)]
    return datos.head(n)


def ajustar_alfa(
    p_validacion: np.ndarray, sim_validacion: np.ndarray, y: np.ndarray, paso: float = 0.05
) -> tuple[float, float]:
    """α que maximiza el AUC-PR en validación; devuelve (α, AUC-PR).

    docs/04 pide precisión@10 por día, pero la validación es una muestra de 150 procesos de
    seis meses, no días completos; el AUC-PR mide el mismo orden sobre todo el rango.
    """
    mejores = [
        (average_precision_score(y, a * p_validacion + (1 - a) * sim_validacion), a)
        for a in np.arange(0, 1 + paso / 2, paso)
    ]
    auc, alfa = max(mejores)
    return round(float(alfa), 2), float(auc)


def explicar(modelo: Pipeline, puntuados: pd.DataFrame) -> list[str]:
    """Términos a favor y en contra de cada fila (para el panel y el correo)."""
    return [formato_terminos(f, c) for f, c in terminos(modelo, puntuados["texto_tfidf"].tolist())]


def guardar_predicciones(
    con: duckdb.DuckDBPyConnection,
    dia: date,
    seleccion: pd.DataFrame,
    modelo: Pipeline,
    version: str,
) -> None:
    """Escribe el top del score combinado en `predicciones`; repetir el día lo reemplaza."""
    filas = seleccion.assign(
        fecha_corrida=dia,
        version_modelo=version,
        terminos_top=explicar(modelo, seleccion),
        rango=range(1, len(seleccion) + 1),
    )
    con.execute(
        "DELETE FROM predicciones WHERE fecha_corrida = ? AND version_modelo = ?", [dia, version]
    )
    con.register("_pred", filas)
    con.execute(
        """
        INSERT INTO predicciones
        SELECT ocid, fecha_corrida, version_modelo, clasificador, similitud, proyecto_similar,
               combinado, terminos_top, rango
        FROM _pred
        """
    )
    con.unregister("_pred")


def alfa_validacion(
    con: duckdb.DuckDBPyConnection, modelo: Pipeline, portafolio: tuple[pd.DataFrame, np.ndarray]
) -> tuple[float, float]:
    """Ajusta α con las etiquetas de validación; devuelve (α, AUC-PR del combinado)."""
    from src.modelo.similitud import embeber, similitud

    ocids = [
        f[0]
        for f in con.execute(
            "SELECT ocid FROM etiquetas WHERE conjunto = 'validacion' AND etiqueta_humana IS NOT NULL"
        ).fetchall()
    ]
    poblar_textos(con, ocids)
    datos = con.execute(
        "SELECT p.texto_tfidf, p.texto_embedding, e.etiqueta_humana AS y "
        "FROM etiquetas e JOIN procesos p USING (ocid) "
        "WHERE e.conjunto = 'validacion' AND e.etiqueta_humana IS NOT NULL ORDER BY e.ocid"
    ).df()
    proyectos, emb_portafolio = portafolio
    sim, _ = similitud(embeber(datos["texto_embedding"]), proyectos, emb_portafolio)
    return ajustar_alfa(puntuar(modelo, datos["texto_tfidf"]), sim, datos["y"].astype(int).values)
