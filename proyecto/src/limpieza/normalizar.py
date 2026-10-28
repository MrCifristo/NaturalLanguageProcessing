"""Texto de entrada de cada proceso y sus dos representaciones (TF-IDF y embeddings)."""

import html
import re
import unicodedata
from pathlib import Path

# Códigos internos que las entidades pegan al título; no aportan significado
PATRONES_RUIDO = [
    r"\bc[oó]d(?:igo)?\.?\s*(?:provisional\s+)?igss(?:\s+provisional)?\s*:?\s*[\w/.-]+",  # CODIGO IGSS: 227281
    r"\bpedido\s*:?\s*(?:bm\s+)?(?:no\.?\s*)?[\w/-]*\d[\w/-]*",       # PEDIDO No. 38811/2026, PEDIDO ADM0794/2026
    r"\bevento\s*(?:no\.?)?\s*:?\s*[\d/-]+",                        # EVENTO 1438/2026
    r"\bamparo\s+no\.?\s*[\d-]+",                                     # AMPARO No. 02075-2026-00197
    r"\b[a-z]{2,6}\s+no\.\s*[\w/-]*\d[\w/-]*",                        # MSPAS No. 2026-ROO-CD-6389, EVENTO NO. 1279/2025
    r"\b(?:ppr|solicitud|c[oó]digo|referencia)\s*(?:no\.?)?\s*:?\s*\d[\d/-]*",  # PPR: 69706-83245, CODIGO 435
    r"\b\d+/20\d\d\b",                                                # 1679/2025 suelto
    r"\[[\w-]+\]",                                                    # [DA-DAC-A18][CD-271]
    r"\bp[-\s]*\d[\d,]{3,}(?:/\d{4})?",                               # P-11111, P 10026, P- 3,793/2026
    r"\bnog\s*:?\s*\d+",
]
RUIDO = re.compile("|".join(PATRONES_RUIDO), flags=re.IGNORECASE)
HTML = re.compile(r"<[^>]+>")
# Separadores que quedan huérfanos al quitar un código: "; ;" o un ";" al inicio
SEPARADORES_SEGUIDOS = re.compile(r"([;:.,])(?:\s*[;:.,])+")
SEPARADORES_INICIO = re.compile(r"^[\s;:.,]+")

# Frases plantilla que remiten a las bases en vez de describir la compra
FRASES_PLANTILLA = re.compile(
    r"seg[uú]n especificaciones t[eé]cnicas(?: adjuntas)?|ver (?:las )?bases|leer bases"
    r"|ver t[eé]rminos de referencia|ver especificaciones t[eé]cnicas",
    flags=re.IGNORECASE,
)

# Campos del formulario del oferente (IVA, NIT, tiempo de entrega…) que las entidades
# cargan como características. La lista sale de los datos: notebook 01, celda 3.8 (D-07)
RUTA_CAMPOS = Path(__file__).with_name("campos_formulario.txt")
_campos: set[str] | None = None


def campos_formulario() -> set[str]:
    global _campos
    if _campos is None:
        _campos = set(RUTA_CAMPOS.read_text(encoding="utf-8").splitlines()) if RUTA_CAMPOS.exists() else set()
    return _campos


# Tope de caracteres: hay procesos con cientos de ítems y el modelo de embeddings corta en 128 tokens
MAX_CARACTERES = 1000


def texto_entrada(titulo: str, items_texto: str | None, caracteristicas: str | None) -> str:
    # Se omiten los fragmentos que ya están dentro del título (ítem copiado del título)
    vistos = titulo.lower()
    partes = [titulo]
    for campo in (items_texto, caracteristicas):
        for frag in (campo or "").split(" | "):
            if frag and frag.lower() not in vistos and frag.strip().lower() not in campos_formulario():
                partes.append(frag)
                vistos += " " + frag.lower()
    return " | ".join(partes)[:MAX_CARACTERES]


def quitar_ruido(texto: str) -> str:
    texto = HTML.sub(" ", html.unescape(texto))
    texto = RUIDO.sub(" ", texto)
    texto = FRASES_PLANTILLA.sub(" ", texto)
    texto = "".join(c if unicodedata.category(c)[0] != "C" else " " for c in texto)  # caracteres de control
    texto = SEPARADORES_SEGUIDOS.sub(r"\1", re.sub(r"\s+", " ", texto))
    return SEPARADORES_INICIO.sub("", texto).strip()


def quitar_tildes(texto: str) -> str:
    # La ñ se conserva: "año" y "ano" no son la misma palabra
    texto = texto.replace("ñ", "\0")
    texto = "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")
    return texto.replace("\0", "ñ")


def para_embedding(texto: str) -> str:
    """Casi original: el modelo preentrenado aprovecha tildes, mayúsculas y contexto."""
    return quitar_ruido(texto)


_nlp = None


def _spacy():
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
    return _nlp


def para_tfidf(textos: list[str], batch_size: int = 2000) -> list[str]:
    """Minúsculas, lemas de spaCy, sin stopwords, sin números ni puntuación, sin tildes.

    Se pasa a minúsculas antes de spaCy porque muchos títulos vienen en mayúsculas
    y el modelo pequeño lematiza mal las palabras en mayúsculas.
    """
    limpios = (quitar_ruido(t).lower() for t in textos)
    salida = []
    for doc in _spacy().pipe(limpios, batch_size=batch_size):
        lemas = [
            quitar_tildes(tok.lemma_)
            for tok in doc
            if tok.is_alpha and not tok.is_stop and len(tok) > 1
        ]
        salida.append(" ".join(lemas))
    return salida
