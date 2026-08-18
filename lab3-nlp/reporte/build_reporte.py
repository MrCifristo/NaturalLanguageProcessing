#!/usr/bin/env python
"""
Convierte el reporte en Markdown a un PDF con estilo académico.

Uso (con el Python de miniforge/base, que tiene weasyprint y markdown instalados):

    /opt/homebrew/Caskroom/miniforge/base/bin/python reporte/build_reporte.py

Genera: reporte/Reporte_Laboratorio3.pdf
"""

from pathlib import Path

import markdown
from weasyprint import HTML

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "reporte_laboratorio3.md"
PDF_PATH = BASE / "Reporte_Laboratorio3.pdf"

# CSS con estética académica: tipografía serif, márgenes, numeración de página,
# tablas con bordes y encabezados jerárquicos.
CSS = """
@page {
    size: Letter;
    margin: 1.7cm 2.1cm 1.5cm 2.1cm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-family: 'Georgia', serif;
        font-size: 8pt;
        color: #888;
    }
}
body {
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 9.1pt;
    line-height: 1.36;
    color: #22262a;
    text-align: left;
}
h1 {
    font-size: 16pt;
    color: #0b3d5c;
    text-align: center;
    margin: 0 0 0.1em 0;
    letter-spacing: -0.2px;
}
h2 {
    font-size: 11pt;
    color: #0b3d5c;
    margin: 0.95em 0 0.38em 0;
    padding-bottom: 0.25em;
    border-bottom: 1.5px solid #0b3d5c;
    page-break-after: avoid;
}
h3 {
    font-size: 9.6pt;
    color: #16679a;
    margin: 0.8em 0 0.2em 0;
    page-break-after: avoid;
}
p { margin: 0.45em 0; }
strong { color: #0b3d5c; }
ul { margin: 0.4em 0 0.7em 0; padding-left: 1.1em; }
li { margin: 0.22em 0; padding-left: 0.15em; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.65em 0;
    font-size: 8pt;
    page-break-inside: avoid;
}
th, td { padding: 3px 8px; text-align: left; }
th {
    color: #0b3d5c;
    font-weight: bold;
    border-bottom: 1.2px solid #0b3d5c;
    border-top: 1.2px solid #0b3d5c;
}
td { border-bottom: 0.5px solid #d8dee3; }
tr:last-child td { border-bottom: 1.2px solid #0b3d5c; }
td:not(:first-child), th:not(:first-child) { text-align: right; }
code {
    font-family: 'Menlo', 'Consolas', monospace;
    font-size: 8.2pt;
    color: #0f4c75;
    background: none;
}
img {
    max-width: 44%;
    display: block;
    margin: 0.4em auto 0.7em auto;
}
.subtitulo {
    text-align: center;
    color: #667;
    font-size: 10pt;
    margin: 0 0 1.1em 0;
    font-style: italic;
}
.datos {
    background: #eef3f7;
    border-left: 3px solid #16679a;
    padding: 0.1em 0.9em;
    margin: 1em 0;
}
.datos p { margin: 0.5em 0; }
hr {
    border: none;
    border-top: 1px solid #c9d3db;
    margin: 0.9em 0 0.5em 0;
}
.repo {
    text-align: center;
    font-size: 8.6pt;
    color: #555;
    line-height: 1.5;
}
.repo a { color: #16679a; text-decoration: none; }
"""


def main():
    md_text = MD_PATH.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "md_in_html"],
    )
    html_doc = f"<html><head><meta charset='utf-8'></head><body>{html_body}</body></html>"

    HTML(string=html_doc, base_url=str(BASE)).write_pdf(
        PDF_PATH, stylesheets=[__import__("weasyprint").CSS(string=CSS)]
    )
    print(f"PDF generado: {PDF_PATH}")


if __name__ == "__main__":
    main()
