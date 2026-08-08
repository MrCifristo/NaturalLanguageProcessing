#!/usr/bin/env python
"""
Convierte el reporte en Markdown a un PDF con estilo académico.

Uso (con el Python de miniforge/base, que tiene weasyprint y markdown instalados):

    /opt/homebrew/Caskroom/miniforge/base/bin/python reporte/build_reporte.py

Genera: reporte/Reporte_Laboratorio1.pdf
"""

from pathlib import Path

import markdown
from weasyprint import HTML

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "reporte_laboratorio1.md"
PDF_PATH = BASE / "Reporte_Laboratorio1.pdf"

# CSS con estética académica: tipografía serif, márgenes, numeración de página,
# tablas con bordes y encabezados jerárquicos.
CSS = """
@page {
    size: Letter;
    margin: 1.8cm 2cm 1.6cm 2cm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-family: 'Georgia', serif;
        font-size: 8pt;
        color: #666;
    }
}
body {
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 9.5pt;
    line-height: 1.32;
    color: #1a1a1a;
    text-align: justify;
}
h1 {
    font-size: 15pt;
    color: #0b3d5c;
    text-align: center;
    margin-bottom: 0.2em;
    border-bottom: 2px solid #0b3d5c;
    padding-bottom: 0.2em;
}
h2 {
    font-size: 11.5pt;
    color: #0b3d5c;
    margin-top: 0.9em;
    margin-bottom: 0.3em;
    border-bottom: 1px solid #cccccc;
    padding-bottom: 0.1em;
    page-break-after: avoid;
}
h3 {
    font-size: 10pt;
    color: #16679a;
    margin-top: 0.6em;
    margin-bottom: 0.2em;
    page-break-after: avoid;
}
p { margin: 0.35em 0; }
ul, ol { margin: 0.3em 0 0.3em 0.4em; }
li { margin: 0.12em 0; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.5em 0;
    font-size: 8.5pt;
}
th, td {
    border: 1px solid #999;
    padding: 3px 6px;
    text-align: left;
}
th { background: #0b3d5c; color: #fff; }
tr:nth-child(even) td { background: #f2f6f9; }
code {
    font-family: 'Menlo', 'Consolas', monospace;
    font-size: 8.5pt;
    background: #eef1f3;
    padding: 1px 3px;
    border-radius: 3px;
}
blockquote {
    border-left: 3px solid #16679a;
    margin: 0.5em 0;
    padding: 0.2em 0.8em;
    background: #f2f6f9;
    color: #333;
    font-size: 8.5pt;
}
img { max-width: 100%; display: block; margin: 0.8em auto; }
.subtitulo { text-align: center; color: #555; font-size: 11pt; margin-top: 0; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.2em 0; }
"""


def main():
    md_text = MD_PATH.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    html_doc = f"<html><head><meta charset='utf-8'></head><body>{html_body}</body></html>"

    HTML(string=html_doc, base_url=str(BASE)).write_pdf(
        PDF_PATH, stylesheets=[__import__("weasyprint").CSS(string=CSS)]
    )
    print(f"PDF generado: {PDF_PATH}")


if __name__ == "__main__":
    main()
