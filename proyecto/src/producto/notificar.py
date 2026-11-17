"""Resumen diario por correo: la lista del día en HTML, enviada por SMTP (D-09).

Con Resend el servidor es smtp.resend.com, el usuario es «resend» y la clave es la API key.

    python -m src.producto.notificar                    # último día con lista; solo genera el HTML
    python -m src.producto.notificar --dia 2026-10-09 --enviar

Sin --enviar no hace falta .env: el HTML queda en datos/correos/ para revisarlo o mostrarlo.
"""

import argparse
import html
import os
import smtplib
import ssl
from dataclasses import dataclass
from datetime import date
from email.message import EmailMessage
from pathlib import Path

import duckdb

from src.db import conectar
from src.producto.app import (
    URL_CONCURSO,
    entidad_legible,
    explicacion,
    fecha_larga,
    lineas_de_negocio,
    lista_guardada,
    plazo,
    titulo_legible,
)

RAIZ = Path(__file__).resolve().parents[2]
CARPETA_CORREOS = RAIZ / "datos" / "correos"
VARIABLES = [
    "SMTP_SERVIDOR",
    "SMTP_PUERTO",
    "SMTP_USUARIO",
    "SMTP_CLAVE_APP",
    "CORREO_REMITENTE",
    "CORREO_DESTINO",
]

# Paleta del panel; el correo lleva estilos en línea porque los clientes ignoran <style>
TINTA, SUAVE, ACENTO, FONDO, LINEA = "#1E2329", "#5C636B", "#5543D1", "#F3F2EE", "#DEDBD3"


@dataclass
class ConfigCorreo:
    servidor: str
    puerto: int
    usuario: str
    clave: str
    remitente: str
    destinos: list[str]


def leer_config(entorno: dict[str, str] | None = None) -> ConfigCorreo:
    """Lee la configuración de .env (o del diccionario dado); avisa qué variables faltan."""
    if entorno is None:
        from dotenv import dotenv_values

        entorno = {**dotenv_values(RAIZ / ".env"), **os.environ}
    faltan = [v for v in VARIABLES if not entorno.get(v)]
    if faltan:
        raise ValueError(f"Faltan en .env: {', '.join(faltan)} (ver .env.example)")
    return ConfigCorreo(
        servidor=entorno["SMTP_SERVIDOR"],
        puerto=int(entorno["SMTP_PUERTO"]),
        usuario=entorno["SMTP_USUARIO"],
        clave=entorno["SMTP_CLAVE_APP"],
        remitente=entorno["CORREO_REMITENTE"],
        destinos=[d.strip() for d in entorno["CORREO_DESTINO"].split(",") if d.strip()],
    )


def ultimo_dia(con: duckdb.DuckDBPyConnection) -> date | None:
    return con.execute("SELECT max(fecha_corrida) FROM predicciones").fetchall()[0][0]


def _fila_html(dia: date, rec: dict, lineas: dict[str, str]) -> str:
    cifra, texto, urgente = plazo(dia, rec["fecha_cierre"])
    color = ACENTO if urgente else TINTA
    url = URL_CONCURSO.format(nog=rec["nog"])
    cierre = f"Cierra el {fecha_larga(rec['fecha_cierre'])}." if rec["fecha_cierre"] else ""
    return f"""
<tr><td style="padding:18px 0;border-top:1px solid {LINEA};vertical-align:top;width:84px">
  <div style="font-size:30px;font-weight:600;color:{color};line-height:1">{cifra}</div>
  <div style="font-size:12px;color:{color if urgente else SUAVE};margin-top:4px">{texto}</div>
</td><td style="padding:18px 0 18px 16px;border-top:1px solid {LINEA};vertical-align:top">
  <a href="{url}" style="color:{TINTA};font-size:16px;font-weight:600;text-decoration:none">
    {html.escape(titulo_legible(rec["titulo"]))}</a>
  <div style="color:{SUAVE};font-size:14px;margin:4px 0 8px">
    {html.escape(entidad_legible(rec["entidad"]))}</div>
  <div style="font-size:14px;color:{TINTA}">{explicacion(rec, lineas)}</div>
  <div style="font-size:13px;color:{SUAVE};margin-top:6px">{cierre}</div>
</td></tr>"""


def resumen(con: duckdb.DuckDBPyConnection, dia: date) -> tuple[str, str, str]:
    """(asunto, html, texto plano) del resumen del día."""
    lista = lista_guardada(con, dia)
    principal = [r for r in lista if r["seccion"] == "principal"]
    computo = [r for r in lista if r["seccion"] == "computo"]
    lineas = lineas_de_negocio()

    asunto = f"Radar de Licitaciones: {len(principal)} concursos para el {fecha_larga(dia)}"
    if not principal:
        asunto = f"Radar de Licitaciones: sin concursos relevantes el {fecha_larga(dia)}"
    filas = "".join(_fila_html(dia, r, lineas) for r in principal)
    extra = (
        f'<p style="font-size:14px;color:{SUAVE};margin-top:20px">Además hubo {len(computo)} '
        "compras de computadoras que no entraron en la lista. Están en el panel.</p>"
        if computo
        else ""
    )
    vacio = (
        ""
        if principal
        else f'<p style="font-size:15px;color:{TINTA}">Ese día no hubo '
        "concursos que el modelo considere relevantes para ALBRAN.</p>"
    )
    cuerpo = f"""<!doctype html><html><body style="margin:0;background:{FONDO}">
<div style="max-width:640px;margin:0 auto;padding:32px 24px;font-family:'IBM Plex Sans',Arial,sans-serif;color:{TINTA}">
<div style="font-size:13px;color:{SUAVE}">Radar de Licitaciones para ALBRAN Industries</div>
<h1 style="font-size:24px;margin:8px 0 6px">Concursos para el {fecha_larga(dia)}</h1>
<p style="font-size:15px;color:{SUAVE};margin:0 0 16px">Los que más se parecen a lo que hace
ALBRAN, con el plazo que queda para ofertar.</p>
{vacio}<table style="width:100%;border-collapse:collapse">{filas}</table>{extra}
<p style="font-size:12px;color:{SUAVE};margin-top:28px">Fuente: OCDS de Guatecompras, Ministerio
de Finanzas Públicas (CC BY 4.0).</p>
</div></body></html>"""

    texto = [asunto, ""]
    for r in principal:
        cifra, que, _ = plazo(dia, r["fecha_cierre"])
        texto += [
            f"- {titulo_legible(r['titulo'])}",
            f"  {entidad_legible(r['entidad'])}. {cifra} {que}.",
            f"  {URL_CONCURSO.format(nog=r['nog'])}",
        ]
    return asunto, cuerpo, "\n".join(texto)


def enviar(config: ConfigCorreo, asunto: str, cuerpo_html: str, texto: str) -> None:
    """Envía por SMTP: SSL directo en el puerto 465, STARTTLS en los demás."""
    mensaje = EmailMessage()
    mensaje["Subject"] = asunto
    mensaje["From"] = config.remitente
    mensaje["To"] = ", ".join(config.destinos)
    mensaje.set_content(texto)
    mensaje.add_alternative(cuerpo_html, subtype="html")
    contexto = ssl.create_default_context()
    if config.puerto == 465:
        with smtplib.SMTP_SSL(config.servidor, config.puerto, context=contexto, timeout=30) as s:
            s.login(config.usuario, config.clave)
            s.send_message(mensaje)
    else:
        with smtplib.SMTP(config.servidor, config.puerto, timeout=30) as s:
            s.starttls(context=contexto)
            s.login(config.usuario, config.clave)
            s.send_message(mensaje)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dia", type=date.fromisoformat, help="AAAA-MM-DD; por defecto el último")
    p.add_argument("--enviar", action="store_true", help="enviar por SMTP además de guardar")
    args = p.parse_args()

    con = conectar(solo_lectura=True)
    try:
        dia = args.dia or ultimo_dia(con)
        if dia is None:
            print("No hay listas guardadas. Generalas con `python -m src.modelo.ranking`.")
            return
        asunto, cuerpo, texto = resumen(con, dia)
    finally:
        con.close()

    CARPETA_CORREOS.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_CORREOS / f"resumen_{dia:%Y-%m-%d}.html"
    ruta.write_text(cuerpo, encoding="utf-8")
    print(f"{asunto}\nHTML en {ruta}")
    if args.enviar:
        config = leer_config()
        enviar(config, asunto, cuerpo, texto)
        print(f"Enviado a {', '.join(config.destinos)}")


if __name__ == "__main__":
    main()
