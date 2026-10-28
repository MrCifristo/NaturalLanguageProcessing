"""Convierte el record package de un mes en una fila por proceso."""

import json
import zipfile
from pathlib import Path

import pandas as pd


def _unicos(textos) -> list[str]:
    # Quita vacíos y repetidos conservando el orden
    return list(dict.fromkeys(t.strip() for t in textos if t and t.strip()))


def _fila(record: dict, mes_archivo: str) -> dict:
    cr = record["compiledRelease"]
    t = cr.get("tender") or {}
    items = t.get("items") or []
    periodo = t.get("tenderPeriod") or {}
    titulo = (t.get("title") or "").strip()

    atributos = [
        a.get("value")
        for it in items
        for a in it.get("attributes") or []
        if a.get("name") in ("Características", "Características Adicionales")
    ]
    return {
        "ocid": record["ocid"],
        "nog": (t.get("id") or "").removeprefix("GT-NOG-"),
        "titulo": titulo,
        # tender.description no viene en Guatecompras; el detalle está en los ítems
        "items_texto": " | ".join(t for t in _unicos(it.get("description") for it in items) if t != titulo),
        "caracteristicas": " | ".join(a for a in _unicos(atributos) if a != titulo),
        "n_items": len(items),
        "unspsc": ",".join(_unicos((it.get("classification") or {}).get("id") for it in items)),
        "entidad": (cr.get("buyer") or {}).get("name"),
        "entidad_id": (cr.get("buyer") or {}).get("id"),
        "modalidad": t.get("procurementMethodDetails"),
        "categoria": t.get("mainProcurementCategory"),
        "estado": t.get("status"),
        "fecha_publicacion": t.get("datePublished"),
        "fecha_inicio": periodo.get("startDate"),
        "fecha_cierre": periodo.get("endDate"),
        "fecha_release": cr.get("date"),
        "mes_archivo": mes_archivo,
    }


def leer_mes(ruta_zip: Path) -> pd.DataFrame:
    with zipfile.ZipFile(ruta_zip) as z:
        paquete = json.loads(z.read(z.infolist()[0]))
    anio, mes = ruta_zip.stem.split("-")
    mes_archivo = f"gc-{anio}-{int(mes)}"
    df = pd.DataFrame([_fila(r, mes_archivo) for r in paquete["records"]])

    # Fechas con zona -06:00; se guardan en hora de Guatemala sin zona
    for col in ("fecha_publicacion", "fecha_inicio", "fecha_cierre", "fecha_release"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_convert("America/Guatemala").dt.tz_localize(None)
    return df
