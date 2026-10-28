# Radar de Licitaciones — Fuente de datos (OCDS Guatecompras)
> Documento técnico · Proyecto final PLN, UFM
> Estado: endpoint de índice verificado; estructura interna por verificar · Actualizado: octubre 2026

## 1. Resumen

Guatecompras publica todos sus procesos como datos abiertos bajo el estándar OCDS (Open Contracting Data Standard). La captura del proyecto usa exclusivamente esa publicación. No se hace scraping del portal para datos que ya están en el OCDS.

Cada etiqueta de esta tabla indica qué tan confirmado está cada dato:

- **Verificado:** comprobado con una petición real o en el código de una herramienta que lo consume.
- **Reportado:** declarado por una fuente secundaria (OCP Data Registry, proyectos de terceros), no comprobado aquí.
- **Por verificar:** suposición razonable que hay que confirmar con el primer archivo descargado.

## 2. Endpoints

### Índice de archivos (verificado)

```http
GET https://ocds.guatecompras.gt/files
```

Respuesta JSON (forma observada el 29 de septiembre de 2026):

```json
{
  "result": [
    {
      "id": "gc-2026-9",
      "files": {
        "csv":  "https://ocds.guatecompras.gt/file/csv/2026/9",
        "xlsx": "https://ocds.guatecompras.gt/file/xlsx/2026/9",
        "json": "https://ocds.guatecompras.gt/file/json/2026/9",
        "sha":  "https://ocds.guatecompras.gt/file/sha/2026/9"
      },
      "year": 2026,
      "month": 9,
      "monthName": "Septiembre",
      "source": "Guatecompras",
      "timestamp": "2026-09-29T03:39:25.241206-06:00"
    }
  ]
}
```

Observaciones verificadas:

- Un registro por mes desde enero 2020 hasta el mes en curso.
- Patrón de URL: `/file/{json|csv|xlsx|sha}/{año}/{mes}`, con el mes sin cero a la izquierda.
- Los meses del año en curso y de años anteriores tenían timestamp del mismo día entre 03:00 y 04:30 (hora de Guatemala). Los archivos se regeneran de madrugada, incluidos meses pasados, porque los procesos cambian de estado.
- Kingfisher Collect (herramienta oficial de Open Contracting Partnership) usa este mismo endpoint en su spider `guatemala_bulk` y trata cada archivo JSON como `record_package`.

### Documentación oficial

| Recurso | URL | Nota |
|---|---|---|
| Documentación de la API | `https://ocds.guatecompras.gt/api-ocds` | Página renderizada con JavaScript; endpoints adicionales no verificados |
| Descarga masiva | `https://ocds.guatecompras.gt/descarga-datos` | Interfaz del mismo índice |
| Spider de referencia | `kingfisher_scrapy/spiders/guatemala_bulk.py` en `github.com/open-contracting/kingfisher-collect` | Implementación de referencia del consumo del índice |

## 3. Formato de los archivos

| Aspecto | Valor | Estado |
|---|---|---|
| Tipo de paquete | Record package | Verificado (Kingfisher) |
| Raíz | `records[]` | Reportado (proyecto Transparencia Ciudadana) |
| Contenido por record | `ocid` + `compiledRelease` con `tender`, `buyer`, `awards`, etc. | Reportado |
| Compresión del JSON | [PENDIENTE: verificar si viene comprimido] | Por verificar |
| Tamaño de un mes | [PENDIENTE: medir] | Por verificar |
| Criterio de asignación al mes | [PENDIENTE: fecha de publicación o de creación del proceso] | Por verificar |

## 4. Campos a extraer

Rutas del estándar OCDS 1.1. Todas son **por verificar** contra el primer archivo descargado; Guatecompras puede usar extensiones.

| Ruta (dentro de `compiledRelease`) | Uso en el proyecto |
|---|---|
| `ocid` | Clave primaria; deduplicación |
| `tender.id` | Probable NOG [PENDIENTE: confirmar] |
| `tender.title` | Texto principal |
| `tender.description` | Texto complementario |
| `tender.items[].description` | Texto complementario |
| `tender.items[].classification` | Categoría asignada por la entidad; solo para la línea base de filtro |
| `tender.mainProcurementCategory` | Categoría general (bienes, servicios, obras) |
| `tender.status` | Filtro de vigencia |
| `tender.tenderPeriod.startDate` / `endDate` | Vigencia y días al cierre |
| `tender.procurementMethod` / `procurementMethodDetails` | Modalidad (compra directa, licitación, etc.) |
| `tender.value.amount` / `currency` | Filtro de monto |
| `buyer.name` / `buyer.id` | Entidad compradora |
| `date` | Fecha del release compilado |
| `tender.documents[]` | Enlaces a bases; no se procesan en el MVP |

## 5. Datos de referencia (OCP Data Registry)

Publicación de Guatemala en `data.open-contracting.org/es/publication/142`. Útil para conteos y para descargas históricas por año, pero **no sirve para el ciclo diario**: el registro obtiene los datos una vez al mes.

| Dato | Valor | Estado |
|---|---|---|
| Cobertura | Enero 2020 – septiembre 2026 | Reportado |
| Licitaciones | 1,768,963 | Reportado |
| Artículos | 3,141,416 | Reportado |
| Documentos | 6,437,472 | Reportado |
| Archivos por año | `.jsonl.gz`, un proceso por línea (2024 ≈ 264 MB, 2025 ≈ 324 MB, 2026 ≈ 309 MB) | Reportado |
| CSV | `main.csv` con tablas relacionadas por `main_ocid` | Reportado |
| Licencia | CC BY 4.0 | Reportado |

Exclusiones declaradas por el registro:

- Concursos con NOG anteriores a 2020.
- Publicaciones del módulo NPG de Guatecompras.

Problemas de calidad declarados:

- Entradas duplicadas en `tender/tenderers` y `parties/additionalContactPoint`.
- Fechas mal escritas en `contracts/dateSigned`.

Ninguno afecta los campos de texto que usa el proyecto, pero los oferentes no deben usarse como variable sin deduplicar.

## 6. Captura diaria

Flujo (detalle de módulos en `docs/03`):

1. `GET /files` y seleccionar el registro del mes en curso (máximo por `year`, `month`).
2. Descargar `files.sha` y compararlo con el último SHA guardado. Si es igual, terminar.
3. Descargar `files.json`.
4. Recorrer `records[].compiledRelease`, insertar solo `ocid` no vistos y actualizar el estado de los ya vistos.
5. El día 1 de cada mes, repetir el proceso también con el mes anterior.

Esqueleto de referencia:

```python
import requests

BASE = "https://ocds.guatecompras.gt"

indice = requests.get(f"{BASE}/files", timeout=60).json()["result"]
mes = max(indice, key=lambda r: (r["year"], r["month"]))

sha = requests.get(mes["files"]["sha"], timeout=60).text.strip()
if sha == cargar_sha_anterior(mes["id"]):
    raise SystemExit("Sin cambios")

paquete = requests.get(mes["files"]["json"], timeout=600).json()
for record in paquete["records"]:
    release = record["compiledRelease"]
    ...
```

Requisitos de robustez:

- Reintentos con backoff en las tres peticiones.
- Guardar el archivo descargado en `datos/crudos/{año}-{mes}.json` antes de procesarlo.
- Si la descarga falla, conservar el último archivo válido y registrar el error.
- Alerta si el SHA del mes en curso no cambia en 48 horas.

## 7. Verificaciones pendientes

Hacer en la Fase 1 antes de escribir el parser definitivo:

1. Descargar el índice y un mes completo; registrar tamaño y si viene comprimido.

   ```bash
   curl -s https://ocds.guatecompras.gt/files | python -m json.tool | head -40
   curl -sI https://ocds.guatecompras.gt/file/json/2026/9
   curl -s -o muestra.json https://ocds.guatecompras.gt/file/json/2026/9
   ```

2. Inspeccionar la estructura real de un record y confirmar las rutas de la sección 4.
3. Confirmar el formato de `ocid` y la relación entre `tender.id` y el NOG.
4. **Rezago de publicación:** anotar 5–10 NOG publicados hoy en guatecompras.gt y comprobar en qué corrida aparecen en el archivo del mes. Si el rezago es mayor a un día, documentarlo como limitación.
5. **Cobertura NPG:** identificar qué modalidades se publican por NPG y quedan fuera del OCDS.
6. Confirmar con qué fecha se asigna un proceso a un mes.
