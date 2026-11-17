# Radar de Licitaciones

Revisa a diario los procesos publicados en Guatecompras y le entrega a ALBRAN Industries una lista corta de concursos relevantes, ordenados y con la razón de cada recomendación. Es el proyecto final del curso de Procesamiento de Lenguaje Natural (UFM).

En cuatro días de prueba de 2026, etiquetados a ciegas, el radar acierta **8 de cada 10** recomendaciones de la lista del día (precisión@10 de 0.80, también medida con las reglas exactas del panel). El filtro por categoría UNSPSC acierta 6.5 y el filtro por palabra clave 2.5. El detalle está en `notebooks/03_evaluacion.ipynb`.

## Cómo funciona

1. **Captura.** Descarga los archivos mensuales del OCDS de Guatecompras y los guarda en una base DuckDB (`datos/radar.duckdb`). Repetir la carga no duplica nada.
2. **Limpieza.** Arma el texto de cada proceso (título, ítems y características) y le quita los códigos internos, el HTML y los campos del formulario del oferente. De ahí salen dos representaciones, una lematizada con spaCy para TF-IDF y otra casi original para los embeddings.
3. **Dos señales.** Un clasificador TF-IDF con regresión logística responde si el concurso es para ALBRAN, y la similitud coseno con embeddings multilingües (`paraphrase-multilingual-MiniLM-L12-v2`) dice a qué proyecto del portafolio se parece.
4. **Ranking.** `score = 0.65 · P(relevante) + 0.35 · similitud máxima`, con α ajustado en validación. La lista del día solo incluye procesos con P ≥ 0.5 y como máximo cinco compras de computadoras. Las demás computadoras van en una sección aparte.
5. **Producto.** Un panel en Streamlit con la semana, el plazo para ofertar, la explicación de cada concurso y los botones «Me interesa» y «No aplica». Además, un resumen diario por correo.

El modelo se entrena una sola vez con las etiquetas humanas. La corrida diaria solo hace inferencia.

## Instalación

El proyecto usa el entorno compartido del repositorio, con Python 3.13.

```bash
# desde la raíz del repositorio
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` incluye el modelo de spaCy `es_core_news_sm`. El modelo de embeddings se descarga solo la primera vez que se usa, y son unos 470 MB.

Los comandos de abajo se corren desde `proyecto/`.

## Cómo correr cada etapa

| Etapa | Comando |
|---|---|
| Carga histórica | `../.venv/bin/python -m src.captura.actualizar --desde 2024-01` |
| Corrida del mes en curso | `../.venv/bin/python -m src.captura.actualizar` |
| Muestras para etiquetar | `../.venv/bin/python -m src.etiquetado.muestreo` |
| Ronda de aprendizaje activo | `../.venv/bin/python -m src.etiquetado.activo` |
| Entrenar y guardar el clasificador | `../.venv/bin/python -m src.modelo.clasificador` |
| Pooling de los días de prueba | `../.venv/bin/python -m src.etiquetado.pooling --reglas-panel` |
| Listas de la semana (simulación diaria) | `../.venv/bin/python -m src.modelo.ranking --hasta 2026-10-09` |
| Resumen por correo | `../.venv/bin/python -m src.producto.notificar [--dia AAAA-MM-DD] [--enviar]` |
| Exportar las etiquetas (o cargarlas con `--importar`) | `../.venv/bin/python -m src.etiquetado.exportar` |
| Panel y vista de etiquetado | `../.venv/bin/streamlit run src/producto/app.py` |

DuckDB no deja escribir si otro proceso tiene la base abierta. Antes de correr algo que escribe hay que cerrar el kernel de los notebooks. El panel abre conexiones cortas y no estorba.

### Correo

Copiar `.env.example` a `.env` y completarlo. Con Resend el servidor es `smtp.resend.com`, el puerto 465, el usuario `resend` y la clave es la API key. El remitente tiene que ser de un dominio verificado en Resend. Sin `--enviar`, `notificar` solo genera el HTML en `datos/correos/`, y no hace falta `.env`.

### En otra computadora

La base, el modelo entrenado y los datos crudos no están en git. Hay dos caminos.

- **Copiar las carpetas.** Pasar `proyecto/datos/` y `proyecto/modelos/` tal cual. Es lo más rápido, y conserva las listas del panel y las decisiones marcadas.
- **Reconstruir desde git.** Cargar el histórico con `src.captura.actualizar --desde 2024-01` (descarga unos 34 archivos mensuales), cargar las etiquetas con `src.etiquetado.exportar --importar`, entrenar con `src.modelo.clasificador` y generar las listas con `src.modelo.ranking`. Las decisiones marcadas en el panel no se recuperan por este camino.

## Estructura

```
proyecto/
├── src/
│   ├── captura/        # cliente del OCDS y corrida idempotente
│   ├── limpieza/       # texto de entrada y sus dos representaciones
│   ├── etiquetado/     # muestreo, aprendizaje activo, pooling y exportación
│   ├── modelo/         # clasificador, similitud y ranking
│   ├── producto/       # panel de Streamlit y resumen por correo
│   └── db.py           # esquema y conexión de DuckDB
├── notebooks/
│   ├── 01_exploracion.ipynb   # exploración y decisiones de limpieza
│   └── 03_evaluacion.ipynb    # cinco métodos, curvas PR, tope y errores
├── etiquetas/etiquetas.csv    # las etiquetas humanas (versionadas)
├── portafolio/proyectos_albran.csv
├── reporte/bitacora.md        # decisiones (D), hallazgos (H) y problemas (P)
├── tests/                     # pytest
└── img/                       # gráficas de los notebooks
```

No se versionan los datos crudos, la base DuckDB, los modelos entrenados ni `.env`.

## Calidad

```bash
../.venv/bin/python -m pytest -q -p no:anyio
../.venv/bin/ruff check src tests
```

## Limitaciones

- El clasificador se entrenó con 150 etiquetas, sin rondas de aprendizaje activo.
- La prueba son cuatro días y un solo anotador, así que la precisión@10 es preliminar.
- El OCDS completa la fecha de cierre días después de publicar. La lista diaria deja entrar los procesos sin esa fecha y el panel los marca como «sin fecha de cierre».
- No se procesan los PDF de las bases del concurso.

## Fuente de datos y licencia

Datos abiertos de Guatecompras en el estándar OCDS, publicados por el Ministerio de Finanzas Públicas de Guatemala en [ocds.guatecompras.gt](https://ocds.guatecompras.gt), con licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
