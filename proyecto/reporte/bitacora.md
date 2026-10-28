# Bitácora — Radar de Licitaciones

Registro de decisiones (D), hallazgos (H) y problemas (P) del proyecto final. Material de respaldo para la presentación y el reporte; el notebook solo resume.

## Hallazgos de la fuente (27/10/2026)

**H-01 · El archivo "JSON" del mes llega comprimido.** `GET /file/json/2026/9` responde `application/zip` (`2026-09_Guatecompras_json.zip`): 37 MB comprimido, 267 MB de JSON. Raíz `records[]`, cada record con `ocid`, `releases[]` y `compiledRelease`. Cierra el pendiente de compresión y tamaño de `docs/02`, sección 3.

**H-02 · `tender.description` y `tender.value` no vienen.** 0 % de presencia en los 35,401 procesos de septiembre de 2026. El texto disponible es `tender.title` más `items[].description` y los atributos de ítem «Características» y «Características Adicionales». Sin monto en el tender, el filtro de monto de `docs/04` no se puede aplicar con este campo.

**H-03 · `procurementMethod` es siempre `open`.** La modalidad real está en `procurementMethodDetails` (texto con el artículo de la LCE). El 92 % de septiembre es «Compra Directa con Oferta Electrónica (Art. 43 LCE Inciso b)».

**H-04 · Identificadores.** `ocid = ocds-xqjsxa-{NOG}` y `tender.id = GT-NOG-{NOG}`. Cada `ocid` aparece en un solo archivo mensual: la carga 2024-01 → 2026-04 no actualizó ningún registro entre meses.

**H-05 · Volumen mayor que el del deck.** Carga 2024-01 → 2026-10: 937,949 procesos (285,594 en 2024, 336,529 en 2025, 315,826 en 2026 hasta el 26/10). Promedio de 27,587 por mes y 1,258 por día hábil, no los ≈1,000 del deck (que salían de 1.77 M / 7 años / 250 días). Diciembre cae a la mitad (14 mil en 2024, 15 mil en 2025) por el cierre fiscal. Notebook 01, celda 3.2.

**H-06 · El caso de la tomografía es real y recurrente.** 2,271 procesos mencionan «software»; 394 (17.3 %) son estudios de imagen o equipo médico («se solicita que el servicio incluya software para visualización de imágenes…»). En la muestra de septiembre parecían mayoría porque el IGSS publica muchos en lote. Celda 3.10.

**H-07 · Ruido en los títulos.** 51.0 % de los títulos trae códigos internos (`CODIGO IGSS: 227281`, `PEDIDO No. 38811/2026`, `AMPARO No. …`, `P-11111`, `[DA-DAC-A18]`), 15.6 % trae HTML (`<br />`) y 48.2 % está todo en mayúsculas. Tras quitarlos, el bigrama más frecuente pasa de «codigo igss» (33.7 %) a «adquisicion de» (15.9 %). Celdas 3.6 y 3.7.

**H-08 · El volumen es compra médica.** 638 entidades; el IGSS publica el 64.2 % de los procesos y con el Ministerio de Salud llega al 79.3 %. El 91.1 % es compra directa, con 10.8 días de mediana para ofertar (confirma los «10 días» del deck). Celdas 3.3, 3.4 y 3.13.

**H-09 · Los dos filtros simples casi no coinciden.** Palabra clave (pre-filtro de `docs/04`): 49,984 procesos (5.3 %). Categoría UNSPSC 43 + 8111: 24,017 (2.6 %). Ambos a la vez: solo 8,695. El 64 % de la categoría TI (15,322) no tiene ninguna palabra clave: escáneres, computadoras, switches. Y 292 procesos mencionan software, no son médicos y no tienen categoría TI (segmento UNSPSC 80, servicios de gestión, entre los más comunes). Celdas 3.9–3.12.

**H-10 · Las características traen el formulario del oferente.** Los fragmentos más repetidos del campo «Características Adicionales» son requisitos de la oferta: «tiempo de entrega» (106,340 procesos), «iva» (67,010), «regimen tributario» (46,686), cuenta bancaria, NIT, teléfono. Son las «frases plantilla» que el deck esperaba encontrar en la descripción. Celda 3.8.

## Decisiones

**D-01 · Proyecto dentro del repo del curso, con el `.venv` raíz.** Por la entrega de avances del 27/10 se usa `proyecto/` con el entorno compartido (Python 3.13) en vez de `uv` con Python 3.12. Se agregaron `duckdb`, `pyarrow` y `spacy` + `es_core_news_sm`.

**D-02 · Esquema de `procesos` ajustado a los datos reales.** Sin `descripcion` ni `monto` (H-02); se agregan `caracteristicas`, `unspsc`, `n_items`, `entidad_id` y `fecha_publicacion`. Fechas convertidas a hora de Guatemala sin zona.

**D-03 · Texto de entrada.** Título + descripciones de ítems + características, sin repetir fragmentos que ya están en el título y cortado en 1,000 caracteres (hay procesos con cientos de ítems; MiniLM corta en 128 tokens).

**D-04 · Limpieza de ruido por expresiones regulares** para los códigos de H-07, HTML y separadores huérfanos. Se aplica a las dos representaciones.

**D-05 · `texto_tfidf` bajo demanda.** spaCy procesa ~460 documentos/s; lematizar el histórico completo tomaría cerca de una hora y no hace falta: el modelo se entrena con ~1,000 etiquetados y el ciclo diario procesa ~1,000 procesos.

**D-07 · Lista de campos de formulario derivada de los datos.** Un fragmento de características es campo de formulario si aparece en ≥ 1,000 procesos, tiene ≤ 8 palabras y no lleva «:» (los «clave: valor» como «grupo: fruta;» describen el producto y se conservan). Salen 134 campos, presentes en el 21.7 % de los procesos; se guardan en `src/limpieza/campos_formulario.txt` y `texto_entrada` los omite. Efecto en la muestra de 20 mil: de 43.5 a 39.7 palabras por documento antes de cualquier otra limpieza. Con todo el pipeline el vocabulario baja de 75,117 a 16,555 tipos (−78 %) y de 39.7 a 22.7 tokens por documento.

**D-06 · Minúsculas antes de spaCy.** Muchos títulos vienen en mayúsculas y el modelo pequeño lematiza peor en mayúsculas. Las tildes se quitan después de lematizar, no antes; la ñ se conserva.

## Problemas

**P-01 · Descarga colgada.** En la carga histórica la conexión de 2026-05 quedó esperando sin datos; con `timeout=600` por lectura podía tardar 10 min en reintentar. Se bajó a `(30, 120)` (conexión, lectura). El servidor respondía normal en paralelo (8 s).

**P-02 · Errores del lematizador pequeño.** `cafetera → cafetero`, `metformina → metformin`, `tableta → tableto`, `acoplar → acoplir`; con el texto en mayúsculas era peor (`metodología → metodologiar`). Limitación de `es_core_news_sm`; es un argumento más para la segunda señal con embeddings, que no lematiza.
