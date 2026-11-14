# Bitácora — Radar de Licitaciones

Registro de decisiones (D), hallazgos (H) y problemas (P) del proyecto final. Material de respaldo para la presentación y el reporte; el notebook solo resume.

## Hallazgos de la fuente (27/10/2026)

**H-01 · El archivo "JSON" del mes llega comprimido.** `GET /file/json/2026/9` responde `application/zip` (`2026-09_Guatecompras_json.zip`): 37 MB comprimido, 267 MB de JSON. Raíz `records[]`, cada record con `ocid`, `releases[]` y `compiledRelease`. Cierra el pendiente de compresión y tamaño del archivo.

**H-02 · `tender.description` y `tender.value` no vienen.** 0 % de presencia en los 35,401 procesos de septiembre de 2026. El texto disponible es `tender.title` más `items[].description` y los atributos de ítem «Características» y «Características Adicionales». Sin monto en el tender, el filtro por monto no se puede aplicar con este campo.

**H-03 · `procurementMethod` es siempre `open`.** La modalidad real está en `procurementMethodDetails` (texto con el artículo de la LCE). El 92 % de septiembre es «Compra Directa con Oferta Electrónica (Art. 43 LCE Inciso b)».

**H-04 · Identificadores.** `ocid = ocds-xqjsxa-{NOG}` y `tender.id = GT-NOG-{NOG}`. Cada `ocid` aparece en un solo archivo mensual: la carga 2024-01 → 2026-04 no actualizó ningún registro entre meses.

**H-05 · Volumen mayor que el del deck.** Carga 2024-01 → 2026-10: 937,949 procesos (285,594 en 2024, 336,529 en 2025, 315,826 en 2026 hasta el 26/10). Promedio de 27,587 por mes y 1,258 por día hábil, no los ≈1,000 del deck (que salían de 1.77 M / 7 años / 250 días). Diciembre cae a la mitad (14 mil en 2024, 15 mil en 2025) por el cierre fiscal. Notebook 01, celda 3.2.

**H-06 · El caso de la tomografía es real y recurrente.** 2,271 procesos mencionan «software»; 394 (17.3 %) son estudios de imagen o equipo médico («se solicita que el servicio incluya software para visualización de imágenes…»). En la muestra de septiembre parecían mayoría porque el IGSS publica muchos en lote. Celda 3.10.

**H-07 · Ruido en los títulos.** 51.0 % de los títulos trae códigos internos (`CODIGO IGSS: 227281`, `PEDIDO No. 38811/2026`, `AMPARO No. …`, `P-11111`, `[DA-DAC-A18]`), 15.6 % trae HTML (`<br />`) y 48.2 % está todo en mayúsculas. Tras quitarlos, el bigrama más frecuente pasa de «codigo igss» (33.7 %) a «adquisicion de» (15.9 %). Celdas 3.6 y 3.7.

**H-08 · El volumen es compra médica.** 638 entidades; el IGSS publica el 64.2 % de los procesos y con el Ministerio de Salud llega al 79.3 %. El 91.1 % es compra directa, con 10.8 días de mediana para ofertar (confirma los «10 días» del deck). Celdas 3.3, 3.4 y 3.13.

**H-09 · Los dos filtros simples casi no coinciden.** Palabra clave (pre-filtro original): 49,984 procesos (5.3 %). Categoría UNSPSC 43 + 8111: 24,017 (2.6 %). Ambos a la vez: solo 8,695. El 64 % de la categoría TI (15,322) no tiene ninguna palabra clave: escáneres, computadoras, switches. Y 292 procesos mencionan software, no son médicos y no tienen categoría TI (segmento UNSPSC 80, servicios de gestión, entre los más comunes). Celdas 3.9–3.12.

**H-10 · Las características traen el formulario del oferente.** Los fragmentos más repetidos del campo «Características Adicionales» son requisitos de la oferta: «tiempo de entrega» (106,340 procesos), «iva» (67,010), «regimen tributario» (46,686), cuenta bancaria, NIT, teléfono. Son las «frases plantilla» que el deck esperaba encontrar en la descripción. Celda 3.8.

**H-11 · Palabras clave del pre-filtro con mucho ruido (13/11/2026).** De 2,367 títulos con «desarroll» (enero 2024 – junio 2025), al menos 1,014 (43 %) son nombres de instituciones: Ministerio de Desarrollo Social, Consejos de Desarrollo, Fondo de Desarrollo Indígena. «Sistema» trae unos 350 mantenimientos de sistemas que no son de TI (frenos, eléctrico, hidráulico, contra incendio), y «aplicación» trae «aplicación de pintura» y «de impermeabilizante». «Tecnologías de la información» suele ser la unidad que pide la compra («solicitado por la Dirección de TI») y no el objeto. Eso no es puro ruido, porque esas compras sí tienden a ser de TI, aunque también aparecen cosas como el aire acondicionado de la Dirección de TI. El pre-filtro de `muestreo.py` usa frases («desarrollo de software», «sistema informático») en lugar de palabras sueltas.

**H-12 · El criterio de relevancia sube la tasa de positivos (13/11/2026).** Al incluir hardware, la categoría UNSPSC 43 sola ya es el 2.0 % del periodo de entrenamiento (8,886 de 453,594), el doble del ~1 % que suponía el deck. Por eso la línea base por categoría será fuerte. El aporte del modelo está en lo relevante sin categoría TI (software fuera de 43, cableado y CCTV con categorías de construcción o seguridad) y en descartar el equipo médico con software (H-06).

## Decisiones

**D-01 · Proyecto dentro del repo del curso, con el `.venv` raíz.** Por la entrega de avances del 27/10 se usa `proyecto/` con el entorno compartido (Python 3.13) en vez de `uv` con Python 3.12. Se agregaron `duckdb`, `pyarrow` y `spacy` + `es_core_news_sm`.

**D-02 · Esquema de `procesos` ajustado a los datos reales.** Sin `descripcion` ni `monto` (H-02); se agregan `caracteristicas`, `unspsc`, `n_items`, `entidad_id` y `fecha_publicacion`. Fechas convertidas a hora de Guatemala sin zona.

**D-03 · Texto de entrada.** Título + descripciones de ítems + características, sin repetir fragmentos que ya están en el título y cortado en 1,000 caracteres (hay procesos con cientos de ítems; MiniLM corta en 128 tokens).

**D-04 · Limpieza de ruido por expresiones regulares** para los códigos de H-07, HTML y separadores huérfanos. Se aplica a las dos representaciones.

**D-05 · `texto_tfidf` bajo demanda.** spaCy procesa ~460 documentos/s; lematizar el histórico completo tomaría cerca de una hora y no hace falta: el modelo se entrena con ~1,000 etiquetados y el ciclo diario procesa ~1,000 procesos.

**D-07 · Lista de campos de formulario derivada de los datos.** Un fragmento de características es campo de formulario si aparece en ≥ 1,000 procesos, tiene ≤ 8 palabras y no lleva «:» (los «clave: valor» como «grupo: fruta;» describen el producto y se conservan). Salen 134 campos, presentes en el 21.7 % de los procesos; se guardan en `src/limpieza/campos_formulario.txt` y `texto_entrada` los omite. Efecto en la muestra de 20 mil: de 43.5 a 39.7 palabras por documento antes de cualquier otra limpieza. Con todo el pipeline el vocabulario baja de 75,117 a 16,555 tipos (−78 %) y de 39.7 a 22.7 tokens por documento.

**D-06 · Minúsculas antes de spaCy.** Muchos títulos vienen en mayúsculas y el modelo pequeño lematiza peor en mayúsculas. Las tildes se quitan después de lematizar, no antes; la ñ se conserva.

**D-08 · Sin LLM, etiquetado por aprendizaje activo (13/11/2026).** El plan usaba Claude Haiku para pre-etiquetar y como línea base zero-shot. Se descarta por dos razones. La primera es que el proyecto debe sostenerse con técnicas del curso. La segunda es el anclaje: quien etiqueta tiende a aceptar la propuesta que ve, y el LLM terminaría evaluado contra etiquetas que él mismo sugirió. En su lugar, el clasificador TF-IDF + RL propone por rondas. La semilla se etiqueta a ciegas, cada ronda se reentrena y el lote siguiente prioriza los procesos con P cercana a 0.5, más una parte aleatoria. El mismo anclaje afectaría a nuestro modelo, así que validación, prueba y pooling se etiquetan sin propuesta. En entrenamiento la meta es de 400–600 etiquetas, con parada cuando el AUC-PR de validación deja de mejorar. A eso se suman las de validación (150–200) y las del pooling (~20–35 por día de prueba), así que el total sigue cerca de 1,000. El ahorro frente al plan original, que eran ≈1,000 más el pooling, es menor de lo que parece. La comparación final queda en cinco métodos. Cambia el esquema de `etiquetas`: salen `etiqueta_llm` y `justificacion_llm` y entran `conjunto`, `ronda`, `p_propuesta` y `terminos_propuesta`.

**D-09 · Resumen diario y alertas por correo, no por Slack (13/11/2026).** El alcance original ya aceptaba Slack o correo. El correo le llega a cualquier persona de ALBRAN sin instalar nada, y las alertas de la Fase 7 usan el mismo canal. Se envía por SMTP con `smtplib` (de la biblioteca estándar) y una contraseña de aplicación de Gmail en `.env`.

**D-10 · Embeddings bajo demanda (14/11/2026).** Igual que `texto_tfidf` (D-05), `texto_embedding` no se calcula para los ~940 mil procesos del histórico. Solo hacen falta para los procesos etiquetados, los días de prueba, el portafolio y la corrida diaria, unos pocos miles en total. Calcular todo el histórico en CPU tomaría horas que el plan del fin de semana no tiene.

**D-11 · Muestreo de la semilla y la validación (14/11/2026).** 150 procesos por conjunto, 70 % del pre-filtro y 30 % aleatorio del resto del periodo. El pre-filtro une 33 frases (H-11), buscadas sin tildes en título e ítems, con las categorías UNSPSC 43 y 8111. En entrenamiento deja pasar 15,851 de 453,594 procesos (3.5 %) y en validación 8,218 de 168,529 (4.9 %). Por eso, en la semilla, cada proceso del pre-filtro representa a unos 150 y cada aleatorio a unos 9,700 (78 y 3,560 en validación). Las métricas de validación hay que reponderarlas por estrato; si no, el AUC-PR sale inflado. El orden de la muestra sale de un hash del `ocid` con la semilla y no de `random()`, porque así no depende de cómo DuckDB reparte el escaneo entre hilos. Los procesos muestreados entran a `etiquetas` con `etiqueta_humana` en NULL, y `ocid` es la llave, así que repetir el muestreo no duplica nada.

## Problemas

**P-01 · Descarga colgada.** En la carga histórica la conexión de 2026-05 quedó esperando sin datos; con `timeout=600` por lectura podía tardar 10 min en reintentar. Se bajó a `(30, 120)` (conexión, lectura). El servidor respondía normal en paralelo (8 s).

**P-02 · Errores del lematizador pequeño.** `cafetera → cafetero`, `metformina → metformin`, `tableta → tableto`, `acoplar → acoplir`; con el texto en mayúsculas era peor (`metodología → metodologiar`). Limitación de `es_core_news_sm`; es un argumento más para la segunda señal con embeddings, que no lematiza. Al pasar el texto a minúsculas (D-06) `cafetera`, `tableta` y `metformina` ya salen bien; queda por ejemplo `jeringa → jeringuir`.

**P-03 · La corrida diaria se caía cuando el SHA cambiaba (13/11/2026).** La consulta del SHA previo usaba `fetchone()`, y en DuckDB 2.0 eso deja el resultado abierto y el `register` siguiente falla con `ResourceInUseException`. En la carga histórica no se vio porque cada mes era nuevo y no había SHA previo; en el ciclo diario pasaría casi todos los días, porque el OCDS regenera los archivos cada madrugada. Lo atrapó `test_sha_nuevo_actualiza_sin_duplicar`; se cambió a `fetchall()`.

**P-04 · Llave repetida en `corridas` (13/11/2026).** El `id` llevaba la hora al segundo y dos corridas del mismo mes en el mismo segundo chocaban (pasa con una corrida `sin_cambios`, que tarda menos de un segundo). Ahora incluye microsegundos.

**P-05 · «CÓDIGO IGSS. 5882.» dejaba el número (13/11/2026).** El patrón aceptaba «:» después de IGSS pero no «.», así que en `texto_embedding` quedaba «5882.» suelto. En TF-IDF no se notaba porque ahí se quitan los números.

**P-06 · API del OCDS en mantenimiento (13/11/2026, 21:30).** `GET /files` respondió 503 con una página HTML «Sitio en Mantenimiento». `_get` lo trata como error y reintenta 4 veces; si persiste, la corrida queda como `error` en `corridas`. Es un caso para la alerta de la Fase 7.

**P-07 · Códigos que la limpieza todavía no quita (13/11/2026).** Al escribir las pruebas de `normalizar.py` aparecieron formatos que `PATRONES_RUIDO` no cubre: «MSPAS 2024 GUA-CD-167», «EPQ-GSI-135-2024 SC.20243105» y «FYB-M 2830-2026». En TF-IDF pesan poco porque los números se quitan, pero las siglas quedan como tokens. Queda anotado y no se corrige hasta medir cuántos procesos los traen.
