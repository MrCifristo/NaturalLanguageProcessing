# Bitácora de trabajo — Laboratorio #2

Registro de hallazgos, decisiones y problemas encontrados durante el desarrollo.
Sirve como material de respaldo para el reporte escrito: cada afirmación del reporte
debería poder rastrearse hasta una entrada de aquí.

**Corpus:** Spanish News Classification (`df_total.csv`) · **Notebook:** `lab2.ipynb`

---

## Entorno

**D-01. El entorno y las dependencias se movieron a la raíz del repositorio.**
`.venv/` y `requirements.txt` vivían dentro de `lab1-nlp/`. Como ambos laboratorios usan
exactamente las mismas librerías, duplicarlos por carpeta no aporta nada. Cada `labN-nlp/`
conserva solo lo específico de su entrega (notebook, corpus, reporte, enunciado).

**P-01. El kernel de Jupyter del Lab #1 estaba roto.**
`Python (lab1-nlp)` apuntaba a `/Users/milton/Github/lab1-nlp/.venv`, ruta que dejó de existir
al mover el proyecto a `NaturalLanguageProcessing/`. Se registró un kernel nuevo,
**`Python (NLP)`**, apuntando al `.venv` de la raíz, y se repuntó el viejo para que `lab1.ipynb`
siga ejecutándose sin modificarlo.

**P-02. NLTK 3.10 bloquea el import de `regex` cuando el `.venv` está dentro del directorio de
trabajo.** El error es `ImportError: Blocked import of regex from current working directory for
security reasons`. Es un falso positivo de su hook de seguridad. Se resuelve con la variable de
entorno `NLTK_DISABLE_IMPORT_SECURITY=1`, que quedó incluida en la definición del kernel.

**D-02. Dependencias nuevas respecto al Lab #1:** `scikit-learn` 1.7.2 (arrastra `scipy` 1.18).
Se usa **únicamente para vectorizar**; la tokenización y normalización siguen siendo de NLTK,
igual que en el Lab #1.

---

## Sección 0 — Preparación del corpus

**D-03. El notebook del Lab #2 es autocontenido.**
El enunciado dice "parta del mismo notebook", pero se optó por un `lab2.ipynb` independiente que
reproduce la carga y el pipeline del Lab #1 en su Sección 0. Razones: `lab1.ipynb` ya está
entregado y no conviene modificarlo, y un notebook por entrega es más limpio que uno solo con
ambos laboratorios encadenados. El costo es duplicar `df_total.csv` en las dos carpetas.

**D-04. Las representaciones se construyen sobre `tokens_stem`, no sobre `tokens_norm`.**
El enunciado define el corpus normalizado como la salida del pipeline completo, que incluye el
paso de "lematización". Por fidelidad a esa definición se usa la versión con *stemming*.
*Costo asumido:* los resultados legibles (top-10 de TF-IDF, bigramas) aparecen como raíces
cortadas — `empres`, `sostenibil`, `inversion` — lo que hace la lectura menos natural.
`tokens_norm` se conserva en el DataFrame por si conviene mostrar alguna tabla en versión
legible.

**D-05. Se re-unen los tokens en una columna de texto (`texto_norm`).**
`CountVectorizer` y `TfidfVectorizer` no aceptan listas de tokens: esperan un iterable de
*strings*. Se une con `" ".join(...)`. Puede parecer contradictorio volver a pegar lo que se
acababa de separar, pero es lo que mantiene el pipeline del Lab #1 como única fuente de verdad:
al unir con un espacio simple, el re-tokenizado interno de sklearn es trivial y no puede
deshacer ninguna decisión ya tomada. La alternativa —pasarle un `analyzer` propio— duplicaría
la lógica de normalización en dos lugares distintos.

**D-06. Se eliminan las 75 filas exactamente duplicadas, igual que en el Lab #1.**
En el Lab #1 la razón era no inflar los conteos. Aquí hay una más fuerte: dos copias del mismo
texto tienen similitud coseno exactamente 1.0 y contaminarían el ranking de documentos más
similares de la Sección 5.

**D-08. Se eliminan además los dos documentos con texto vacío (ver H-05).**
Corpus de trabajo: **1,140 documentos**, dos menos que los 1,142 del Lab #1. La diferencia debe
explicarse en el reporte: no es una discrepancia sino una limpieza adicional, motivada por un
defecto de calidad de datos que el Lab #1 no había detectado porque su chequeo buscaba nulos y
estas filas contienen un espacio, no un nulo.

**V-01. El pipeline se reprodujo sin desviaciones.**
El corpus normalizado da **299,799 tokens y 13,103 tipos**, cifras idénticas a las reportadas en
el Lab #1. Confirma que la Sección 0 reconstruye el mismo estado y no una versión aproximada.

---

## Sección 1 — Bolsa de palabras

**R-01. Forma de la matriz BoW: `(1140, 13080)`** — 1,140 documentos × 13,080 términos.
El vocabulario no cambió al eliminar los dos documentos vacíos, porque no aportaban ningún
token.

**H-01. El vocabulario de la matriz (13,080) no coincide con los tipos del corpus (13,103).**
La diferencia son 23 términos y **no** se debe a la normalización, sino al parámetro
`token_pattern` de `CountVectorizer`, cuyo valor por defecto `(?u)\b\w\w+\b` exige tokens de al
menos dos caracteres. Los descartados son:
`b c d f g h i k l m n o p q r s t u v w x z π`

**D-07. Se mantiene el `token_pattern` por defecto.**
Los 23 tokens descartados suman **218 ocurrencias de 299,799**, un **0.07%** del corpus, y todos
son letras sueltas sin valor semántico (residuos de siglas y enumeraciones partidas por el
tokenizador). Cambiar el patrón a `(?u)\b\w+\b` los recuperaría y daría un vocabulario de
exactamente 13,103, pero solo añadiría ruido. La decisión queda cuantificada, no por omisión.

**V-02. La matriz se validó contra un conteo independiente hecho con `Counter` de NLTK.**
Cinco pruebas, todas superadas:

| Verificación | Resultado |
|---|---|
| Suma total de la matriz = tokens − descartados | 299,581 = 299,799 − 218 |
| Cada columna vs. `Counter`, término por término | 0 discrepancias en 13,080 columnas |
| Cada fila vs. largo real del documento | 0 discrepancias en todas las filas |
| Vocabulario == tipos de ≥2 caracteres | True |
| Top-8 por frecuencia, matriz vs. `Counter` | Idénticos |

No existe un "resultado esperado" externo contra el cual comparar un BoW; lo verificable es la
**consistencia**, es decir, que la matriz sea una traducción fiel del corpus que la generó.

**R-02. Términos más frecuentes del corpus:**
`bbva`, `econom`, `banc`, `pais`, `inflacion`, `empres`, `nuev`, `año`.
Que `bbva` encabece la lista sugiere que buena parte del corpus proviene de notas asociadas a esa
entidad; es un sesgo de la fuente que conviene tener presente al interpretar los resultados.

**V-03. La pérdida de orden se demostró empíricamente, no solo se afirmó.**
`"banco demanda empresa"` y `"empresa demanda banco"` producen vectores **idénticos**. El ejemplo
importa en este corpus concreto porque las relaciones económicas son direccionales: quién
adquiere a quién, quién regula a quién, quién sanciona a quién.

---

## Sección 2 — Dispersión

**R-03. La matriz BoW completa es 98.71% ceros.**
14,911,200 celdas, de las cuales solo 192,895 son distintas de cero.

**R-04. La dispersión crece con el tamaño del vocabulario:**

| `max_features` | Vocabulario | No-ceros | Dispersión | Cobertura de tokens |
|---|---|---|---|---|
| 100 | 100 | 35,825 | 68.57% | 25.2% |
| 500 | 500 | 99,805 | 82.49% | 58.3% |
| 1,000 | 1,000 | 133,312 | 88.31% | 74.0% |
| 5,000 | 5,000 | 181,752 | 96.81% | 95.9% |
| completo | 13,080 | 192,895 | 98.71% | 100.0% |

**H-06. La cobertura es el contrapeso de la dispersión.**
Con 1,000 términos —el 7.6% del vocabulario— se retiene el 74% de las ocurrencias del corpus, y
con 5,000 el 95.9%. La cola de términos raros aporta muchísima dispersión y muy poca masa de
texto, que es lo que hace defendible recortar el vocabulario cuando el costo importa. Este dato
no lo pedía el enunciado; se agregó porque sin él la tabla sugiere que recortar solo tiene
ventajas.

**H-02. La causa de que la dispersión crezca es asimétrica.**
Al ampliar el vocabulario, el denominador (número de celdas) crece **proporcionalmente** al
vocabulario, mientras que el numerador (celdas ocupadas) crece cada vez más despacio: las
palabras que se van agregando son cada vez más raras y aparecen en cada vez menos documentos.
Pasar de 5,000 a 13,080 términos multiplica las columnas por 2.6 pero solo añade un 6% de
celdas ocupadas.

**H-03. El 45.9% del vocabulario (6,000 términos) aparece en un solo documento.**
Es la explicación estructural de la dispersión: casi la mitad de las columnas de la matriz tienen
un único valor distinto de cero y 1,139 ceros. Conecta directamente con la cola de
*hapax legomena* observada en la Ley de Zipf del Lab #1. Ampliando el criterio, el **65.3%** del
vocabulario aparece en 3 documentos o menos, y solo el **3.6%** (476 términos) supera los 100
documentos.

**H-04. Un documento usa en promedio 169 palabras distintas, el 1.29% del vocabulario.**
Mediana 154, mínimo 25, máximo 666. Ningún documento se acerca a ocupar una fracción apreciable
de las 13,080 columnas disponibles: incluso el más largo llega apenas al 5%.

**R-05. Costo de almacenamiento: 119.3 MB en denso contra 2.32 MB en CSR, un factor de 51×.**
Desglose del CSR: `data` 192,895 valores (1.54 MB), `indices` 192,895 valores (0.77 MB),
`indptr` 1,141 valores (0.005 MB).

**H-05. Dos documentos del corpus estaban vacíos. RESUELTO.**
Los índices **143** (Sostenibilidad) y **237** (Macroeconomia) tenían `news` igual a `' '`, un
único espacio. No son nulos, así que pasaron el chequeo de valores nulos del Lab #1, pero al
normalizar quedaban como filas completamente en cero. En BoW eran inofensivos; en la Sección 5 sí
importaban, porque un vector de ceros da similitud coseno 0 contra todo y arrastra hacia abajo el
promedio intra-categoría de esas dos categorías.
*Decisión:* eliminarlos en la Sección 0 (ver D-08). *Verificación:* tras el cambio, el mínimo de
palabras distintas por documento pasó de **0 a 25**, confirmando que ya no quedan filas nulas.

**D-09. La gráfica de dispersión usa eje x logarítmico.**
El vocabulario va de 100 a 13,080, así que en escala lineal los cuatro primeros puntos quedarían
amontonados contra el eje. Se conserva el estilo de las figuras del Lab #1 (mismo azul `#16679a`,
mismo `figsize`, salida en `reporte/img/`) para que el reporte se vea consistente. Archivo:
`reporte/img/dispersion.png`.

---

## Sección 3 — N-gramas

**R-06. Tamaño del vocabulario según `ngram_range`:**

| Configuración | `ngram_range` | Vocabulario | vs. unigramas | No-ceros | Dispersión |
|---|---|---|---|---|---|
| Unigramas | `(1, 1)` | 13,080 | 1.0× | 192,895 | 98.706% |
| Bigramas | `(2, 2)` | 198,880 | 15.2× | 281,807 | 99.876% |
| Trigramas | `(3, 3)` | 271,910 | 20.8× | 292,948 | 99.905% |
| Uni + bi | `(1, 2)` | 211,960 | 16.2× | 474,702 | 99.804% |
| Uni + bi + tri | `(1, 3)` | 483,870 | 37.0× | 767,650 | 99.861% |

**R-07. Top 15 bigramas** (corpus con *stemming*): `banc central` (212), `bbva research` (186),
`tas interes` (183), `cad vez` (183), `cambi climat` (150), `amer latin` (146), `millon eur` (140),
`punt basic` (139), `bbva mexic` (132), `polit monetari` (129), `pued ser` (129),
`preci consumidor` (128), `año pas` (125), `indic preci` (116), `ultim años` (106).

**H-07. Once de los quince bigramas aportan información real; cuatro son ruido.**
Aportan los conceptos técnicos (`banc central`, `polit monetari`, `tas interes`, `punt basic`,
`preci consumidor`, `indic preci`), las entidades compuestas (`bbva research`, `bbva mexic`,
`amer latin`, `millon eur`) y sobre todo `cambi climat`. Este último es el mejor argumento del
apartado: `cambi` es de las palabras más ambiguas del corpus —aparece en cambio de divisas,
cambio de directiva y cambio climático— y solo el bigrama separa el tema de sostenibilidad de los
otros dos. Son ruido `cad vez`, `pued ser`, `año pas` y `ultim años`: muletillas periodísticas
que aparecen en todas las categorías por igual. Sobreviven porque el filtro de stopwords del
Lab #1 opera palabra por palabra, no sobre pares.

**H-08. Los n-gramas saltan sobre las palabras que el pipeline eliminó.**
Se calculan sobre `texto_norm`, que ya no tiene stopwords, así que un bigrama puede unir dos
palabras que en el texto original **no eran contiguas**: `tas interes` viene de "tasa **de**
interés" y `preci consumidor` de "precios **al** consumidor". En este corpus el efecto juega a
favor, porque compacta conceptos que en español van unidos por preposiciones y permite que una
idea de tres palabras quepa en un bigrama. Pero deja de ser cierto que los bigramas representen
adyacencia en el texto real. **Hay que declararlo en el reporte**: es una consecuencia del orden
del pipeline, no una propiedad de los n-gramas.

**H-09. El *stemming* consolida los conteos de los bigramas.**
`banc central` suma 212 ocurrencias frente a las 156 de su versión legible `banco central`,
porque la raíz fusiona *banco/bancos* y *central/centrales*. Igual con `tas interes` (183) contra
`tasas interés` (121). Las variantes de número y género se consolidan, así que el conteo del
bigrama con stemming representa mejor el concepto subyacente.

**H-10. El vocabulario de n-gramas crece con material inservible.**
Es el hallazgo central de la sección y el argumento fuerte contra usar n-gramas crudos:

| | Vocabulario | Aparece 1 sola vez |
|---|---|---|
| Unigramas | 13,080 | 5,189 (**39.7%**) |
| Bigramas | 198,880 | 159,228 (**80.1%**) |
| Trigramas | 271,910 | 255,114 (**93.8%**) |

Una columna que se activa en un único documento no permite generalizar, solo memorizar: es
material directo de sobreajuste. Esto explica por qué la dispersión sube tanto — el vocabulario se
multiplica por 15 o por 20 mientras las celdas ocupadas crecen apenas 46% y 52%.

**H-11. `min_df=2` recorta el 82% del vocabulario de bigramas.**
De 198,880 a 35,803 términos, eliminando exactamente los pares que ocurrían en un único documento.
La configuración `(1,2)` con `min_df=2` deja 42,883 columnas y baja la dispersión de 99.804% a
99.375%. Se midió para poder afirmar en el reporte que el problema tiene solución práctica, en vez
de solo constatar la explosión del vocabulario.

**D-10. Los 15 bigramas se listan en dos versiones, con raíces y legible.**
La versión con *stemming* es la oficial (coherente con D-04), pero las raíces cortadas hacen
difícil juzgar si un bigrama aporta información. La versión legible se incluye solo como apoyo a
la interpretación. Son dos rankings independientes, no una traducción fila por fila: el orden no
coincide exactamente porque el stemming redistribuye los conteos (ver H-09).

---

## Sección 4 — TF-IDF

**R-08. La matriz TF-IDF tiene la misma forma que la de BoW: `(1140, 13080)`.**
No cambia qué se representa sino cuánto pesa cada término: las celdas pasan de `int64` a pesos
`float64`. El IDF va de **1.621** (término más extendido) a **7.347** (términos únicos).

**V-04. Se verificaron dos propiedades de `TfidfVectorizer` en vez de darlas por sentadas.**
1. Todas las filas tienen **norma L2 = 1** (mín. y máx. coinciden en 1.000000). Esto neutraliza la
   longitud del documento y es la razón de que en la Sección 5 la similitud coseno se reduzca a un
   producto punto.
2. El vector `idf_` de sklearn reproduce exactamente `ln((1+n)/(1+df))+1` calculado a mano
   (`np.allclose` → True). Importa porque es la variante **suavizada** de scikit-learn, no la
   fórmula de libro de texto, y el reporte debe citar la que realmente se usó.

**D-11. Los 3 documentos se eligen por criterio reproducible, no a dedo.**
En cada categoría se toma el documento cuya longitud es la más cercana a la mediana de esa
categoría, para no analizar casos extremos. Resultado: Macroeconomía doc 1010 (248 tokens),
Sostenibilidad doc 81 (393), Innovación doc 134 (288).

**D-12. Se construyó un glosario raíz → palabra completa para poder leer las tablas.**
Con raíces cortadas (`desperdici`, `inteligent`, `reapertur`) es imposible juzgar si el resultado
tiene sentido. El glosario mapea cada raíz a su forma legible más frecuente en el corpus. Es solo
ayuda de lectura: **todos los cálculos siguen hechos sobre las raíces** (coherente con D-04).

**R-09. Coincidencia entre el top-10 por TF-IDF y el top-10 por conteo:**
Macroeconomía 6/10, Sostenibilidad 4/10, Innovación 7/10.

**H-12. `bbva` es el mejor ejemplo del efecto del IDF en todo el laboratorio.**
Es la palabra **más frecuente del documento de Innovación** (8 apariciones) y aun así **no entra**
al top-10 por TF-IDF. Aparece en 484 de 1,140 documentos (42% del corpus), así que no distingue
ese artículo de ningún otro. El conteo simple la corona, el IDF la elimina: es la diferencia entre
ambas medidas concentrada en un solo término. Enlaza con R-02, donde `bbva` ya había salido como
el unigrama más frecuente del corpus.

**H-13. `sostenible` queda fuera del top-10 del documento de Sostenibilidad.**
Aparece 4 veces en el documento pero está en 243 documentos del corpus. Dentro de un corpus de
noticias de sostenibilidad, decir "sostenible" no informa nada; lo informativo es `desperdicio` y
`envases`. Es un argumento redondo de que el IDF aprende qué es genérico **en este dominio**, algo
que ninguna lista fija de stopwords del español podría anticipar.

**H-14. El grado de coincidencia entre ambos rankings es informativo por sí mismo.**
Innovación coincide en 7/10 porque las palabras que más repite (`artificial`, `factory`, `ai`,
`startups`) ya son raras en el corpus, así que ambos criterios apuntan al mismo lado.
Sostenibilidad coincide solo en 4/10 porque se apoya en vocabulario común (`productos`, `años`,
`cada`, `vez`) y ahí el reordenamiento del IDF es máximo. **La coincidencia mide cuán
especializado es el vocabulario del documento.**

**H-15. TF-IDF saca a la superficie los nombres propios.**
`Powell`, `Greta Thunberg`, `Sofia`, `Navarra` entran al top-10 apareciendo solo 2 o 3 veces,
porque su IDF ronda 5–7. El conteo simple no los muestra nunca. Son justamente los términos que
mejor identifican una noticia concreta.

**H-16. Términos expulsados por el IDF** (top-10 por conteo que no llegan al top-10 por TF-IDF):

| Documento | Término | Veces | IDF | Documentos |
|---|---|---|---|---|
| Macroeconomía | `economía` | 3 | 1.70 | 565 de 1,140 |
| Macroeconomía | `parte` | 3 | 1.69 | 572 |
| Sostenibilidad | `cada` / `vez` | 4 | 2.10 / 2.16 | 378 / 356 |
| Sostenibilidad | `sostenible` | 4 | 2.54 | 243 |
| Innovación | `bbva` | 8 | 1.86 | 484 |
| Innovación | `empresas` | 6 | 1.88 | 473 |

`cada` y `vez` reaparecen aquí después de haber sido los bigramas de ruido de H-07: el IDF los
degrada automáticamente, sin necesidad de una lista manual.

**P-03. Dos afirmaciones del borrador se corrigieron contra la fuente antes de fijarlas.**
Se había escrito que `greta thunberg` era un bigrama (son dos unigramas independientes) y que el
artículo de Macroeconomía trataba del debate sobre si la inflación era "transitoria". Al leer el
documento 1010 se confirmó el tema (IPC de junio, "mayor aumento desde 2008", costos de la
reapertura) pero **no** aparece esa discusión. Ambas se ajustaron a lo que el texto dice. Anotado
como recordatorio de verificar contra la fuente cualquier interpretación de un documento concreto
antes de meterla al reporte.

---

## Pendientes

- Secciones 5 y 6.
- Redactar con palabras propias las discusiones marcadas en el notebook (1.4, 2.6, 3.4).
- Actualizar en la celda 1.4 el conteo de documentos: `(1142, 13080)` → `(1140, 13080)`.
