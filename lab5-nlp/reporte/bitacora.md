# Bitácora de trabajo — Laboratorio #5

Registro de hallazgos, decisiones y problemas encontrados durante el desarrollo.
Sirve como material de respaldo para el reporte escrito: cada afirmación del reporte
debería poder rastrearse hasta una entrada de aquí.

**Corpus:** *El ingenioso hidalgo Don Quijote de la Mancha*, Miguel de Cervantes
(Kaggle: `manuelmaaf97/quijote`) · **Notebook:** `lab5.ipynb`

---

## Entorno y estructura

**D-01. Corpus nuevo: se rompe con los laboratorios #1–#4.**
Los cuatro laboratorios anteriores trabajaron sobre *Spanish News Classification*
(`df_total.csv`). El Lab #5 cambia de corpus por completo: el enunciado exige el texto del
*Quijote*. No se reutiliza nada del pipeline anterior salvo la estructura del notebook.

**D-02. El pipeline de normalización se invierte.**
Todo lo que los labs #1–#4 quitaban, aquí se conserva:

| Labs #1–#4 (clasificación) | Lab #5 (modelado de lenguaje) |
|---|---|
| Se eliminan stopwords | Se conservan: son las palabras que un LM más predice |
| Stemming con `SnowballStemmer` | Ninguno: se predice la palabra de superficie |
| Bolsa de palabras, orden irrelevante | El orden **es** el dato |
| Unidad de análisis: el documento | Unidad de análisis: la **oración**, con `<s>` y `</s>` |
| Partición estratificada por etiqueta | Partición aleatoria 80/10/10 (no hay etiquetas) |

**D-03. Sin dependencias nuevas de NLP.**
El enunciado prohíbe librerías especializadas en modelos de lenguaje (`nltk.lm`, `kenlm`);
los modelos se implementan desde cero con `defaultdict`. Para sentenizar y tokenizar basta
NLTK (`punkt`/`punkt_tab` en español), ya instalado desde el Lab #1; spaCy es opcional en el
enunciado y no se usa. Única adición: `kagglehub`, solo para la descarga del corpus.

**D-04. Numeración de las secciones: esta vez el enunciado es consistente.**
A diferencia de los labs #3 (D-03) y #4 (D-03), aquí no hay corrimiento: el PDF numera 1–6 y
la tabla de entregables remite al análisis como sección 6, que es la que efectivamente está
titulada "Análisis". Se respeta la numeración del enunciado.

**D-05. Se añade generación de texto, no pedida explícitamente en la sección 5.**
La sección 6 pregunta por "los resultados de perplejidad **y de generación de texto** entre
unigrama, bigrama y trigrama", pero la sección 5 solo pide una función de autocompletado
top-5. Para poder responder esa pregunta con evidencia propia se añade una celda que genera
oraciones completas por muestreo con cada uno de los tres modelos.

**D-06. El corpus se descarga con `kagglehub`; no se guarda copia local en el repositorio.**
Los labs #1–#4 guardaban una copia de `df_total.csv` dentro de cada carpeta para que el
notebook fuera autocontenido. Aquí se rompe esa costumbre: `kagglehub.dataset_download(
"manuelmaaf97/quijote")` descarga el corpus **sin pedir credenciales** —el dataset es
público— y lo cachea en `~/.cache/kagglehub/`. Se comprobó ejecutándolo en limpio: descarga
817 KB y devuelve la ruta de la versión 1.

Razón: la descarga es pública y reproducible, así que el notebook se sigue pudiendo correr
de cero en cualquier máquina, que es el sentido que importa de "autocontenido", sin arrastrar
una quinta copia de un corpus al repositorio. Costo asumido: el notebook depende de que el
dataset siga publicado en Kaggle y de que haya red en la primera ejecución.

---

## Sección 1 — Preparación del corpus

### Inspección del archivo crudo

El archivo entregado por Kaggle es `don-quijote.txt`: 2.2 MB, 37,861 líneas, ~384,000
palabras. Antes de escribir el pipeline se inspeccionó a nivel de bytes, y tiene cuatro
particularidades que obligan a limpiar antes de sentenizar. Ninguna es visible si uno se
limita a abrir el archivo y mirar el principio.

**H-01. No es un texto plano del Quijote: es el ebook #2000 de Project Gutenberg.**
Trae cabecera y pie de licencia **en inglés**, delimitados por marcadores explícitos:

| Marcador | Línea |
|---|---|
| `*** START OF THIS PROJECT GUTENBERG EBOOK DON QUIJOTE ***` | 19 |
| `*** END OF THIS PROJECT GUTENBERG EBOOK DON QUIJOTE ***` | 37,501 |
| fin del archivo | 37,861 |

Son 19 líneas de cabecera y **360 líneas de texto legal en inglés** al final. Sin recortarlas,
el modelo de lenguaje aprendería a predecir secuencias como *"Project Gutenberg-tm electronic
works"* dentro de un corpus que se supone es español del siglo XVII, contaminando vocabulario,
conteos de n-gramas y perplejidad.

**D-07. Se recorta el texto entre los dos marcadores `*** START ... ***` y `*** END ... ***`.**
Se usan los marcadores y no números de línea fijos, para que el recorte no se rompa si el
dataset cambia de versión.

**H-02. El archivo está codificado en UTF-8 **con BOM** y con saltos de línea CRLF.**
`file` lo reporta como *"Unicode text, UTF-8 (with BOM) text, with CRLF line terminators"*.
Leerlo con `encoding="utf-8"` a secas deja el carácter `\ufeff` pegado al primer token del
corpus.

**D-08. Se lee con `encoding="utf-8-sig"`**, que consume el BOM. Los `\r` del CRLF los
normaliza Python al leer en modo texto.

**H-03. El texto tiene *wrap* duro a ~76 columnas: las oraciones cruzan varias líneas físicas.**
Este es el hallazgo con más consecuencias. Ejemplo real del archivo (líneas 15,002–15,004):

```
Allí fue el desear de la espada de Amadís, contra quien no tenía fuerza de
encantamento alguno; allí fue el maldecir de su fortuna; allí fue el
exagerar la falta que haría en el mundo su presencia el tiempo que allí
```

El salto de línea es tipográfico, no sintáctico: no marca fin de oración. Sentenizar línea
por línea partiría casi todas las oraciones a la mitad y produciría un corpus de fragmentos
sin sentido, con `<s>` y `</s>` en lugares arbitrarios. El daño no sería solo cualitativo:
inflaría el número de oraciones, contaminaría los conteos de bigramas y trigramas en las
fronteras falsas, y haría incomparable la perplejidad.

**D-09. Se reconstruyen los párrafos antes de sentenizar.**
Los párrafos están delimitados por líneas en blanco. El orden del pipeline es: recortar
Gutenberg → unir las líneas de cada párrafo en una sola cadena → sentenizar el párrafo
completo con `nltk.sent_tokenize(..., language="spanish")` → tokenizar cada oración.

**H-04. El texto incluye 126 encabezados de capítulo y una sección de preliminares.**
Los encabezados tienen la forma `Capítulo primero. Que trata de la condición y ejercicio del
famoso hidalgo`, y antes del capítulo I hay material que no es prosa narrativa: la TASA (fe de
un escribano sobre el precio del libro), el privilegio real, y los versos preliminares.

**H-05. Recortar por los marcadores no elimina todo el aparato editorial: quedan dos líneas
en inglés dentro de la región delimitada.** Verificado al ejecutar la sección 0:

```
Produced by an anonymous Project Gutenberg volunteer. Text
End of Project Gutenberg's Don Quijote, by Miguel de Cervantes Saavedra
```

La primera es el crédito del transcriptor, situado *después* del marcador `*** START ... ***`;
la segunda es la línea de cierre, situada *antes* del `*** END ... ***`. El recorte por
marcadores descarta 19,334 caracteres (0.9 % del archivo) y deja el texto en 2,098,163
caracteres, pero estas dos líneas sobreviven. Son pocas —irrelevantes frente a ~384,000
palabras— pero están en inglés y no son prosa de Cervantes, así que se limpian junto con los
encabezados de capítulo en la sección 1. El cuerpo real del libro empieza en el título
`El ingenioso hidalgo don Quijote de la Mancha` y termina en la palabra `Fin`.

**H-06. El BOM y los CRLF quedan resueltos con `utf-8-sig` y la lectura en modo texto.**
Comprobado: tras leer con `utf-8-sig`, `texto_crudo.startswith("\ufeff")` es `False`, y no
queda ningún `\r` porque Python normaliza los saltos de línea al leer en modo texto. El
archivo son 2,117,497 caracteres en 37,861 líneas.

**D-11. Recorte fino por anclas del propio libro (celda 0.7).** Como los marcadores de
Gutenberg no bastan (H-05), el cuerpo se delimita entre la portadilla `El ingenioso hidalgo
don Quijote de la Mancha` y la línea que contiene solo `Fin`. Para el cierre se usa `rindex`
sobre `"\nFin\n"` —la línea exacta, buscada desde el final— porque un `index` sobre `"Fin"`
atraparía cualquiera de las decenas de `Finalmente` o el `Finis` que cierra el libro de 1605.
Resultado: **2,097,945 caracteres y cero líneas con "Gutenberg"**.

### Pipeline y particiones

**H-07. Los encabezados de capítulo no producen una oración mala: producen dos.**
Verificado ejecutando Punkt sobre el primer encabezado:

```
Capítulo primero. Que trata de la condición y ejercicio del famoso hidalgo
don Quijote de la Mancha
        ↓ sent_tokenize(..., language="spanish")
   · "Capítulo primero."
   · "Que trata de la condición y ejercicio del famoso hidalgo don Quijote de la Mancha"
```

La primera es una "oración" de dos palabras; la segunda, un fragmento sin verbo principal.
Cada una recibiría sus propios `<s>` y `</s>`, de modo que el daño no cae en cualquier
conteo sino justo en las transiciones de inicio y de fin de oración, que son las que un
modelo n-grama usa para decidir cómo empieza y cómo termina una frase.

**D-10 (resuelta). Se eliminan los encabezados estructurales; se conservan los preliminares.**
Se descartan los 126 encabezados de capítulo y las 5 divisiones de parte (`Primera/Segunda/
Tercera/Cuarta parte del ingenioso...`): **131 párrafos de 5,191**, detectados por expresión
regular sobre el inicio del párrafo ya reconstruido. El criterio decisivo no fue el volumen
—era despreciable, como decía el argumento en contra— sino H-07: no son texto mal
proporcionado, son texto mal formado.

Los preliminares (TASA, testimonio de erratas, privilegio real, dedicatoria, prólogo y versos)
**se conservan**: son español de la época con oraciones bien formadas, y delimitarlos exigiría
reglas frágiles sobre encabezados en mayúsculas. Costo asumido y medido: sobreviven unas 30
"oraciones" de un solo token que son rótulos (`tasa`, `prólogo`, `soneto` ×8), un 0.3 % de las
oraciones del corpus.

**H-08. `word_tokenize` de NLTK deja pegados dos signos que el español usa constantemente.**
El tokenizador de NLTK está diseñado para el inglés. Sobre este corpus produce:

| Entrada | Sin arreglo | Correcto |
|---|---|---|
| `-Porque` | `-porque` | `-` + `porque` |
| `¿cómo` | `¿cómo` | `¿` + `cómo` |

No es cosmético. El Quijote es casi todo diálogo —el texto trae **7,032 rayas**, 959 `¿` y
682 `¡`— así que una misma palabra frecuente entraba al vocabulario hasta tres veces:
`porque`, `-porque` y `¿porque`. Medido sobre el corpus completo son **1,299 tipos espurios**.
Es dispersión artificial: no viene de la lengua sino de la herramienta, y es la peor clase
porque el suavizado de la sección 3 gastaría masa de probabilidad en tapar un error propio.

| Tokenización | Vocabulario |
|---|---|
| Cruda | 23,947 |
| Separando la raya | 23,190 |
| Separando raya y `¿` `¡` | **22,873** |

**D-12. La raya solo se separa cuando no tiene letra a ambos lados.**
`(?<![letra])-|-(?![letra])` deja intactos compuestos del tipo *físico-químico*. Este corpus
no tiene ninguno —tras el arreglo quedan **0 tipos con guion interno**—, pero la regla
correcta es esa y no un `replace("-", " - ")` a ciegas. Los signos se separan **antes** de
sentenizar, para que Punkt vea `¿` y `-` como piezas sueltas y no como parte de la palabra.

**D-13. Todo se pasa a minúsculas.** `En` y `en` son la misma palabra; distinguirlas solo
porque una abre oración parte los conteos en dos sin aportar información. Cuesta la
distinción de nombres propios (*Quijote* / *quijote*), que este laboratorio no necesita.

**D-14. La puntuación se conserva como tokens.** Forma parte de la secuencia que un modelo de
lenguaje predice, y en el autocompletado de la sección 5 una coma es una sugerencia legítima.
Consecuencia visible: los tokens más frecuentes del corpus son `,` (40,083) y `.` (7,869).

**H-09. Corpus final tras el pipeline: 10,572 oraciones y 444,836 tokens**, con una longitud
media de **42.1 tokens por oración** y una máxima de **555**. Las oraciones de Cervantes son
larguísimas para lo que un n-grama espera, y eso tiene una consecuencia directa en la sección
2: multiplicar 42 probabilidades menores que 1 —557 en el peor caso— desborda por abajo la
precisión de punto flotante. Los cálculos tendrán que hacerse en log-espacio, igual que en el
Lab #3 (su H-18).

**D-15. La partición reparte ORACIONES, aleatoriamente, con `random.seed(42)`.**
Aleatoria y no estratificada, porque aquí no hay etiquetas que balancear (la razón que sí
aplicaba en el Lab #3). La unidad es la oración y no el párrafo o el capítulo: cada oración se
evalúa de forma independiente empezando en `<s>`, así que no hay contexto que se filtre de una
partición a otra. Semilla 42, la misma de los labs #3 y #4.

**H-10. Tamaños de la partición.**

| Partición | Oraciones | % | Tokens |
|---|---|---|---|
| Entrenamiento | 8,457 | 80.0 % | 373,242 |
| Validación | 1,057 | 10.0 % | 45,780 |
| Prueba | 1,058 | 10.0 % | 46,958 |

**H-11. Vocabulario 20,497 tipos, y la mitad se estimó con una sola observación.**
Los *hapax legomena* —palabras que aparecen exactamente una vez en entrenamiento— son
**10,277, el 50.1 % del vocabulario**. Ese es el dato que explica el OOV: si media lengua
aparece una sola vez en 373 mil palabras, es inevitable que otra porción parecida no aparezca
ninguna.

La tasa de OOV se midió de las dos formas posibles, porque responden a preguntas distintas y
dan cifras muy separadas:

| Partición | OOV por token | OOV por tipo |
|---|---|---|
| Validación | 2.72 % | 19.39 % |
| Prueba | **2.82 %** | **19.60 %** |

La brecha entre 2.8 % y 19.6 % dice que las palabras desconocidas son casi todas raras: las
frecuentes se aprenden con cualquier corpus, la cola larga no. Aun así, **el 53.3 % de las
oraciones de prueba contiene al menos un OOV**, que es la cifra que de verdad importa cuando
se evalúa una oración completa.

**H-12. La dispersión medida sobre n-gramas, que es lo que el modelo necesita haber visto.**
El OOV mide dispersión a nivel de palabra, pero un modelo n-grama no necesita conocer la
palabra: necesita haber visto la **secuencia**. Proporción de n-gramas de validación que no
aparecen ni una vez en entrenamiento:

| n | No vistos en entrenamiento | % |
|---|---|---|
| 1 (palabras) | 1,243 / 45,780 | **2.7 %** |
| 2 (bigramas) | 10,969 / 44,723 | **24.5 %** |
| 3 (trigramas) | 26,287 / 43,666 | **60.2 %** |

Este es el resultado que justifica las secciones 3 y 4 completas. Con estimación por máxima
verosimilitud un solo n-grama de cuenta cero anula la probabilidad de toda la oración, porque
la regla de la cadena multiplica. Con un 60.2 % de trigramas no vistos, el modelo de
trigramas asignaría probabilidad **cero** a casi cualquier oración nueva y su perplejidad
sería infinita: no estaría diciendo "esta oración es improbable" sino "esta oración es
imposible", que es falso. Y es la medición directa del compromiso contexto/dispersión que
pregunta la sección 6: de n=1 a n=3 el modelo gana contexto y pierde datos, 2.7 % → 60.2 %.

---

## Sección 2 — Modelos n-grama

**D-16. Un solo `ModeloNGrama(n)` para los tres modelos, con `k` de suavizado desde el
principio.** Dos `defaultdict(int)`: `conteo` guarda el n-grama completo y `contexto` sus
n−1 primeras palabras, que es el denominador. `k=0` es máxima verosimilitud, así que la
sección 3 no necesita código nuevo, solo pasar otro `k`.

**D-17. El `<s>` se replica n−1 veces y el unigrama lo descarta.**
Un trigrama necesita dos `<s>` para poder estimar P(w₁ | `<s>`, `<s>`); las oraciones se
guardan con uno solo, así que el padding se aplica en el modelo. Con n=1 se descarta el
`<s>`: el unigrama no lo predice ni condiciona en él.

El efecto buscado es que **los tres modelos predigan exactamente los mismos tokens** —las
palabras más `</s>`— y que sus perplejidades sean comparables en la sección 4. Verificado
sobre la oración de ejemplo: 13 tokens predichos con n=1, 2 y 3.

**P-01. Un `defaultdict` inserta la clave al consultarla.**
Leer `self.conteo[grama]` para un n-grama no visto lo añade a la tabla con valor 0. Evaluar
el corpus de validación completo llenaría las tablas de miles de entradas fantasma y
cambiaría `len(self.conteo)` entre una celda y otra. En `prob()` se lee con `.get(grama, 0)`;
`defaultdict` se sigue usando donde sirve, que es acumular en el entrenamiento.

**H-13. Tamaño de las tablas: cada modelo dobla los parámetros con los mismos datos.**

| n | n-gramas distintos | Contextos distintos |
|---|---|---|
| 1 | 20,496 | 1 |
| 2 | 124,041 | 20,496 |
| 3 | 248,584 | 123,996 |

Los tres se estiman con los mismos 373,242 tokens, así que cada entrada del trigrama se
apoya en la mitad de evidencia que una del bigrama. Es H-12 visto desde el lado del modelo.

**H-14. El contexto paga, y se puede medir en una sola palabra.**

| Estimación | Cuenta | Probabilidad |
|---|---|---|
| P(quijote) | 1,703 / 364,785 | 0.0047 |
| P(quijote \| don) | 1,697 / 2,069 | **0.8202** |
| P(quijote \| respondió don) | 212 / 223 | **0.9507** |

**H-15. La oración de ejemplo: el bigrama gana 12 órdenes de magnitud y el trigrama da cero.**
Oración de validación `- y yo lo digo también - respondió don quijote - .`, 13 tokens
predichos:

| Modelo | log P | P |
|---|---|---|
| Unigrama | −65.179 | 4.93 × 10⁻²⁹ |
| Bigrama | −38.270 | 2.40 × 10⁻¹⁷ |
| Trigrama | −inf | **0** |

**H-16. El trigrama se cae por UN solo n-grama de trece.**
El desglose muestra que doce de los trece trigramas de la oración sí están en entrenamiento,
varios con contextos muy fuertes (`respondió don quijote` aparece 212 veces sobre 223). El
único ausente es `lo digo también`, con cuenta 0. Como la regla de la cadena multiplica, ese
cero anula el producto entero. Es la demostración concreta de por qué hace falta suavizar: el
modelo no está diciendo que la oración sea rara, está diciendo que es imposible.

**H-17. Sin logaritmos el cálculo no da un número pequeño, da cero.**
La oración más larga de entrenamiento tiene 329 palabras. Su log-probabilidad bajo el
unigrama es **−2,050.5**, pero el producto directo de sus factores devuelve **0.0**, porque
el menor float positivo representable es 5 × 10⁻³²⁴. Sin log-espacio el modelo declararía
imposible una oración que está en su propio corpus de entrenamiento. Mismo problema y misma
solución que en el Lab #3 (su H-18).

---

## Sección 3 — Suavizado

<!-- Laplace, add-k con al menos dos valores de k, comparación con/sin suavizado. -->

---

## Sección 4 — Perplejidad

<!-- PPL de los tres modelos en validación, mejor modelo en prueba, barrido de k. -->

---

## Sección 5 — Autocompletado

<!-- Función top-5, cinco fragmentos de prueba, comentario sobre el español del s. XVII. -->

---

## Sección 6 — Análisis

<!-- Respuestas a las cuatro preguntas. -->

---

## Problemas encontrados

<!-- P-01, P-02, ... -->
