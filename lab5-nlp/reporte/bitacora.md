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

**D-10. Pendiente:** decidir si se conservan o se eliminan los encabezados de capítulo y los
preliminares. Argumento para quitarlos: no son la lengua que el modelo dice modelar (prosa
narrativa y diálogo), y la fórmula repetida de los encabezados introduce n-gramas
artificialmente frecuentes. Argumento para dejarlos: son ~126 líneas frente a ~384,000
palabras, un efecto despreciable, y quitarlos añade un paso de limpieza más que puede fallar.
Se resolverá al implementar la sección y se documentará el criterio aquí.

### Pipeline y particiones

<!-- Tokenización, <s>/</s>, split 80/10/10, tamaño de vocabulario, tasa de OOV en prueba. -->

---

## Sección 2 — Modelos n-grama

<!-- Estructuras de conteo, probabilidad de la oración de ejemplo bajo cada modelo. -->

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
