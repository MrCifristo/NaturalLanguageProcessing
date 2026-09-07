# CLAUDE.md — Guía para el asistente

## Reglas de trabajo (IMPORTANTE)

- **NUNCA operar con git.** No hacer `git add`, `git commit`, `git push`, ni ninguna
  otra operación que modifique el repositorio o su historial. El usuario maneja su
  repositorio manualmente, siempre.
- Solo **avisar** al usuario cuando sea buen momento para hacer un commit (por ejemplo,
  al terminar una fase del laboratorio), pero es el usuario quien lo ejecuta.
- El usuario quiere **aprender haciendo**: guiar paso a paso, explicar el *qué* y el
  *por qué*, dar pistas y revisar. No resolver el laboratorio completo ni escribir
  todos los scripts por él. El usuario escribe su propio código en el notebook.

## Contexto del proyecto

- Repositorio de laboratorios del curso de NLP. Un directorio por entrega:
  `lab1-nlp/` … `lab5-nlp/`, los cinco entregados.
- **Lab #1:** preparación del corpus "Spanish News Classification" y EDA. Notebook
  `lab1-nlp/lab1.ipynb`.
- **Lab #2:** representaciones básicas de texto (BoW, dispersión/CSR, n-gramas, TF-IDF,
  similitud coseno). Notebook `lab2-nlp/lab2.ipynb`, **autocontenido**: repite la carga y
  el pipeline del Lab 1 antes de vectorizar. Vectoriza sobre `tokens_stem` (la salida
  completa del pipeline), unidos en una columna de texto porque los vectorizadores de
  scikit-learn reciben strings, no listas de tokens.
- **Lab #3:** clasificación con Naive Bayes Multinomial (particiones estratificadas, BoW vs
  TF-IDF, análisis de errores, `max_features`/bigramas, implementación propia del algoritmo).
  Notebook `lab3-nlp/lab3.ipynb`, también autocontenido. Su corpus quita **6 filas más** que el
  Lab #2 —tres pares con el mismo texto y etiquetas contradictorias— y trabaja con **1,134
  documentos**; sin eso habría fuga entre entrenamiento y prueba. Añade `seaborn` a las
  dependencias.
- **Lab #4:** modelos lineales, regresión logística (BoW vs TF-IDF, interpretación de pesos,
  comparación contra el Naive Bayes del Lab #3, efecto de la regularización `C` y `l1`/`l2`,
  análisis de errores). Notebook `lab4-nlp/lab4.ipynb`, autocontenido: reconstruye las
  particiones y los vectorizadores del Lab #3 y lo verifica con cinco `assert`. Añade `scipy`
  (test de McNemar).
- **Lab #5:** modelos de lenguaje n-grama, **el laboratorio que rompe con los cuatro
  anteriores**. Notebook `lab5-nlp/lab5.ipynb`. Cambia de corpus (*Don Quijote*, ver abajo) y
  de objetivo: ya no clasifica documentos sino que predice la siguiente palabra, así que el
  pipeline se **invierte** — se conservan las stopwords, no hay stemming, el orden es el dato,
  la unidad es la oración con `<s>`/`</s>` y la partición es aleatoria (no hay etiquetas).
  Uni/bi/trigramas implementados desde cero con `defaultdict` (el enunciado prohíbe `nltk.lm`
  y `kenlm`), suavizado add-k, perplejidad y autocompletado. **No usa scikit-learn.**
- Corpus en `df_total.csv` (columnas: `url`, `news`, `Type`), **1217 documentos**
  (el archivo tiene ~4570 líneas, pero las noticias traen saltos de línea internos;
  `pd.read_csv` los agrupa correctamente en 1217 filas). 7 categorías. Sin nulos.
  75 filas totalmente duplicadas y 79 con `news` repetido. Tras `drop_duplicates()`
  quedan **1142 documentos**, que es la base de trabajo de los labs #1 a #4.
  Hay una copia del CSV dentro de cada `labN-nlp/` para que los notebooks sean autónomos.
- Corpus del **Lab #5**: *El ingenioso hidalgo don Quijote de la Mancha*, dataset público
  `manuelmaaf97/quijote` de Kaggle. **No se guarda copia en el repo**: se descarga en tiempo
  de ejecución con `kagglehub.dataset_download(...)`, que no pide credenciales y cachea en
  `~/.cache/kagglehub`. El archivo es el ebook #2000 de Project Gutenberg y trae cuatro
  trampas —cabecera y licencia en inglés, BOM, *wrap* duro a ~76 columnas que parte las
  oraciones, y 126 encabezados de capítulo—; todas documentadas en su bitácora. Tras
  limpiarlo: **10,572 oraciones, 444,836 tokens, vocabulario de 20,497 tipos**.

## Entorno

- **Compartido y en la raíz del repo**, no dentro de cada laboratorio: `.venv/` y
  `requirements.txt` viven en `/`. Lo específico de cada entrega (notebook, corpus,
  reporte, enunciado) vive en su `labN-nlp/`.
- `.venv` creado con el Python de miniforge (3.13).
- Librerías: pandas, numpy, nltk, matplotlib, wordcloud, **scikit-learn** (Lab 2),
  **seaborn** (Lab 3), **scipy** (Lab 4), **kagglehub** (Lab 5), jupyter/ipykernel.
- Se usa **solo NLTK** para NLP en español; en los labs #1–#4 la "lematización" se hace con
  `SnowballStemmer('spanish')` (técnicamente stemming, no lematización real).
  scikit-learn se usa únicamente para vectorizar y clasificar, nunca para tokenizar, y en el
  Lab #5 no se usa en absoluto.
- Kernel de Jupyter registrado: **"Python (NLP)"** (nombre interno `nlp`). Incluye la
  variable `NLTK_DISABLE_IMPORT_SECURITY=1` para evitar un falso positivo del hook de
  seguridad de NLTK 3.10 (el `.venv` está dentro del cwd). Sin esa variable, `import nltk`
  falla con *"Blocked import of regex from current working directory"*. El notebook del
  Lab #5 además la asigna con `os.environ[...]` **antes** de `import nltk`, para correr
  aunque el kernel seleccionado sea el `.venv` directo y no el registrado.
