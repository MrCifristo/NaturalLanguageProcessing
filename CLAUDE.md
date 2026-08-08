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

- Laboratorio #1 de NLP: preparación de un corpus ("Spanish News Classification") y EDA.
- Corpus en `df_total.csv` (columnas: `url`, `news`, `Type`), **1217 documentos**
  (el archivo tiene ~4570 líneas, pero las noticias traen saltos de línea internos;
  `pd.read_csv` los agrupa correctamente en 1217 filas). 7 categorías. Sin nulos.
  75 filas totalmente duplicadas y 79 con `news` repetido.
- Trabajo principal en `lab1.ipynb`.

## Entorno

- `.venv` local creado con el Python de miniforge (3.13).
- Librerías: pandas, nltk, matplotlib, wordcloud, jupyter/ipykernel (ver `requirements.txt`).
- Se usa **solo NLTK** para NLP en español; la "lematización" se hace con
  `SnowballStemmer('spanish')` (técnicamente stemming, no lematización real).
- Kernel de Jupyter registrado: "Python (lab1-nlp)". Incluye la variable
  `NLTK_DISABLE_IMPORT_SECURITY=1` para evitar un falso positivo del hook de
  seguridad de NLTK 3.10 (el `.venv` está dentro del cwd).
