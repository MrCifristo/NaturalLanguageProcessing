# Natural Language Processing — Laboratorios

Repositorio de los laboratorios del curso de **Natural Language Processing**.

**Autor:** Milton Beltrán

| Laboratorio | Tema | Carpeta |
|-------------|------|---------|
| #1 | Preparación de un corpus y EDA | [`lab1-nlp/`](lab1-nlp/) |
| #2 | Representaciones básicas de texto (BoW, n-gramas, TF-IDF, similitud coseno) | [`lab2-nlp/`](lab2-nlp/) |
| #3 | Clasificación de texto con Naive Bayes | [`lab3-nlp/`](lab3-nlp/) |
| #4 | Modelos lineales: regresión logística | [`lab4-nlp/`](lab4-nlp/) |
| #5 | Modelos de lenguaje n-grama, suavizado y perplejidad | [`lab5-nlp/`](lab5-nlp/) |

Los laboratorios **#1 a #4** trabajan sobre el mismo corpus: **Spanish News Classification**
(`df_total.csv`, noticias reales en español clasificadas en 7 categorías).

El **#5** cambia de corpus y de objetivo: usa el texto de ***Don Quijote de la Mancha***
([dataset de Kaggle](https://www.kaggle.com/datasets/manuelmaaf97/quijote)) y, en lugar de
clasificar documentos, predice la siguiente palabra de una secuencia. Ese corpus no se guarda
en el repositorio: el notebook lo descarga con `kagglehub`, que no requiere credenciales.

## Estructura

```
NaturalLanguageProcessing/
├── .venv/              # Entorno virtual compartido (ignorado por git)
├── requirements.txt    # Dependencias comunes a todos los laboratorios
├── README.md
├── lab1-nlp/           # Laboratorio #1
├── lab2-nlp/           # Laboratorio #2
├── lab3-nlp/           # Laboratorio #3
├── lab4-nlp/           # Laboratorio #4
└── lab5-nlp/           # Laboratorio #5
```

El entorno y las dependencias son **compartidos**: viven en la raíz, no dentro de cada
laboratorio. Cada carpeta `labN-nlp/` contiene únicamente lo específico de esa entrega
(notebook, corpus, reporte y enunciado).

## Configuración del entorno

Entorno virtual creado con el Python de **miniforge** (3.13):

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Datos de NLTK necesarios (tokenizador y *stopwords*):

```bash
.venv/bin/python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

Kernel de Jupyter. Se registra con la variable `NLTK_DISABLE_IMPORT_SECURITY=1`, necesaria
porque NLTK 3.10 bloquea el import de `regex` cuando el `.venv` vive dentro del directorio
de trabajo (falso positivo de su hook de seguridad):

```bash
.venv/bin/python -m ipykernel install --user --name nlp \
    --display-name "Python (NLP)" --env NLTK_DISABLE_IMPORT_SECURITY 1
```

Los notebooks se ejecutan seleccionando el kernel **"Python (NLP)"**.

## Reportes en PDF

Los reportes se escriben en Markdown y se convierten a PDF con **WeasyPrint**, que
requiere librerías del sistema y se instala vía conda-forge (miniforge):

```bash
conda install -c conda-forge weasyprint markdown
python labN-nlp/reporte/build_reporte.py
```
