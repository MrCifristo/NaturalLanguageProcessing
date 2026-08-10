# Natural Language Processing — Laboratorios

Repositorio de los laboratorios del curso de **Natural Language Processing**.

**Autor:** Milton Beltrán

| Laboratorio | Tema | Carpeta |
|-------------|------|---------|
| #1 | Preparación de un corpus y EDA | [`lab1-nlp/`](lab1-nlp/) |
| #2 | Representaciones básicas de texto (BoW, n-gramas, TF-IDF, similitud coseno) | [`lab2-nlp/`](lab2-nlp/) |

Ambos laboratorios trabajan sobre el mismo corpus: **Spanish News Classification**
(`df_total.csv`, noticias reales en español clasificadas en 7 categorías).

## Estructura

```
NaturalLanguageProcessing/
├── .venv/              # Entorno virtual compartido (ignorado por git)
├── requirements.txt    # Dependencias comunes a todos los laboratorios
├── README.md
├── lab1-nlp/           # Laboratorio #1
└── lab2-nlp/           # Laboratorio #2
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
