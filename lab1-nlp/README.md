# Laboratorio #1 — NLP: Preparación de un corpus y EDA

Preparación y **Análisis Exploratorio de Datos (EDA)** del corpus
**Spanish News Classification** (noticias reales en español, clasificadas por categoría),
usando Python, `pandas` y `NLTK`.

**Curso:** Natural Language Processing  
**Autor:** Milton Beltrán

## Contenido del laboratorio

1. Carga y exploración del corpus (documentos, columnas, categorías, nulos y duplicados).
2. Pipeline de normalización: tokenización → minúsculas → eliminación de puntuación →
   eliminación de *stopwords* → *stemming* (con conteo de tokens y tipos en cada paso).
3. Investigación teórica sobre el EDA.
4. EDA del corpus: riqueza léxica, top-20 de frecuencias, histograma de longitudes,
   Ley de Zipf, comparación por categoría y nube de palabras.
5. Análisis de resultados.

## Estructura del repositorio

```
lab1-nlp/
├── lab1.ipynb                     # Notebook: código completo (carga, normalización y EDA)
├── df_total.csv                   # Corpus (Spanish News Classification)
├── requirements.txt               # Dependencias de Python
├── README.md
├── reporte/
│   ├── reporte_laboratorio1.md    # Fuente del reporte (Markdown)
│   ├── build_reporte.py           # Script Markdown → PDF (WeasyPrint)
│   ├── Reporte_Laboratorio1.pdf   # Reporte final (entregable)
│   └── img/                        # Figuras del EDA generadas por el notebook
└── devdocs/                        # Material de clase y notas (ignorado por git)
```

## Requisitos y configuración del entorno

Entorno virtual local creado con el Python de **miniforge** (3.13):

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Descargar los datos de NLTK necesarios (tokenizador y *stopwords*):

```bash
.venv/bin/python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

Registrar el kernel de Jupyter (incluye la variable `NLTK_DISABLE_IMPORT_SECURITY=1`,
necesaria porque el `.venv` vive dentro del proyecto):

```bash
.venv/bin/python -m ipykernel install --user --name lab1-nlp --display-name "Python (lab1-nlp)"
```

## Cómo ejecutar

**Notebook:** abrir `lab1.ipynb` en Jupyter/VS Code, seleccionar el kernel
**"Python (lab1-nlp)"** y ejecutar todas las celdas (Kernel → Restart & Run All).
Las figuras del EDA se guardan en `reporte/img/`.

**Reporte PDF:** se genera a partir del Markdown con WeasyPrint. WeasyPrint requiere
librerías del sistema; se instala fácilmente vía conda-forge (miniforge):

```bash
conda install -c conda-forge weasyprint markdown
python reporte/build_reporte.py   # genera reporte/Reporte_Laboratorio1.pdf
```

## Entregables

- **Notebook** `lab1.ipynb` — código completo y comentado (carga, preparación y EDA).
- **Reporte** `reporte/Reporte_Laboratorio1.pdf` — investigación sobre EDA, EDA del
  corpus y análisis de resultados.
