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
├── README.md
├── reporte/
│   ├── reporte_laboratorio1.md    # Fuente del reporte (Markdown)
│   ├── build_reporte.py           # Script Markdown → PDF (WeasyPrint)
│   ├── Reporte_Laboratorio1.pdf   # Reporte final (entregable)
│   └── img/                        # Figuras del EDA generadas por el notebook
└── devdocs/                        # Material de clase y notas (ignorado por git)
```

## Requisitos y configuración del entorno

El entorno virtual y las dependencias son **comunes a todo el repositorio** y viven en la
raíz (`.venv/` y `requirements.txt`). Ver el
[README de la raíz](../README.md) para las instrucciones de instalación.

## Cómo ejecutar

**Notebook:** abrir `lab1.ipynb` en Jupyter/VS Code, seleccionar el kernel
**"Python (NLP)"** y ejecutar todas las celdas (Kernel → Restart & Run All).
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
