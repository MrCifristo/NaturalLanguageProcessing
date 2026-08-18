# Bitácora de trabajo — Laboratorio #3

Registro de hallazgos, decisiones y problemas encontrados durante el desarrollo.
Sirve como material de respaldo para el reporte escrito: cada afirmación del reporte
debería poder rastrearse hasta una entrada de aquí.

**Corpus:** Spanish News Classification (`df_total.csv`) · **Notebook:** `lab3.ipynb`

---

## Entorno y estructura

**D-01. El notebook del Lab #3 vuelve a ser autocontenido.**
Misma decisión que en el Lab #2 (D-03 de aquella bitácora): `lab3.ipynb` reproduce en su
Sección 0 la carga y el pipeline de normalización de los labs anteriores, en lugar de
encadenarse a `lab2.ipynb`. Los notebooks ya entregados no se tocan y cada carpeta
`labN-nlp/` queda autónoma. Costo asumido: una tercera copia de `df_total.csv`.

**D-02. Dependencia nueva: `seaborn` 0.13.2.**
El enunciado sugiere `seaborn.heatmap` para las matrices de confusión. No estaba en el
`.venv` compartido; se instaló y se agregó a `requirements.txt` en un bloque propio del
Lab #3. Todo lo demás (scikit-learn 1.7.2, NLTK 3.10.1) ya estaba disponible.

**D-03. Numeración de las secciones: el enunciado tiene un hueco.**
El PDF numera las secciones 1–6, con la 6 titulada "Análisis". Pero la tabla de entregables
pide explícitamente "implementación propia de Naive Bayes y su comparación con scikit-learn"
y remite al análisis como **"sección 7"**; además, la pregunta 3 de ese análisis pregunta
"al implementar Naive Bayes desde cero, ¿qué parte le resultó más difícil...?". Se concluye
que falta una sección en el documento y se adopta esta numeración:

| Sección | Contenido |
|---------|-----------|
| 1 | Conjuntos de entrenamiento, validación y prueba |
| 2 | Construcción del clasificador Naive Bayes (BoW y TF-IDF) |
| 3 | Entrenamiento y evaluación |
| 4 | Análisis de errores |
| 5 | Efecto de la representación de texto |
| **6** | **Implementación propia de Naive Bayes** (sección ausente en el PDF) |
| 7 | Análisis |

---

## Sección 0 — Preparación del corpus

**H-01. Tres pares de noticias tienen el mismo texto y etiquetas contradictorias.**
Tras `drop_duplicates()` (1142 filas) y el descarte de los 2 documentos vacíos del Lab #2
(1140), quedan **6 filas con la columna `news` repetida**, agrupadas en 3 pares. En los tres
casos la *url* también es idéntica: es la misma noticia anotada dos veces con categorías
distintas.

| Noticia | Etiquetas de las dos copias |
|---------|-----------------------------|
| "Bbva en Colombia y el Sena se unieron…" | `Otra` / `Alianzas` |
| "Cemex Colombia y Bbva anunciaron una alianza…" | `Alianzas` / `Otra` |
| "TaxExpress y Uber anunciaron el lanzamiento de Uber Taxi…" | `Alianzas` / `Regulaciones` |

En el Lab #2 esto era irrelevante (no había etiquetas de por medio). En el Lab #3 hace daño
por dos vías: **fuga de información** si una copia cae en entrenamiento y la otra en prueba, y
un **techo imposible**, porque ninguna predicción puede acertar las dos etiquetas — y los
errores caerían justo en el par `Alianzas`/`Otra`.

**D-04. Se eliminan las 6 filas contradictorias (ambas copias de cada par).**
No hay criterio objetivo para decidir cuál de las dos etiquetas es la correcta, así que
quedarse con una sería inventar la anotación. Son 6 documentos de 1140 (0.5 %), efecto
despreciable en las métricas, y a cambio el split queda libre de fuga por construcción.
**Corpus definitivo del Lab #3: 1134 documentos, 7 categorías.**

**H-02. El corpus está fuertemente desbalanceado.**
Distribución final: Macroeconomia 319, Alianzas 244, Innovacion 152, Regulaciones 141,
Otra 128, Sostenibilidad 124 y **Reputacion 26**. La clase mayoritaria supera a la minoritaria
en proporción **12:1**. Con particiones de 15 %, a Reputacion le tocan **≈4 documentos** en
validación y 4 en prueba: cada acierto o fallo mueve su *recall* en 25 puntos. Esto obliga a
estratificar (Sección 1) y a leer el F1 macro con cuidado, porque esa clase pesa lo mismo que
Macroeconomia en el promedio macro pese a tener 12 veces menos evidencia.

---

## Sección 1 — Conjuntos de entrenamiento, validación y prueba

**D-05. El split se hace sobre los índices del DataFrame, no sobre las columnas sueltas.**
`train_test_split` recibe `df_prep.index` y de ahí salen `X_*_txt` / `y_*` con `.loc`. La
alternativa habitual —partir directamente `df_prep["texto_norm"].values`— rompe el vínculo con
la fila original. En la Sección 4 hay que leer el texto **sin stemming** (`news`) de los
documentos mal clasificados; con raíces cortadas ese análisis sería ilegible. Conservando los
índices, recuperar cualquier documento es un `df_prep.loc[i, "news"]`.

**D-06. `random_state = 42` fijo en ambos llamados.** Sin semilla fija, cada ejecución del
notebook produce métricas distintas y ninguna cifra del reporte sería reproducible.

**D-07. El segundo split usa `test_size=0.50`, no `0.15`.**
El segundo llamado no parte el corpus completo sino el 30 % restante, así que la mitad de ese
30 % es el 15 % del total. Y su `stratify` apunta a `y_total.loc[idx_temp]`, no a `y_total`:
estratificar contra las etiquetas del corpus entero en ese paso es un error silencioso, porque
`train_test_split` fallaría o desbalancearía Reputacion (4 documentos por conjunto).

**H-03. La estratificación funcionó: 793 / 170 / 171 documentos.**
Desviación máxima entre las proporciones de una misma categoría en los tres conjuntos:
**0.65 puntos porcentuales**. Ninguna categoría quedó ausente de ningún conjunto.

| Categoría | Entrenamiento | Validación | Prueba |
|-----------|---------------|------------|--------|
| Macroeconomia | 223 (28.1 %) | 48 (28.2 %) | 48 (28.1 %) |
| Alianzas | 171 (21.6 %) | 36 (21.2 %) | 37 (21.6 %) |
| Innovacion | 106 (13.4 %) | 23 (13.5 %) | 23 (13.5 %) |
| Regulaciones | 99 (12.5 %) | 21 (12.4 %) | 21 (12.3 %) |
| Otra | 89 (11.2 %) | 19 (11.2 %) | 20 (11.7 %) |
| Sostenibilidad | 87 (11.0 %) | 19 (11.2 %) | 18 (10.5 %) |
| Reputacion | 18 (2.3 %) | 4 (2.4 %) | 4 (2.3 %) |

Confirma H-02: Reputacion entrena con **18 documentos** y se evalúa con **4**. Cualquier
métrica suya (precision, recall, F1) se mueve en saltos de 25 puntos y no debe leerse como una
estimación estable.

---

## Sección 2 — Construcción del clasificador Naive Bayes

**D-08. Los vectorizadores se ajustan solo con entrenamiento (`fit_transform` en train,
`transform` en validación y prueba).** Es el requisito explícito del enunciado. En TF-IDF la
restricción es doble: el `fit` aprende el vocabulario **y** los IDF; calcular el IDF sobre el
corpus completo metería en el peso de cada palabra información sobre en cuántos documentos de
validación y prueba aparece — una fuga más sutil que la del vocabulario, pero fuga igual.

**H-04. Ajustar solo con entrenamiento cuesta 2,102 términos de vocabulario.**
El vocabulario aprendido en entrenamiento es de **10,968 términos**; ajustando con el corpus
completo serían **13,070** (+19 %). Esos 2,102 términos son exactamente la fuga que se evitó.
Su efecto real se midió sobre validación: de 45,918 ocurrencias, **1,460 (3.18 %)** caen fuera
del vocabulario y el modelo simplemente no las ve, repartidas en 1,114 términos distintos.
Es el precio honesto de no hacer trampa, y es bajo: el 96.8 % del texto de validación sí tiene
representación.

**D-09. `MultinomialNB` con parámetros por defecto: `alpha=1.0` y `fit_prior=True`.**
`alpha=1.0` es el suavizado de Laplace, necesario porque cualquier palabra con conteo cero en
una categoría anularía el producto entero. `fit_prior=True` estima P(c) de la frecuencia de
clases en entrenamiento en vez de asumirlas uniformes; dado el desbalance 12:1 (H-02), asumir
priors uniformes sería negar la estructura real del corpus.

**H-05. Las verosimilitudes aprendidas son interpretables y ya anticipan los errores.**
Midiendo cada raíz contra su segunda categoría más probable:

| Raíz | Categoría más probable | P(w\|c) | Ventaja sobre la 2ª |
|------|------------------------|---------|---------------------|
| `inflacion` | Macroeconomia | 0.0126 | 29× |
| `reput` | Reputacion | 0.0038 | 44× |
| `alianz` | Alianzas | 0.0073 | 14× |
| `emision` | Sostenibilidad | 0.0022 | 3.1× |
| `sostenibil` | Sostenibilidad | 0.0032 | 1.8× |
| `banc` | Otra | 0.0095 | 1.5× |

`reput` discrimina fortísimo pese a que Reputacion solo tiene 18 documentos de entrenamiento:
la verosimilitud se estima sobre *proporciones dentro de la clase*, así que una clase pequeña
puede tener palabras muy características. En el otro extremo, `banc` (1.5×) y `sostenibil`
(1.8×) son señales casi planas: el corpus entero habla de bancos y de sostenibilidad, así que
esas palabras no separan. Anticipa dónde van a estar las confusiones de la Sección 4.

---

## Sección 3 — Entrenamiento y evaluación

**D-10. El conjunto de prueba NO se evalúa en la Sección 3.**
El enunciado dice "evalúe sobre validación y, al final, sobre prueba". Se toma "al final" en
sentido literal: quedan decisiones pendientes (comparación de representaciones en la Sección 5,
implementación propia en la Sección 6), y medir en prueba antes de tomarlas convertiría ese
conjunto en un segundo set de validación. La evaluación en prueba va en una sección propia al
cierre del notebook.

**H-06. BoW gana a TF-IDF por 25 puntos de accuracy.**

| Representación | Conjunto | Accuracy | F1 macro | F1 ponderado |
|---|---|---|---|---|
| BoW | entrenamiento | 0.941 | 0.936 | 0.941 |
| BoW | **validación** | **0.800** | **0.732** | **0.795** |
| TF-IDF | entrenamiento | 0.710 | 0.548 | 0.663 |
| TF-IDF | validación | 0.547 | 0.360 | 0.474 |

**H-07. TF-IDF no clasifica: apuesta a la clase mayoritaria.**
De las 170 predicciones de validación, TF-IDF asigna **109 a Macroeconomia** (que solo tiene 48
documentos reales), 0 a Reputacion, 1 a Otra y 2 a Regulaciones. Su *precision* alta en varias
clases (1.00 en Otra, Regulaciones y Sostenibilidad) es un espejismo: predice esas clases
tan pocas veces que casi no se equivoca, mientras su *recall* se hunde a 0.05–0.16.

**H-08. La causa medida: la normalización L2 de TF-IDF deja al prior compitiendo de igual a
igual con la evidencia léxica.**
El score de cada clase es `log P(c) + Σ nᵢ·log P(wᵢ|c)`. Midiendo sobre validación cuánto separa
cada término a las clases:

| | Suma de valores por documento | Rango de la evidencia entre clases | Rango del prior | Proporción |
|---|---|---|---|---|
| BoW | 261.5 | 242.09 | 2.52 | **96.2×** |
| TF-IDF | 9.91 | 4.76 | 2.52 | **1.9×** |

En BoW cada documento aporta ~261 conteos y la evidencia pesa 96 veces más que el prior, así
que el prior es irrelevante. En TF-IDF los pesos están normalizados a norma L2 = 1, cada
documento suma 9.9 y la evidencia apenas dobla al prior: con esa proporción, P(Macroeconomia)
= 0.281 contra P(Reputacion) = 0.023 basta para inclinar la decisión salvo que el texto sea
inequívoco. **TF-IDF pondera mejor las palabras pero destruye la magnitud de los conteos, que
es exactamente lo que `MultinomialNB` necesita.** La mejor representación del Lab #2 (similitud
coseno) es la peor para este clasificador.

**H-09. Sobreajuste moderado en BoW; TF-IDF no está sobreajustado, está mal planteado.**
Brecha entrenamiento → validación: BoW 0.941 → 0.800 (**14.1 pp**), TF-IDF 0.710 → 0.547
(16.3 pp). En BoW la brecha se explica por 10,968 features contra 793 documentos, pero el
desempeño en validación se sostiene. En TF-IDF el modelo falla **también en entrenamiento**
(0.710), y un modelo que no aprende ni los datos que vio no tiene un problema de
generalización.

**H-10. Comportamiento por categoría en BoW (validación).**
Innovacion: recall 1.00 con precision 0.66 — recibe 35 predicciones para 23 documentos reales,
funciona como imán. Otra: precision 1.00 con recall 0.58 — cuando el modelo la predice acierta,
pero se le escapan 8 de 19. Reputacion: 1 acierto de 4, F1 0.40, sin valor estadístico.
**La confusión mutua más frecuente es Alianzas ↔ Regulaciones (4 + 3 = 7 errores cruzados)**,
que es el par a analizar en la Sección 4.

---

## Sección 4 — Análisis de errores

