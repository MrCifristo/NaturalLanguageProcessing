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

**D-11. El par más confundido se identifica sumando las dos direcciones de la matriz, no
buscando la celda más alta.** La pregunta del enunciado es por confusión *mutua*: se calcula
`cm[a,b] + cm[b,a]` para los 21 pares posibles. Alianzas ↔ Regulaciones sale con 7 (4+3), muy
por encima del siguiente (Innovacion ↔ Macroeconomia, 4).

**D-12. Para atribuir culpa a palabras concretas se usa la diferencia de log-verosimilitudes.**
`culpa(w) = conteo(w) · [log P(w|predicha) − log P(w|real)]`. Naive Bayes decide comparando
scores, así que lo que importa no es qué palabras son frecuentes en el documento sino cuáles
aportan más a la categoría equivocada *por encima* de lo que aportan a la correcta. Ordenar por
esa cantidad da directamente las palabras que inclinaron la balanza.

**H-11. Los 7 errores cruzados se explican por tres causas distintas.**
1. *El tema arrastra*: doc 667 (aniversario de Uber Taxi, etiquetado Alianzas) se va a
   Regulaciones por `uber` ×7, `taxistas`, `taxi`. En entrenamiento Uber aparece casi siempre en
   el conflicto legal con los taxis. La palabra `alianza`, presente una sola vez, no compensa.
   Igual el doc 72 (`taxistas`, `multas`, `ilegal`).
2. *La etiqueta original es discutible*: docs 589 y 296 son columnas de opinión sobre elecciones
   y campañas, etiquetadas Alianzas. El modelo predice Regulaciones por `elecciones`, `pueblo`,
   `decreto`, `subsidios`, `regulación`. Es defendible que el modelo tenga más razón que la
   anotación.
3. *Gana la palabra literal*: doc 756 (convenio MinCiencias–CRC, etiquetado Regulaciones) se va
   a Alianzas por `alianza` y `convenio`; doc 797 (regionalización de agua potable) se va por
   `agua`, `potable`, `saneamiento`.

**H-12. El modelo se equivoca con 100 % de confianza en 5 de los 7 errores cruzados.**
`predict_proba` devuelve 1.00 para la clase incorrecta en los docs 589, 667, 756, 296 y 72. No
es un empate mal resuelto: el modelo está seguro. Es la firma del supuesto de independencia —
palabras correlacionadas (`uber`, `taxi`, `taxistas`, `tarifas`) se suman como si fueran
evidencia independiente, y el score se dispara.

**H-13. Alianzas y Regulaciones comparten 9 de sus 20 palabras más probables.**
Las compartidas son `colombia`, `empresas`, `hace`, `mercado`, `nueva`, `pago`, `país`, `puede`
y `servicios`. Cada categoría tiene una sola palabra realmente propia en el tope (`alianza` con
P = 0.0073 y `regulación` con P = 0.0073); el resto de su vocabulario característico es
genérico. Explica por qué son el par más confundido: si la noticia no dice literalmente
"alianza" o "regulación", la decisión queda en manos de palabras que no distinguen nada.
En contraste, Macroeconomia (`inflación`, `precios`, `tasa`, `crecimiento`) y Sostenibilidad
(`energía`, `sostenible`, `electricidad`, `agua`) tienen vocabularios propios y son las que
mejor clasifica.

---

## Sección 5 — Efecto de la representación de texto

**H-14. Recortar el vocabulario a 1000 términos MEJORA el modelo.**

| Representación | Vocabulario | Acc. train | Acc. val | F1 macro val | Vectorizar | Entrenar |
|---|---|---|---|---|---|---|
| BoW completo | 10,968 | 0.941 | 0.800 | 0.732 | 0.08 s | 1.9 ms |
| **BoW max_features=1000** | **1,000** | **0.888** | **0.835** | **0.837** | 0.08 s | 1.3 ms |
| BoW max_features=5000 | 5,000 | 0.937 | 0.812 | 0.786 | 0.08 s | 1.6 ms |
| BoW uni+bigramas | 156,417 | 0.989 | 0.759 | 0.680 | 0.28 s | 9.6 ms |
| BoW uni+bi max_features=5000 | 5,000 | 0.940 | 0.824 | 0.798 | 0.29 s | 1.7 ms |
| TF-IDF completo | 10,968 | 0.710 | 0.547 | 0.360 | 0.08 s | 1.6 ms |

El barrido completo (200 → 10,968 términos) muestra la curva clásica de sobreajuste: la
accuracy de entrenamiento crece de forma monótona (0.830 → 0.941) mientras la de validación
sube hasta 1000 términos (0.835) y luego cae. La brecha entrenamiento–validación baja de
**14.1 pp a 5.3 pp** al recortar. Las palabras más allá de las primeras mil son demasiado raras
para generalizar y solo sirven para memorizar.

**H-15. El recorte beneficia sobre todo a las categorías pequeñas.**

| Categoría | Docs en val | F1 completo | F1 con 1000 | Δ |
|---|---|---|---|---|
| Reputacion | 4 | 0.400 | 0.889 | **+0.489** |
| Innovacion | 23 | 0.793 | 0.885 | +0.092 |
| Otra | 19 | 0.733 | 0.800 | +0.067 |
| Sostenibilidad | 19 | 0.810 | 0.857 | +0.048 |
| Regulaciones | 21 | 0.718 | 0.762 | +0.044 |
| Alianzas | 36 | 0.789 | 0.806 | +0.017 |
| Macroeconomia | 48 | 0.884 | 0.860 | −0.024 |

Reputacion entrena con 18 documentos: sus estimaciones sobre palabras raras eran ruido, y
quitarlas la deja decidir con vocabulario frecuente bien medido. Es la razón de que el F1 macro
suba 10 puntos mientras la accuracy sube solo 3.5: la accuracy la dominan las clases grandes.

**H-16. Los bigramas crudos cuestan 14× más vocabulario y empeoran el desempeño; podados,
ayudan un poco.** `ngram_range=(1,2)` sin poda: 156,417 términos, 3.5× más tiempo de
vectorización, 5× más de entrenamiento, accuracy 0.759 (peor que unigramas). Concuerda con el
Lab #2, donde se midió que **el 80 % de los bigramas aparece una sola vez**: no generalizan,
solo permiten memorizar (accuracy de entrenamiento 0.989). Con `max_features=5000`, en cambio,
uni+bi (0.824 / 0.798) supera a los 5000 unigramas solos (0.812 / 0.786): al podar sobreviven
los bigramas con significado (`banc central`, `cambi climat`). Aun así ninguna configuración con
bigramas le gana a las 1000 palabras sueltas.

**D-13. El mejor modelo del laboratorio es BoW con `max_features=1000`**, elegido sobre
validación. Es el que se usa como referencia en la evaluación final sobre prueba.

---

## Sección 6 — Implementación propia de Naive Bayes

**H-17. La implementación propia reproduce `MultinomialNB` exactamente.**
Diferencia máxima en log P(w|c): **0.00**. En log P(c): 6.7e-16 (ruido de punto flotante).
Predicciones idénticas en el **100 %** de los documentos de validación, con y sin recorte de
vocabulario. Accuracy 0.8000 y F1 macro 0.7324 en ambas, iguales a la Sección 3.

**D-14. El punto delicado del suavizado es el denominador, no el numerador.**
Sumar α a cada conteo es lo evidente; lo que se olvida es que si se suma α a las 10,968 palabras
del vocabulario, el total de la categoría crece en α·|V|, no en α. Dividir entre el total
original deja las "probabilidades" sin sumar 1. En el código se resuelve sumando la fila
**después** de sumar α (`numerador.sum(axis=1)`), que es la misma cuenta pero imposible de
equivocar.

**H-18. Sin logaritmos el algoritmo no funciona en absoluto: no da un número pequeño, da cero.**
El producto directo de las verosimilitudes de un documento de validación con 479 palabras
(318 términos distintos) es **exactamente 0.0**. Las siete categorías empatarían en cero. Los
mismos scores en logaritmos son del orden de −4000 y se comparan sin problema. Es la
justificación práctica —no teórica— de por qué la fórmula se implementa siempre en logaritmos.

**H-19. Entrenar Naive Bayes es contar, no optimizar.**
No hay iteraciones ni convergencia: agrupar por categoría, sumar conteos, tomar logaritmos. El
modelo entero son dos tablas (7 valores y 7 × 10,968) y predecir es una multiplicación de
matrices. Entrena en **0.7–1.9 ms**, mientras que vectorizar el corpus con bigramas tarda 290 ms:
el costo del laboratorio está en la representación, no en el clasificador. La implementación
propia sale ligeramente más rápida que `MultinomialNB` solo porque no valida entradas ni
soporta `sample_weight`, clases vacías o `partial_fit`.

**H-20. Tabla comparativa final (validación).**

| Modelo | Vocabulario | Accuracy | F1 macro | Entrenar |
|---|---|---|---|---|
| **BoW max_features=1000** | 1,000 | **0.8353** | **0.8370** | 1.3 ms |
| BoW max_features=1000 (propia) | 1,000 | 0.8353 | 0.8370 | 0.7 ms |
| BoW uni+bi, max_features=5000 | 5,000 | 0.8235 | 0.7981 | 1.8 ms |
| BoW max_features=5000 | 5,000 | 0.8118 | 0.7855 | 1.6 ms |
| BoW completo | 10,968 | 0.8000 | 0.7324 | 1.9 ms |
| BoW completo (propia) | 10,968 | 0.8000 | 0.7324 | 1.2 ms |
| BoW unigramas + bigramas | 156,417 | 0.7588 | 0.6796 | 9.7 ms |
| TF-IDF completo | 10,968 | 0.5471 | 0.3602 | 1.9 ms |

---

## Sección 7 — Evaluación final y análisis

