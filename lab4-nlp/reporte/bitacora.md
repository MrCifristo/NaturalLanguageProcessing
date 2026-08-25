# Bitácora de trabajo — Laboratorio #4

Registro de hallazgos, decisiones y problemas encontrados durante el desarrollo.
Sirve como material de respaldo para el reporte escrito: cada afirmación del reporte
debería poder rastrearse hasta una entrada de aquí.

**Corpus:** Spanish News Classification (`df_total.csv`) · **Notebook:** `lab4.ipynb`

---

## Entorno y estructura

**D-01. El notebook vuelve a ser autocontenido.**
Misma decisión que en los labs #2 y #3: `lab4.ipynb` reproduce en su Sección 0 la carga, el
pipeline de normalización, las particiones y los vectorizadores de los laboratorios anteriores.
Costo asumido: una cuarta copia de `df_total.csv`.

**D-02. Sin dependencias nuevas.**
Todo lo que pide el laboratorio (`LogisticRegression`, `DummyClassifier`) ya viene en
scikit-learn 1.7.2, que estaba instalado desde el Lab #2. `requirements.txt` no se toca.

**D-03. Numeración de las secciones: el enunciado tiene un hueco, otra vez.**
El PDF numera las secciones 1–6, con la 6 titulada "Análisis". Pero hay dos señales de que falta
una sección intermedia: la tabla de entregables pide explícitamente "**efecto de la
regularización**" —que no aparece como sección propia— y remite al análisis como "**sección 7**";
y la pregunta 2 de ese análisis habla del "análisis **de errores (sección 6)**" cuando el PDF lo
numera como 5. Hay un corrimiento de uno. Se adopta esta numeración:

| Sección | Contenido |
|---------|-----------|
| 1 | Construcción del clasificador de regresión logística |
| 2 | Entrenamiento y evaluación |
| 3 | Interpretación de pesos |
| 4 | Comparación con Naive Bayes |
| **5** | **Efecto de la regularización** (sección ausente en el PDF) |
| 6 | Análisis de errores |
| 7 | Análisis |

Es el mismo criterio aplicado en el Lab #3 (su D-03), donde faltaba la sección de implementación
propia.

---

## Sección 0 — Preparación del corpus

**D-04. Las particiones y los vectorizadores se reconstruyen, no se importan — y se verifica.**
El enunciado exige reutilizar "exactamente las mismas particiones y los mismos vectorizadores
ajustados en el Laboratorio #3". Como el notebook es autocontenido, no se importan desde
`lab3.ipynb`: se vuelven a construir. Eso solo es equivalente si el resultado es idéntico, cosa
que se sostiene porque todo el pipeline es determinista (`RANDOM_STATE = 42`, `CountVectorizer()`
y `TfidfVectorizer()` sin parámetros, mismo corpus de partida).

"Debería ser idéntico" no es una comprobación, así que se añadieron cinco `assert` que fallan
ruidosamente si algo se desvía:

| Celda | Comprobación | Valor esperado |
|---|---|---|
| 0.2 | documentos tras la limpieza | 1,134 |
| 0.3 | tokens y tipos del pipeline | 298,753 / 13,093 |
| 0.4 | tamaño de las particiones | 793 / 170 / 171 |
| 0.5 | vocabulario ajustado con entrenamiento | 10,968 términos |
| 0.6 | accuracy de Naive Bayes en validación | 0.8000 (BoW) / 0.5471 (TF-IDF) |

Los cinco pasan. El último es el más fuerte de todos: si Naive Bayes reproduce exactamente los
números del Lab #3, entonces las particiones, el vocabulario y las matrices son los mismos, y la
comparación del laboratorio es legítima. Sin esto, cualquier diferencia de desempeño entre Naive
Bayes y regresión logística podría venir de las particiones y no del algoritmo.

---

## Sección 1 — Construcción del clasificador

**H-01. Los dos modelos convergen con el `max_iter` por defecto, pero no cuesta lo mismo.**
La expectativa inicial era que BoW no convergiera en las 100 iteraciones por defecto de `lbfgs`,
por trabajar con conteos crudos sin acotar. **No ocurre:** no hay `ConvergenceWarning` en ninguno
de los dos modelos y `max_iter` se deja en su valor por defecto.

| Modelo | `n_iter_` | Límite |
|---|---|---|
| BoW | 55 | 100 |
| TF-IDF | 33 | 100 |

La diferencia sí existe y va en la dirección esperada: BoW necesita **1.7× más iteraciones**. La
razón es la escala. Un documento de BoW aporta conteos enteros sin cota (el documento 0 de
entrenamiento suma 93 ocurrencias sobre 74 términos distintos), mientras que TF-IDF normaliza cada
documento a **norma L2 = 1**, dejando una superficie mucho mejor condicionada para el optimizador.
Es el primer indicio de algo que la Sección 2 debería confirmar en el desempeño.

**D-05. No se pasa `multi_class`.**
La forma aparentemente obvia de manejar 7 categorías sería `LogisticRegression(multi_class=
"multinomial")`. Es un error en este entorno: el parámetro está **deprecado desde scikit-learn 1.5
y se elimina en la 1.8**; en la 1.7.2 pasarlo emite un `FutureWarning`. Con `multi_class='auto'`,
`n_classes >= 3` y solver `lbfgs`, el modelo **ya usa softmax multinomial**. El "ajuste necesario"
que pide el enunciado no es un parámetro: es que el modelo pasa de un vector de pesos a siete
(`coef_.shape == (7, 10968)`) y de la sigmoide a softmax.

**H-02. Que el modelo es multinomial y no one-vs-rest se comprueba, no se supone.**
Mirar `coef_.shape` no distingue los dos casos: one-vs-rest también daría (7, 10968). La prueba
está en rehacer el cálculo a mano sobre un documento (celda 1.3):

- softmax de los siete `z` **== `predict_proba`** → `True`
- sigmoide de cada `z` normalizada (lo que daría OvR) == `predict_proba` → `False`

En el documento 0 de validación (Sostenibilidad, `z = 7.251`) la diferencia es enorme: softmax da
**0.9953** y la sigmoide normalizada **0.3185**. Sirve de paso como verificación de que se entendió
la cadena `z → probabilidad → clase` que pide explicar el enunciado.

**H-03. Los pesos negativos son la diferencia estructural con Naive Bayes.**
Las dos matrices tienen la misma forma (7, 10968) y ambos son clasificadores lineales, pero lo que
guarda cada celda es distinto:

| | Rango | Signo |
|---|---|---|
| `nb_bow.feature_log_prob_` | −11.185 a −4.240 | **todos negativos** (son log-probabilidades) |
| `lr_bow.coef_` | −0.824 a 1.013 | 31,002 positivos, 45,774 negativos, 0 exactos |

En Naive Bayes todos los valores son negativos porque son logaritmos de probabilidades, y cada fila
exponenciada suma 1: una palabra solo puede aportar evidencia *a favor*, con más o menos fuerza.
En regresión logística **un peso negativo es evidencia en contra** de la categoría, algo que la
estimación aislada de `P(wᵢ|c)` no tiene forma de expresar. Y son mayoría: el 60 % de los 76,776
pesos son negativos. Material para la Sección 3.

**H-04. Ningún peso es exactamente cero.** Con la penalización `l2` por defecto los pesos se
encogen pero no se anulan: los 76,776 son distintos de cero. Contrastar en la Sección 5 contra
`penalty="l1"`, que sí produce esparsidad.
