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

---

## Sección 2 — Entrenamiento y evaluación

**H-05. La regresión logística con BoW memoriza el conjunto de entrenamiento por completo.**
Accuracy de entrenamiento **1.0000**: no se aproxima a los datos, los reproduce sin un solo error.
Con 10,968 features para 793 documentos hay muchas más dimensiones que ejemplos y las siete
categorías resultan linealmente separables. Aun así generaliza mejor que nada visto antes en este
corpus.

| Representación | Conjunto | Accuracy | F1 macro | F1 ponderado |
|---|---|---|---|---|
| **BoW** | entrenamiento | **1.0000** | 1.0000 | 1.0000 |
| **BoW** | **validación** | **0.8647** | **0.8286** | **0.8633** |
| TF-IDF | entrenamiento | 0.9508 | 0.9043 | 0.9486 |
| TF-IDF | validación | 0.8000 | 0.7263 | 0.7893 |

Brechas: BoW **13.5 pp**, TF-IDF **15.1 pp**, muy parecidas a las de Naive Bayes en el Lab #3
(14.1 y 16.3). La brecha mide cuánto se memorizó, no cuánto se fracasa: 0.8647 en validación supera
al mejor modelo del Lab #3 (0.8353, Naive Bayes con `max_features=1000`).

**H-06. La predicción de que TF-IDF le ganaría a BoW con regresión logística era falsa.**
La hipótesis de partida era que, al no depender la regresión logística de la magnitud de los
conteos, TF-IDF invertiría el resultado del Lab #3. **No ocurre: BoW gana igual**, 0.8647 contra
0.8000. Lo que sí cambia radicalmente es la distancia entre ambas representaciones, que cae de
**25.3 pp** (Naive Bayes) a **6.5 pp**.

**H-07. Lo que la regresión logística arregla es el colapso de TF-IDF, no su desventaja.**
En el Lab #3, `MultinomialNB` con TF-IDF se derrumbaba a 0.5471 prediciendo Macroeconomia 109 veces
de 170 y Reputacion ninguna. Con la misma matriz, la regresión logística sube a 0.8000 (**+25.3 pp**)
y su reparto de predicciones se normaliza:

| Categoría | Reales | NB TF-IDF (Lab #3) | LR TF-IDF | LR BoW |
|---|---|---|---|---|
| Macroeconomia | 48 | **109** | 49 | 47 |
| Alianzas | 36 | 41 | 47 | 38 |
| Innovacion | 23 | 14 | 30 | 26 |
| Regulaciones | 21 | 2 | 16 | 21 |
| Sostenibilidad | 19 | 3 | 19 | 23 |
| Otra | 19 | 1 | 8 | 13 |
| Reputacion | 4 | **0** | 1 | 2 |

La causa es estructural: Naive Bayes **suma** un prior estimado por separado a una evidencia cuya
magnitud depende de la representación, así que cuando TF-IDF achica la evidencia el prior decide.
La regresión logística aprende el sesgo `b` y los pesos `w` **conjuntamente**, y el sesgo no puede
dominar por un accidente de escala.

**H-08. De la desventaja de TF-IDF, dos tercios los causa normalizar y un tercio el IDF.**
TF-IDF hace dos cosas a la vez. Evaluando la representación intermedia (BoW normalizado a L2 = 1
pero sin IDF) se separan:

| Representación | acc train | acc val | F1 macro val | iteraciones |
|---|---|---|---|---|
| BoW crudo | 1.0000 | **0.8647** | **0.8286** | 55 |
| BoW normalizado L2 | 0.9395 | 0.8235 | 0.7566 | 35 |
| TF-IDF (normaliza + IDF) | 0.9508 | 0.8000 | 0.7263 | 33 |

**Normalizar cuesta 4.1 pp y el IDF otros 2.4 pp.** Igual que en el Lab #3, el daño mayor lo hace
destruir la magnitud de los conteos, aunque aquí por un mecanismo distinto y mucho más leve.

**H-09. El mecanismo es escala frente a regularización, no pérdida de información.**
En BoW la magnitud de un documento no está acotada: en validación va de 27 a 1,011 conteos,
**37×** entre el mayor y el menor. En TF-IDF todos valen exactamente 1. Al achicar las features se
achican los puntajes (|z| máximo medio de **7.77 a 1.89**), pero la penalización `l2` sigue con la
misma fuerza porque `C = 1.0` en ambos: **TF-IDF no está peor representado, está más regularizado.**
Comprobado en la Sección 5 — con `C = 1000` TF-IDF alcanza exactamente el 0.8647 de BoW.

**H-10. BoW acierta más pero está sobreconfiado; TF-IDF calibra mucho mejor.**

| | \|z\| máx medio | Confianza media | Docs con confianza > 0.99 | corr(largo, \|z\|) |
|---|---|---|---|---|
| BoW | 7.77 | 0.9021 | **77 / 170** | **+0.655** |
| TF-IDF | 1.89 | 0.5078 | **0 / 170** | +0.079 |

El tamaño de los puntajes de BoW correlaciona con **el largo del documento** (+0.66): un texto largo
acumula más conteos, `z` crece y la softmax satura, de modo que la "seguridad" del modelo mide en
parte cuántas palabras tenía la noticia. Relevante para dos cosas: el análisis de errores de la
Sección 6 (equivocarse con 99 % de certeza es el mismo problema que se le documentó a Naive Bayes)
y la pregunta de la Sección 7 sobre probabilidades bien calibradas.

**D-06. El conjunto de prueba no se toca en esta sección.**
El enunciado pide evaluar "sobre validación y, al final, sobre prueba". Quedan decisiones por tomar
sobre validación —la elección de `C` en la Sección 5—, y medir en prueba antes convertiría ese
conjunto en un segundo conjunto de validación. La evaluación final se hace **una sola vez**, en la
Sección 4, junto a la comparación con Naive Bayes. Mismo criterio que el Lab #3, que dejó su
evaluación en prueba para la Sección 7.

**D-07. Figura de la sección:** `img/confusion_lr_validacion.png`, dos paneles (BoW y TF-IDF) con
`sns.heatmap`, mismo estilo del Lab #3.

---

## Sección 3 — Interpretación de pesos

**D-08. Modelo y categorías analizadas.** Se interpretan los pesos del **mejor modelo de la
Sección 2**, regresión logística con BoW (0.8647 en validación). Categorías: **Macroeconomia** (la
mejor clasificada) y el par **Alianzas / Regulaciones**, que en el Lab #3 fue el que Naive Bayes más
confundía (7 errores cruzados de 34), para enlazar con el análisis de errores de la Sección 6.

**H-11. Los pesos negativos son el vocabulario de las otras categorías.**
En Macroeconomia los positivos son el léxico del tema (`inflación` +0.47, `aumento`, `economía`,
`precios`, `ipc`, `alza`, `crecimiento`) y los negativos son `personas`, `digital`, `usuarios`,
`tecnología`, `alianza`, `regulación`: el modelo aprendió también a qué **no** se parece una noticia
macroeconómica.

**H-12. En el par confundido, cada categoría empuja hacia abajo la palabra insignia de su rival.**

| Término | Peso en Alianzas | Peso en Regulaciones |
|---|---|---|
| `alianza` | **+1.01** | −0.29 |
| `regulación` | −0.36 | **+0.73** |
| `bbva` | **−0.82** | −0.40 |

El peso de `regulación` en Alianzas no sale de contar apariciones, sino de que bajarlo reduce la
confusión entre las dos categorías. Es la firma de la optimización conjunta.

**H-13. El solapamiento entre Alianzas y Regulaciones pasa de 7/15 a 0/15.**
Es el resultado central de la sección. En el Lab #3 se documentó que ambas compartían buena parte de
su vocabulario más probable, y que por eso el modelo las confundía cuando la noticia no decía
literalmente "alianza" o "regulación".

| Método | Palabras compartidas en el top-15 |
|---|---|
| Naive Bayes, mayor `P(w\|c)` | **7 de 15** — `colombia`, `empresas`, `hace`, `mercado`, `nueva`, `país`, `servicios` |
| Reg. logística, mayor peso | **0 de 15** |

**H-14. Naive Bayes premia palabras frecuentes; la regresión logística, palabras discriminativas.**
Cuantificado sobre las 7 × 15 = 105 posiciones del top de cada categoría:

| | Términos distintos | Exclusivos de una sola categoría |
|---|---|---|
| Naive Bayes, mayor `P(w\|c)` | 64 | 41 (**64 %**) |
| Reg. logística, mayor peso | 100 | 97 (**97 %**) |

Naive Bayes repite: `empresas` está en el top-15 de **cinco** categorías; `país`, `nueva` y `bbva`
en cuatro. Una palabra que caracteriza a cinco categorías no distingue ninguna.

**H-15. La coincidencia entre ambos métodos depende de si la categoría tiene vocabulario propio.**

| Categoría | En común (top-15) |
|---|---|
| Macroeconomia | 7 de 15 |
| Regulaciones | 4 de 15 |
| Alianzas | 3 de 15 |

Encaja con el Lab #2: Alianzas era la categoría **menos cohesionada** del corpus (razón intra/inter
1.12×), porque "alianza" es un tipo de evento y no un tema. Donde no hay léxico propio, los dos
métodos divergen.

**H-16. `bbva` es el ejemplo completo de la diferencia entre contar y optimizar.**
El corpus son noticias publicadas por BBVA, así que la palabra aparece en **338 de 793 documentos
(43 %)**.

| Categoría | Peso (reg. logística) | Puesto en `P(w\|c)` (NB) |
|---|---|---|
| Innovacion | **+0.631** | **1** de 10,968 |
| Sostenibilidad | +0.278 | 2 |
| Otra | +0.196 | 2 |
| Macroeconomia | +0.266 | 7 |
| Reputacion | −0.151 | 7,820 |
| Regulaciones | −0.396 | 3,495 |
| Alianzas | **−0.824** | 2,531 |

Para Naive Bayes es **la palabra más característica de Innovacion** y la segunda de otras dos:
información inútil, porque caracteriza a casi todas. La regresión logística le da el peso **más
negativo de toda la categoría Alianzas**, aprendiendo que *"si menciona a BBVA, probablemente no es
una alianza"* —las alianzas se anuncian nombrando al socio—. Es una inferencia que `P(wᵢ|c)` no
puede expresar en ninguna forma, porque todos sus valores son log-probabilidades y una palabra solo
puede aportar evidencia a favor.

**H-17. Hay pesos altos que son pistas de la fuente, no del tema.**
`research` (+0.20 en Macroeconomia) viene de "BBVA Research", el servicio de estudios que publica
los informes de inflación; `podcast` aparece en el top positivo de dos categorías por el formato de
la publicación. Son regularidades del corpus que no generalizarían a noticias de otro medio, y son
coherentes con la accuracy de 1.0000 en entrenamiento (H-05).

**D-09. Figura de la sección:** `img/pesos_par_confundido.png`, barras horizontales con los 10 pesos
más positivos (azul) y los 10 más negativos (rojo) de Alianzas y Regulaciones.
