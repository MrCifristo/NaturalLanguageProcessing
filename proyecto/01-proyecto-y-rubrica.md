# Radar de Licitaciones — Proyecto y rúbrica
> Documento de contexto · ALBRAN Industries · Proyecto final PLN, UFM
> Estado: Propuesta presentada · Actualizado: octubre 2026

## 1. Contexto

El Radar de Licitaciones revisa a diario lo publicado en Guatecompras y entrega a una PyME de software solo los concursos que le interesan, ordenados por relevancia y con una explicación de por qué se recomiendan.

Hoy alguien revisa el portal a mano, o nadie lo revisa y se pierden oportunidades. Guatecompras publica del orden de 260 mil procesos al año (1.77 millones entre enero 2020 y septiembre 2026), aproximadamente 1,000 por día hábil, y solo una fracción mínima es relevante para una empresa de software.

Un filtro tradicional no basta. Tres casos reales:

| Concurso | Realidad | Filtro | Resultado |
|---|---|---|---|
| «Adquisición de software para la Policía Municipal de Tránsito de Poptún» | Es software | Por categoría | Lo pierde: está clasificado como consultoría |
| «Servicio de 3,000 envíos y recepción de sobres…» | No es software | Por categoría | Lo deja pasar: está clasificado en tecnología |
| «Tomografía computarizada… que incluya software para visualización» | No es software | Por palabra clave | Lo marca: el texto menciona «software» |

Hay que interpretar el texto, por eso PLN es la herramienta adecuada. Como referencia de mercado, existe un servicio comercial similar que cobra $99/mes.

Estado real: la propuesta se presentó como idea. No hay implementación verificada en este repositorio; existe una sesión previa en Claude Code en otro equipo cuyo alcance hay que inventariar.

## 2. A quién va dirigido

- **Cliente / dueño:** Milton Beltrán, como proyecto académico. Primer usuario real: ALBRAN Industries.
- **Usuarios finales:** PyMEs de desarrollo de software en Guatemala. La persona que decide a qué concursos aplicar abre cada mañana una lista corta, decide y dedica su tiempo a preparar propuestas.
- **Otros stakeholders:** el catedrático del curso de PLN (evaluación). El Ministerio de Finanzas Públicas como publicador de los datos (licencia CC BY 4.0, exige citar la fuente).

## 3. Alcance

**Cubre:**

- Captura automatizada diaria desde la API OCDS de Guatecompras.
- Limpieza y preparación documentada del texto de los procesos.
- Clasificación de relevancia (TF-IDF + regresión logística) y similitud semántica contra el portafolio de la empresa (embeddings).
- Ranking diario con explicación por concurso.
- Panel web y resumen diario por Slack o correo.
- Evaluación con métricas adecuadas a clases desbalanceadas y comparación contra líneas base.

**No cubre (MVP):**

- Lectura de los PDF de bases de cada concurso.
- Publicaciones del módulo NPG de Guatecompras, que no están en el OCDS.
- Reentrenamiento automático. La retroalimentación se guarda; reentrenar queda como trabajo futuro.
- Preparación de ofertas o seguimiento posterior a la decisión de aplicar.
- Multiempresa con cuentas de usuario. El diseño lo permite (el perfil de empresa es un CSV), pero no se implementa.

## 4. Funcionalidades

| Funcionalidad | Para quién | Estado |
|---|---|---|
| Descarga diaria del archivo del mes desde la API OCDS | Sistema | Planificado |
| Detección de procesos nuevos y abiertos | Sistema | Planificado |
| Clasificación de relevancia con probabilidad | Usuario final | Planificado |
| Similitud con proyectos anteriores de la empresa | Usuario final | Planificado |
| Ranking diario con explicación (palabras con más peso + proyecto más parecido) | Usuario final | Planificado |
| Panel web con filtros por modalidad, monto y fecha de cierre | Usuario final | Planificado |
| Resumen diario de los 10 mejores por Slack o correo | Usuario final | Planificado |
| Botones «Me interesa» / «No aplica» guardados como etiquetas | Usuario final | Planificado |
| Herramienta de etiquetado con pre-etiqueta de LLM | Milton (preparación) | Planificado |

## 5. Stack técnico

Ver `CLAUDE.md` (sección Stack) y `docs/03-arquitectura-y-pipeline.md`. Resumen: Python 3.12, DuckDB, scikit-learn, sentence-transformers, spaCy, Streamlit, API de Anthropic para pre-etiquetado.

## 6. Detalles operativos

| Elemento | Valor |
|---|---|
| Fuente de datos | `https://ocds.guatecompras.gt/files` |
| Licencia de datos | CC BY 4.0, Ministerio de Finanzas Públicas |
| Horario de captura | 05:00 hora de Guatemala (archivos regenerados de madrugada) |
| Horario del resumen | 07:00 hora de Guatemala |
| Host de ejecución | [PENDIENTE: confirmar. Candidato: servidor homelab siempre encendido] |
| Presentación de la propuesta | Deck "Radar de Licitaciones" en claude.ai (15 diapositivas) |

### Rúbrica del curso

Objetivo del proyecto final: producto end-to-end basado en PLN que resuelva un problema real y agregue valor a un usuario final concreto. Ciclo obligatorio:

1. **Captura de información:** datos reales (scraping, APIs, datasets abiertos, documentos, etc.).
2. **Limpieza y preparación:** tokenización, normalización, manejo de ruido, faltantes o duplicados, con decisiones documentadas.
3. **Extracción de información / modelado:** al menos una técnica de PLN vista en el curso.
4. **Producto y presentación:** algo usable (demo, app simple, dashboard, bot, API o notebook interactivo), no solo un reporte de métricas.

Pregunta guía: ¿a quién le sirve esto y qué haría esa persona con el resultado?

Criterios de evaluación de la propuesta:

| Criterio | Qué se busca |
|---|---|
| Claridad de la idea | Producto y usuario final claros en una o dos frases |
| Justificación | Valor real para alguien fuera del equipo, no solo interés académico |
| Viabilidad de los datos | Los datos existen, son accesibles y suficientes |
| Fundamento teórico | Técnica de PLN adecuada y conectada con un tema del syllabus |
| Enfoque end-to-end | Flujo completo claro: captura, limpieza, modelado y producto |
| Comunicación | Presentación clara, ordenada y en el tiempo asignado |

Temas del syllabus que sustentan el proyecto: clasificación de texto (regresión logística), TF-IDF y similitud de documentos, semántica distribucional y embeddings, modelos preentrenados.

Entregas:

| Entrega | Fecha | Estado |
|---|---|---|
| Propuesta (idea, justificación, fundamento teórico; 4–6 diapositivas sugeridas) | Martes 22 de septiembre de 2026 | Presentada |
| Entrega final | [PENDIENTE: fecha y formato] | Pendiente |
| Entregas intermedias | [PENDIENTE: confirmar si existen] | — |

## 7. Personalización / diseño

Identidad visual usada en la presentación, para mantener consistencia en el panel de Streamlit:

| Elemento | Valor |
|---|---|
| Fondo oscuro | `#14171B` |
| Superficie oscura | `#1E2329` |
| Fondo claro | `#F3F2EE` |
| Acento (sobre oscuro) | `#9A8CFF` |
| Acento (sobre claro) | `#5543D1` |
| Texto secundario (sobre oscuro) | `#B4BBC4` |
| Tipografía de texto | IBM Plex Sans |
| Tipografía monoespaciada | JetBrains Mono |

## 8. Estado y pendientes

- [PENDIENTE: comentarios del catedrático sobre la propuesta, si los hubo]
- [PENDIENTE: fecha y formato de la entrega final]
- [PENDIENTE: inventario de lo hecho en la sesión anterior de Claude Code]
- Datos de la presentación no verificados de forma independiente: el ~1% de relevancia, la ventana de 10 días de las compras directas y la referencia de $99/mes. El ~1% se medirá con el etiquetado.
