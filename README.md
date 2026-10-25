# Sofia — Asistente Clínico Inteligente y RAG de Psicofarmacología

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sofiaasistenteragdepsicofarmacolog.streamlit.app)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![RAG Smoke Test](https://img.shields.io/badge/RAG%20Smoke%20Test-N%3D4%20(preliminar)-blue.svg)](data/processed/rag_evaluation_report.json)

> 🚀 **Aplicación Web Desplegada en Streamlit Cloud:**  
> Accede a la interfaz interactiva en: **[https://sofiaasistenteragdepsicofarmacolog.streamlit.app](https://sofiaasistenteragdepsicofarmacolog.streamlit.app)**

---

## 📌 ¿Qué es Sofia y cuál es el alcance real de su RAG?

**Sofia** es un sistema de soporte a la decisión clínica, farmacovigilancia y generación aumentada por recuperación (RAG) especializado en **psicofarmacología basada en la evidencia**.

Para garantizar la seguridad médica y evitar alucinaciones, **Sofia opera bajo una política de *Grounding Estricto*:**
> **El modelo responde únicamente en base a los 26 documentos clínicos curados en [`pre_training/`](pre_training/) y a las tablas estructuradas de [`data/processed/`](data/processed/)** (la base indexada, [`rag_knowledge_base.json`](data/processed/rag_knowledge_base.json), contiene 57 documentos: los 26 curados + 22 fichas de recomendaciones CANMAT + 9 fichas de marcas comerciales).  
> Si una consulta excede este corpus, el sistema **declara explícitamente su limitación**: con recuperación léxica (TF-IDF) lo hace mediante un umbral mínimo de similitud (heurística, puede fallar en consultas ambiguas) y, con LLM, mediante el prompt de *grounding estricto*. Las citas `[DOC-ID]` de las respuestas del LLM se validan contra los documentos recuperados y se avisa si alguna no figura.

---

## 🎯 Capacidades Reales: Qué SÍ puede responder el RAG

El motor RAG cuenta con información detallada, validada y estructurada para responder con alto rigor científico a preguntas en las siguientes cinco áreas:

### 1. Monografías y Posología de 9 Psicofármacos Clave
Cada una de las siguientes moléculas dispone de una monografía completa (mecanismo neuroquímico, afinidad de receptores, rangos de dosificación según Jufe y Stahl, efectos comunes vs. graves y poblaciones especiales):

| Principio Activo | Clase Terapéutica | Código ATC | Rango Terapéutico Usual | Aspectos Específicos Indexados |
| :--- | :--- | :---: | :---: | :--- |
| **Sertralina** | Antidepresivo ISRS | `N06AB06` | 50 – 200 mg/día | Cardiopatía isquémica (estudio SADHART), marcas y laboratorios. |
| **Fluoxetina** | Antidepresivo ISRS | `N06AB03` | 20 – 60 mg/día | Vida media prolongada (norfluoxetina), antagonismo 5-HT2C, lavado IMAO. |
| **Paroxetina** | Antidepresivo ISRS | `N06AB05` | 20 – 50 mg/día | Perfil anticolinérgico M1, sedación, síndrome de discontinuación intenso. |
| **Bupropión** | Antidepresivo IRND | `N06AX12` | 150 – 300 mg/día | Cero impacto sexual/peso, metabolismo CYP2B6, umbral convulsivo. |
| **Haloperidol** | Antipsicótico Típico ($D_2$) | `N05AD01` | 2 – 15 mg/día | Síntomas extrapiramidales agudos, riesgo QTc, cascada con biperideno. |
| **Risperidona** | Antipsicótico Atípico | `N05AX08` | 2 – 6 mg/día | Metabolismo CYP2D6 hacia 9-OH-risperidona, hiperprolactinemia. |
| **Clozapina** | Antipsicótico Atípico | `N05AH02` | 300 – 600 mg/día | Dependencia de CYP1A2 (efecto tabaco), protocolo de agranulocitosis. |
| **Carbonato de Litio**| Estabilizador del Ánimo | `N05AN01` | Litemia: 0.6–1.0 mEq/L | Eliminación renal exclusiva, litemias tóxicas (>1.5 mEq/L), AINEs/tiazidas. |
| **Lamotrigina** | Estabilizador / Antiepiléptico | `N03AX09` | 100 – 200 mg/día | Glucuronidación UGT1A4, titulación lenta, prevención de Stevens-Johnson. |

*(Nota: Fármacos adicionales como Quetiapina, Olanzapina, Aripiprazol, Valproato, Escitalopram, Citalopram, Venlafaxina o Duloxetina están indexados en las tablas CANMAT y el catálogo ATC, pero no cuentan con una monografía de texto largo independiente).*

### 2. Interacciones Enzimáticas del Citocromo P450 y Aclaramiento Renal
El RAG explica el mecanismo molecular y la conducta clínica esperada ante:
- **Inhibición de CYP2D6:** Impacto de inhibidores potentes (Paroxetina, Bupropión, Fluoxetina) sobre sustratos de índice estrecho (Risperidona, Haloperidol).
- **Metabolismo de CYP1A2 y Clozapina:** Elevación tóxica de clozapina ante el cese del consumo de tabaco (los hidrocarburos inducen 1A2) o coadministración de inhibidores.
- **Glucuronidación (UGT1A4) de Lamotrigina:** Duplicación de niveles plasmáticos al asociarse con Ácido Valproico y necesidad de reducir la dosis inicial en $\ge 50\%$.
- **Toxicidad Lítica por Aclaramiento Renal:** Disminución del filtrado de Litio provocada por AINEs (ibuprofeno, naproxeno), diuréticos tiazídicos e IECA.

### 3. Detección de Cascadas de Prescripción y Desprescripción
Identificación del patrón donde el efecto adverso de un psicofármaco se confunde con una nueva enfermedad:
- **Haloperidol $\to$ Parkinsonismo $\to$ Biperideno:** Alerta sobre carga anticolinérgica, riesgo de discinesia tardía irreversible y deterioro cognitivo; recomienda ajuste de dosis o rotación a atípicos.
- **Clozapina $\to$ Sialorrea nocturna:** Protocolo con medidas locales (atropina sublingual) frente al uso erróneo de anticolinérgicos orales sistémicos.
- **Litio $\to$ Poliuria:** Alerta crítica de contraindicación de diuréticos tiazídicos para tratar la diabetes insípida nefrogénica inducida por litio.

### 4. Guías Internacionales CANMAT / ISBD 2018
Recomendaciones oficiales tabuladas por líneas de tratamiento y niveles de evidencia científica (Level 1 a Level 4):
- **Trastorno Bipolar (Manía Aguda):** Fármacos de primera línea (Litio, Quetiapina, Divalproato, combinaciones).
- **Trastorno Bipolar (Depresión Bipolar):** Primera línea (Quetiapina, Lurasidona) y **alerta de viraje maníaco** (contraindicación formal Level 1 de daño para ISRS en monoterapia).
- **Trastorno Depresivo Mayor (TDM):** Fases agudas y algoritmos de potenciación en resistencia (ej. adición de Aripiprazol o Litio).

### 5. Trazabilidad Comercial y Clasificación ATC (OMS)
- Mapeo directo a códigos anatómicos oficiales de la OMS.
- Nombres comerciales y laboratorios de referencia en España y Latinoamérica (ej. *Zoloft, Serlift, Prozac, Paxil, Wellbutrin, Leponex, Haldol, Plenur, Lamictal*).

---

## ⚠️ Fronteras y Límites: Qué NO puede responder el RAG

Para mantener expectativas realistas sobre el sistema interactivo, el usuario debe tener en cuenta lo siguiente:

| Categoría de Consulta | Capacidad del RAG | Razón / Comportamiento del Sistema |
| :--- | :---: | :--- |
| **Psicofármacos no indexados en monografía** *(ej. Clonazepam, Lorazepam, Mirtazapina, Cariprazina, Trazodona)* | ❌ Limitada | Si el principio activo no está en `pre_training/`, el modelo declarará no tener suficiente contexto clínico para emitir una recomendación segura. |
| **Medicina General / No Psiquiátrica** *(ej. antibióticos, oncología, antihipertensivos generales)* | ❌ Fuera de alcance | El corpus está restringido a psicofármacos y a interacciones directas con ellos (como AINEs/IECA con Litio). |
| **Diagnóstico Psicopatológico Puro** *(ej. criterios diagnósticos DSM-5/CIE-11 desvinculados de la medicación)* | ❌ No diseñado | Sofia es un asesor psicofarmacológico, no una herramienta de entrevista o diagnóstico clínico formal. |
| **Psicoterapia y Abordajes no Farmacológicos** | ❌ No disponible | No incluye protocolos de terapia cognitivo-conductual, psicoanálisis o terapias de tercera generación. |
| **Cálculo de Aclaramiento Dinámico de Creatinina** | ⚠️ Orientativo | Puede indicar la necesidad de ajuste renal según la monografía, pero no calcula ecuaciones dinámicas (Cockcroft-Gault / CKD-EPI) a partir de valores sueltos de laboratorio. |
| **Consultas Abiertas sin Evidencia en el Corpus** | 🛡️ Grounding Estricto | Umbral de similitud en modo TF-IDF (mensaje de "sin cobertura") y prompt que prohíbe inventar datos ausentes del contexto. La detección léxica es heurística, no infalible. |

---

## ⚙️ Modos de Inferencia en la App Interactiva

El motor [`ClinicalRAGEngine`](src/rag/engine.py) implementa una arquitectura híbrida con dos vías de ejecución según la disponibilidad de recursos y conectividad:

```mermaid
flowchart TD
    Q["Consulta del Usuario"] --> VS["ClinicalVectorStore (Recuperación top_k=3)"]
    VS -->|Ollama Activo (Local o Cloud API)| V_Densa["Búsqueda Semántica Densa (embeddinggemma)"]
    VS -->|Ollama Inactivo / Sin API Key| V_TFIDF["Búsqueda Léxica TF-IDF Scikit-Learn (ngram 1,2)"]
    
    V_Densa --> Ctx["Ensamblado de Contexto con Identificadores [DOC-ID]"]
    V_TFIDF --> Ctx
    
    Ctx --> LLM_Check{"¿Servidor LLM Disponible?"}
    LLM_Check -->|Sí: Ollama Local / Cloud API| Gen_LLM["Generación Neural Estructurada en 4 Apartados<br>(qwen3.5:2b-mlx / T=0.2)"]
    LLM_Check -->|No: Contingencia Sin LLM| Gen_Det["Plantilla Fija de Texto (4 frases hardcoded)<br>rellenada con la Ficha Recuperada"]
    
    Gen_LLM --> Resp["Respuesta Clínica Dinámica + Voice Snippet + Fuentes [DOC-ID]"]
    Gen_Det --> Resp
```

### 1. Modo Neural Completo (Ollama Local o Cloud API)
- **Embeddings:** `embeddinggemma:latest` para búsqueda semántica vectorial densa basada en significado clínico.
- **LLM:** `qwen3.5:2b-mlx` (temperatura 0.2) con el prompt clínico de Sofia (`SOFIA_CLINICAL_PROMPT`).
- **Salida:** Respuesta clínica redactada dinámicamente en 4 apartados estructurados, razonando sobre el caso concreto, citando identificadores `[DOC-ID]` y extrayendo el *Voice Snippet* (<40s).
- 📱 **Requisito en ordenadores básicos, teléfonos móviles o Streamlit Cloud:**  
  Un teléfono móvil o un servidor web gratuito sin GPU (como Streamlit Community Cloud) **no puede correr `qwen` ni `embeddinggemma` localmente**. Si se abre la aplicación desde un teléfono o a través de internet sin hardware dedicado, **se necesita obligatoriamente conexión y una API Key de Ollama Cloud** (o un servidor Ollama remoto accesible) para que el modelo de lenguaje pueda redactar y responder.

### 2. Modo Contingencia (Sin servidor LLM / Fallback Scikit-Learn)
- **¿Qué significa que "el modelo conmuta a contingencia"?:**  
  En este modo **no hay ningún modelo de IA en ejecución**. El modelo de lenguaje generativo está completamente apagado. En su lugar, el sistema recurre a un **algoritmo estadístico clásico de búsqueda de palabras de los años 70 (TF-IDF de Scikit-Learn)** que simplemente busca coincidencias léxicas exactas de términos en los archivos Markdown.
- **¿Qué responde en contingencia?:**  
  No hay redacción ni razonamiento clínico. El sistema devuelve una **plantilla estática fija de 4 frases predefinidas (hardcoded en el código)** que rellena con el título y el resumen pregrabado del documento recuperado (`"Seguir la dosificación descrita en [DOC-ID]"`). No responderá dudas médicas personalizadas ni interpretará combinaciones farmacológicas que no estén escritas literalmente.
- **¿Qué significa que "nunca se rompe"?:**  
  Esta afirmación aplica **exclusivamente a nivel de estabilidad de software:** gracias a los bloques de captura de excepciones (`try...except`), la aplicación web **no lanza una pantalla roja de error de Python** (`ConnectionRefusedError` o `Timeout`) ante la caída del servidor. Sin embargo, a nivel funcional y de experiencia de usuario, **el modelo generativo no está activo** y el chat funciona como un simple visor rígido de fichas.

---

## 🖥️ Módulos de la Aplicación Web (`streamlit_app.py`)

La aplicación interactiva divide sus capacidades en cuatro pestañas especializadas:

1. **💬 Asistente RAG Clínico:** Chat en lenguaje natural con respuestas en 4 secciones (*Juicio Clínico, Mecanismo y Cinética, Pauta Operativa, Alertas de Seguridad*), tarjeta de *Voice Snippet* verbalizable y panel desplegable de documentos clínicos recuperados.
2. **⚡ Matriz Interactiva CYP450:** Selector múltiple de fármacos que evalúa combinaciones mediante un semáforo clínico determinista (bloqueos de CYP2D6, duplicidad de ISRS, neurotoxicidad lítica, advertencias de Clozapina/Lamotrigina).
3. **🛑 Detector de Cascadas de Prescripción:** Selector de fármaco primario + síntoma emergente que identifica efectos adversos enmascarados y despliega protocolos de desprescripción.
4. **📖 Guías CANMAT y Catálogo:** Explorador interactivo con filtros por patología (TDM vs. Bipolar) y buscador de marcas comerciales y códigos ATC oficiales.

---

## 📊 Evaluación Formal del Sistema RAG (Benchmark)

La evaluación del RAG se ejecuta con [`scripts/evaluate_rag.py`](scripts/evaluate_rag.py) sobre cuatro escenarios clínicos (smoke test preliminar, **N=4**). El informe versionado ([`rag_evaluation_report.json`](data/processed/rag_evaluation_report.json)) se generó en **modo TF-IDF sin LLM** (`top_k=3`, igual que la app); por tanto **no mide la calidad de la generación del modelo de lenguaje**.

| Caso Evaluado | Categoría Clínica | Documento Clave | Context Recall | Cobertura de entidades | Groundedness (heurístico) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Caso 1** | Monografía y Posología (Sertralina) | `[MONO-SERTRALINA-001]` | 100.0% | 60.0% | 100.0% |
| **Caso 2** | Citocromos CYP450 (Paroxetina + Bupropión) | `[INTER-CYP2D6-001]` | 100.0% | 100.0% | 100.0% |
| **Caso 3** | Guías CANMAT (Manía y Viraje Bipolar) | `[GUIA-CANMAT-MANIA-001]` | 100.0% | 40.0% | 70.0% |
| **Caso 4** | Farmacovigilancia y Cascada (Haloperidol - Biperideno) | `[CASC-HAL-BIP-001]` | 100.0% | 80.0% | 100.0% |

> *Nota de honestidad metodológica:* "Groundedness" es una **métrica heurística** (presencia de los 4 apartados + cobertura de entidades esperadas), no una medida de fidelidad semántica; en modo contingencia la plantilla fija ya incluye los 4 apartados. El *Context Recall* indica que el documento esperado aparece entre los 3 recuperados para estas 4 preguntas; no debe interpretarse como cobertura sobre toda la farmacopea médica universal. Las latencias de generación con LLM en local dependen del hardware utilizado y no están reflejadas en el informe preliminar generado en modo contingencia.


---

## 📁 Estructura del Repositorio

```text
Sofia/
├── pre_training/                               # 26 documentos clínicos Markdown + Manifiesto técnico
│   ├── 01_monografias_farmacologicas/          # 9 monografías completas
│   ├── 02_interacciones_farmacologicas_cyp450/ # Interacciones enzimáticas críticas
│   ├── 03_cascadas_prescripcion_y_desprescripcion/
│   ├── 04_guias_clinicas_y_algoritmos_decision/ # Recomendaciones CANMAT
│   └── 05_alertas_seguridad_y_caja_negra/      # Agranulocitosis, Stevens-Johnson, litemia
├── tests/                                      # Tests de regresión automatizados (pytest)
│   └── test_core.py                            # 19 tests unitarios para ATC y motor RAG
├── data/
│   ├── raw/                                    # Fuentes originales en PDF (Jufe, CANMAT)
│   └── processed/                              # CSVs limpios y base de conocimiento (rag_knowledge_base.json)
├── notebooks/                                  # Pipeline de datos y análisis (00 a 04)
│   ├── 00_limpieza_y_eda.ipynb                 # Auditoría de calidad y visualizaciones clínicas
│   ├── 01_preprocesamiento_jufe.ipynb          # Extracción y estructuración de Jufe
│   ├── 02_preprocesamiento_stahl.ipynb         # Extracción de Stahl y citocromos CYP450
│   ├── 03_preprocesamiento_canmat.ipynb        # Recomendaciones y niveles de evidencia CANMAT
│   └── 04_rag_psicofarmacologia.ipynb          # Fusión de base de conocimiento y RAG
├── scripts/                                    # Scripts de evaluación y mantenimiento
│   └── evaluate_rag.py                         # Benchmark formal del sistema RAG (smoke test N=4)
├── src/
│   ├── enrichers/                              # Catálogo ATC de la OMS y marcas comerciales
│   ├── extractors/                             # Extractores de texto clínico estructurado
│   └── rag/                                    # Motor ClinicalRAGEngine y ClinicalVectorStore
├── streamlit_app.py                            # Aplicación web interactiva multi-pestaña
├── requirements.txt                            # Dependencias de ejecución de la app (Streamlit, Scikit-Learn, Pandas)
├── requirements-dev.txt                        # Dependencias de desarrollo, notebooks y pytest
└── LICENSE                                     # Licencia MIT
```

---

## 🚀 Instalación y Puesta en Marcha Local

### 1. Clonar el repositorio e instalar dependencias
```bash
git clone https://github.com/anacolinaarismendi/Sofia.git
cd Sofia
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. (Opcional) Activar inferencia neural local con Ollama
```bash
# Descargar modelo de embeddings densos y un LLM compatible:
ollama pull embeddinggemma:latest
ollama pull qwen2.5:3b
```
> 💡 *Nota de modelos:* Sofia utiliza por defecto `embeddinggemma:latest` para embeddings semánticos densos. Para la generación neural con LLM se recomienda un modelo ligero como `qwen2.5:3b` o `qwen2.5:1.5b`. Si cuentas con un modelo personalizado o cuantizado (como `qwen3.5:2b-mlx` en Mac), puedes configurarlo fácilmente en la barra lateral de la app o mediante la variable de entorno `export OLLAMA_MODEL="tu-modelo"`.

### 3. Ejecutar la aplicación interactiva
```bash
streamlit run streamlit_app.py
```

### 4. (Opcional) Tests y entorno de desarrollo
```bash
pip install -r requirements-dev.txt
pytest tests
```
> ℹ️ *Pipeline de datos y scripts:* El repositorio versiona [`scripts/evaluate_rag.py`](scripts/evaluate_rag.py) para evaluar el motor RAG. La base de datos y el archivo de conocimiento [`rag_knowledge_base.json`](data/processed/rag_knowledge_base.json) se construyen directamente a través de los cuadernos en [`notebooks/`](notebooks/) (`01 -> 02 -> 03 -> 00 -> 04`). Para ejecutar todos los cuadernos de forma secuencial y desatendida en local, se puede emplear el runner headless [`scripts/run_notebook_in_process.py`](scripts/run_notebook_in_process.py) (mantenido localmente en `.gitignore`).

---

## 🔮 Roadmap de Expansión

Para ampliar el alcance de Sofia hacia un vademécum psiquiátrico de cobertura extendida:
1. **Ampliación del Vademécum:** Redacción de monografías para antipsicóticos atípicos de última generación (Cariprazina, Brexpiprazol, Lurasidona) y estabilizadores de segunda línea (Valproato, Carbamazepina).
2. **Conexión en Vivo con APIs Sanitarias:** Integración con la API de **CIMA (AEMPS)** y **OpenFDA** para actualizar estados de desabastecimiento y prospectos oficiales en tiempo real.
3. **Cálculo de Función Renal:** Incorporación de una calculadora de filtrado glomerular (CKD-EPI) vinculada a los protocolos de ajuste de Litio.

---

## ⚖️ Aviso Legal y de Responsabilidad Médica

**Sofia es una herramienta de soporte a la decisión clínica con fines formativos, de investigación y de farmacovigilancia.**  
No sustituye el juicio clínico independiente del médico especialista, quien mantiene en todo momento la responsabilidad última sobre el diagnóstico, la prescripción y el seguimiento individualizado de cada paciente.
