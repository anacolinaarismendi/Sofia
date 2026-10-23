# Sofia — Asistente Clínico Inteligente y RAG de Psicofarmacología

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sofia-clinical-rag.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![RAG Benchmark](https://img.shields.io/badge/RAG%20Benchmark-Validated-success.svg)](data/processed/rag_evaluation_report.json)

> 🚀 **Aplicación Web Desplegada en Streamlit Cloud:**  
> Accede directamente a la versión interactiva en la nube: **[https://sofia-clinical-rag.streamlit.app/](https://sofia-clinical-rag.streamlit.app/)**

**Sofia** es un sistema integral de soporte a la decisión clínica, farmacovigilancia y generación asistida por recuperación (RAG) especializado en psiquiatría y psicofarmacología. Su arquitectura está orientada a resolver problemas clínicos de alta complejidad en polifarmacia:
- Interacciones enzimáticas en el citocromo P450 (**CYP1A2, CYP2D6, CYP3A4, CYP2B6**).
- Prevención y reversión de **cascadas de prescripción** (ej. Haloperidol $\to$ Biperideno por extrapiramidalismo).
- Jerarquía de evidencia según guías internacionales **CANMAT / ISBD 2018** (Trastorno Depresivo Mayor y Trastorno Bipolar).
- Soporte dual: **escrito estructurado** (4 secciones clínicas) y **oral verbalizable** (*Voice Snippets* < 40 segundos para asistentes de voz).

---

## 📚 Naturaleza del Corpus y Política Ética de Citación

> [!IMPORTANT]
> **Corpus Clínico de Elaboración Propia:**  
> La base de conocimiento de Sofia ha sido elaborada, sintetizada y estructurada internamente en fichas técnicas con identificadores únicos (`[DOC-ID]`), tomando como marco de evidencia científica las obras canónicas de la **Dra. Gabriela Jufe** (*Psicofarmacología Práctica*), el **Dr. Stephen Stahl** (*Prescriber's Guide*), las directrices oficiales **CANMAT / ISBD 2018** y la clasificación **ATC de la OMS**.

### Estándar de Citación y Trazabilidad por Identificador (`[DOC-ID]`)
Para salvaguardar la integridad médica y prevenir la **falsa atribución** (*citas apócrifas a nombres personales sobre textos generados*), **Sofia respalda cada afirmación clínica citando exclusivamente el identificador único del documento del corpus** (ej: `[MONO-SERT-001]`, `[CYP-2D6-001]`, `[CASC-HAL-BIP-001]`, `[CANMAT-BIP-MANIA-001]`), garantizando un rastro de auditoría (*audit trail*) transparente y verificable por el profesional médico.

---

## 🏗️ Arquitectura del Sistema

```mermaid
flowchart TD
    A["Corpus pre_training/<br>26 Docs Markdown con YAML y Voice Snippets"] --> B["Almacén Vectorial Dual<br>ClinicalVectorStore"]
    B -->|Modo Preferente| C["Ollama Local<br>embeddinggemma:latest"]
    B -->|Fallback Local| D["Scikit-Learn<br>TF-IDF Híbrido ngram 1,2"]
    
    E["Consulta Médica"] --> B
    B --> F["Top-k Documentos Recuperados + Voice Snippet"]
    
    F --> G["Plantilla Clínica Sofia<br>Clinical System Prompt con Grounding [DOC-ID]"]
    G --> H["Ollama LLM<br>qwen3.5:2b-mlx o Fallback Determinista"]
    H --> I["Respuesta Médica en 4 Apartados<br>+ Audio Snippet <40s"]
```

### Estructura del Repositorio

```text
Sofia/
├── pre_training/                   # 27 Documentos clínicos curados con YAML Frontmatter y Voice Snippets
│   ├── 00_MANIFIESTO_Y_ESPECIFICACION_EMBEDDINGS.md
│   ├── 01_monografias_farmacologicas/          # 9 Fármacos base (Sertralina, Clozapina, etc.)
│   ├── 02_interacciones_farmacologicas_cyp450/ # CYP2D6, CYP1A2, Litio, Lamotrigina + Valproato
│   ├── 03_cascadas_prescripcion_y_desprescripcion/
│   ├── 04_guias_clinicas_y_algoritmos_decision/ # CANMAT TDM y Bipolar
│   └── 05_alertas_seguridad_y_caja_negra/      # Agranulocitosis, Stevens-Johnson, etc.
├── finetunningpre/                 # Dataset de Fine-Tuning para modelos Qwen/Ollama
│   ├── dataset_sofia_medical_qwen.jsonl       # 29 Casos clínicos en formato ChatML
│   ├── Modelfile.example                      # Configuración para despliegue en Ollama
│   └── README_FINETUNING.md                   # Guía de entrenamiento LoRA / Unsloth
├── data/
│   ├── raw/                        # Documentos PDF fuente (CANMAT BDI-20-97, Jufe)
│   ├── cache/                      # Caché de páginas extraídas
│   └── processed/                  # CSVs limpios (0 nulos) y base de conocimiento JSON
│       ├── canmat_alertas_clinicas_clean.csv
│       ├── canmat_recomendaciones_clean.csv
│       ├── jufe_efectos_adversos_clean.csv
│       ├── jufe_farmacos_clean.csv
│       ├── stahl_eventos_adversos_clean.csv
│       ├── stahl_farmacos_clean.csv
│       ├── rag_knowledge_base.json
│       └── rag_evaluation_report.json
├── notebooks/                      # Cuadernos interactivos (100% ejecutables, 0 rutas de usuario)
│   ├── 00_limpieza_y_eda.ipynb                # Auditoría de datos y gráficos estadísticos
│   ├── 01_preprocesamiento_jufe.ipynb         # Catálogo de Jufe y eventos adversos
│   ├── 02_preprocesamiento_stahl.ipynb        # Perfil de receptores y citocromos Stahl
│   ├── 03_preprocesamiento_canmat.ipynb       # Algoritmos y recomendaciones CANMAT
│   └── 04_rag_psicofarmacologia.ipynb        # Motor RAG y generación clínica
├── src/                            # Módulos Python reutilizables
│   ├── enrichers/
│   │   └── atc_enricher.py                    # Clasificación ATC OMS y marcas comerciales
│   ├── extractors/
│   │   ├── pdf_extractor.py                   # Extractor de PDFs con caché en disco
│   │   ├── jufe_extractor.py                  # Extractor clínico de Jufe
│   │   └── canmat_extractor.py                # Extractor de directrices CANMAT
│   ├── downloaders/
│   │   └── guidelines_downloader.py           # Gestor de disponibilidad de guías
│   └── rag/
│       └── engine.py                          # ClinicalRAGEngine y ClinicalVectorStore
├── scripts/
│   └── evaluate_rag.py                        # Benchmark y evaluación formal de la Tríada de RAG
├── streamlit_app.py                # Interfaz web multimodular (Chat, CYP450, Cascadas, CANMAT)
└── requirements.txt                # Dependencias del proyecto
```

---

## 🚀 Inicio Rápido

### 1. Clonar el repositorio e instalar dependencias
```bash
git clone https://github.com/anacolinaarismendi/Sofia.git
cd Sofia
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. (Opcional) Modelos locales en Ollama
Sofia funciona de forma 100% autónoma gracias a su fallback determinista con Scikit-Learn TF-IDF. Para habilitar la inferencia neural completa con LLM local:
```bash
# Descargar modelo de embeddings densos
ollama pull embeddinggemma:latest

# Descargar modelo de generación clínica
ollama pull qwen3.5:2b-mlx
```

---

## 📓 Ejecución de Cuadernos Jupyter

Todos los cuadernos están diseñados bajo la directriz estricta de **cero rutas de usuario hardcodeadas** y son **100% ejecutables de principio a fin** (`Kernel -> Restart & Run All`):

```bash
jupyter lab
```

| Cuaderno | Propósito Clínico y Técnico | Estado |
| :--- | :--- | :--- |
| [`00_limpieza_y_eda.ipynb`](notebooks/00_limpieza_y_eda.ipynb) | Auditoría de nulos, distribución de severidad de eventos adversos, matriz de calor (*heatmap*) CYP450 y jerarquía CANMAT. | ✅ 100% Ejecutable |
| [`01_preprocesamiento_jufe.ipynb`](notebooks/01_preprocesamiento_jufe.ipynb) | Extracción monográfica de Jufe, posologías, rangos y normalización de efectos adversos. | ✅ 100% Ejecutable |
| [`02_preprocesamiento_stahl.ipynb`](notebooks/02_preprocesamiento_stahl.ipynb) | Extracción de perfiles de receptores, perlas clínicas y matriz binaria de citocromos. | ✅ 100% Ejecutable |
| [`03_preprocesamiento_canmat.ipynb`](notebooks/03_preprocesamiento_canmat.ipynb) | Tablas de 1ª/2ª línea para TDM y Bipolar, y alertas críticas de viraje maníaco. | ✅ 100% Ejecutable |
| [`04_rag_psicofarmacologia.ipynb`](notebooks/04_rag_psicofarmacologia.ipynb) | Pipeline RAG completo con vector store, Clinical System Prompt (`SOFIA_CLINICAL_PROMPT`) y consultas de farmacovigilancia. | ✅ 100% Ejecutable |

Los cuadernos son 100% reproducibles y pueden ejecutarse de forma secuencial en cualquier entorno Jupyter estándar o VS Code sin dependencias ocultas ni rutas absolutas.

---

## 📊 Evaluación Formal del Sistema RAG (Benchmark Clínico)

El sistema cuenta con una suite de evaluación cuantitativa ejecutada sobre preguntas clínicas reales (sin identificadores embebidos) y con generación neural completa mediante LLM (`qwen3.5:2b-mlx` en Ollama):

```bash
python scripts/evaluate_rag.py
```

### Resultados de la Tríada de RAG (Evaluación Real con LLM Activo):

| Métrica de la Tríada | Resultado Obtenido | Criterio de Aceptación | Estado |
| :--- | :---: | :---: | :---: |
| **Context Recall (Recuperación Canónica)** | **100.0%** | $\ge 80\%$ | ✅ Superado |
| **Answer Relevance (Relevancia y Cobertura)** | **95.0%** | $\ge 85\%$ | ✅ Superado |
| **Faithfulness / Groundedness (Cero Alucinaciones)** | **96.9%** | $\ge 90\%$ | ✅ Superado |
| **Trazabilidad Ética de Fuentes (`[DOC-ID]`)** | **100.0%** | $100\%$ | ✅ Superado |
| **Cobertura de Voice Snippets Verbalizables** | **100.0% (4/4)** | $100\%$ | ✅ Superado |
| **Latencia Media End-to-End con LLM** | **~21.9 s** | $< 30\text{ s}$ | ✅ Óptimo (Inferencia Local 2B) |

*El informe completo con el desglose de cada escenario y respuesta generada se exporta automáticamente a [`data/processed/rag_evaluation_report.json`](data/processed/rag_evaluation_report.json).*

---

## 💻 Aplicación Web Interactiva (Streamlit)

La aplicación está disponible en la nube o para ejecución local inmediata:

- 🌐 **Versión en Producción (Streamlit Cloud):** [https://sofia-clinical-rag.streamlit.app/](https://sofia-clinical-rag.streamlit.app/)
- 💻 **Ejecución Local:**
  ```bash
  streamlit run streamlit_app.py
  ```

### Módulos Clínicos y Capturas de la Interfaz:

#### 1. 💬 Asistente RAG Clínico y Briefing Oral
Interfaz conversacional con respuestas estructuradas en 4 apartados (*Juicio Clínico, Mecanismo y Cinética, Pauta Operativa, Alertas de Seguridad*), reproductor de **Voice Snippet** (< 40s) y trazabilidad completa de citas y fragmentos de evidencia recuperados. Soporta inferencia remota vía **Ollama Cloud** / API remota para su despliegue público en la nube.

![Interfaz del Chat RAG de Sofia](assets/sofia_chat_interface.jpg)

#### 2. ⚡ Matriz Interactiva CYP450 (Semáforo Enzimático)
Selector dinámico de combinaciones de psicofármacos. Detección inmediata de bloqueos enzimáticos críticos (ej. Paroxetina + Bupropión sobre CYP2D6 con riesgo de convulsiones), duplicidad de ISRS (síndrome serotoninérgico), neurotoxicidad de Litio (combinación con Haloperidol / síndrome de Cohen & Cohen) o advertencias de interacciones. Si una pareja no cuenta con evidencia formal en la matriz, el sistema clasifica con rigor preventivo como *"Sin datos"* en lugar de asumir riesgo bajo por defecto.

![Matriz Semáforo CYP450](assets/sofia_cyp450_matrix.jpg)

#### 3. 🛑 Detector de Cascadas de Prescripción
Evaluación heurística que distingue entre un efecto adverso inducido y una nueva patología. Genera alertas críticas ante prescripciones de anticolinérgicos (Biperideno) frente a síntomas extrapiramidales inducidos por Haloperidol, sugiriendo protocolos seguros de desprescripción y rotación a antipsicóticos atípicos.

![Detector de Cascadas de Prescripción](assets/sofia_detector_cascadas.jpg)

#### 4. 📖 Explorador CANMAT y Catálogo Farmacológico
Filtro interactivo de directrices internacionales por patología y nivel de evidencia (1ª, 2ª y 3ª línea para Trastorno Depresivo Mayor y Trastorno Bipolar), junto con buscador de principios activos, laboratorios, marcas comerciales registradas en España y códigos ATC oficiales de la OMS.

---

## 🔮 ¿Qué mejorarías? (Posibles Mejoras Futuras / Roadmap)

Para evolucionar Sofia desde un prototipo avanzado de soporte clínico hacia una herramienta certificable de grado médico hospitalario, las siguientes líneas de desarrollo representan las prioridades técnicas y clínicas:

1. **Integración en Tiempo Real con APIs Sanitarias Oficiales (AEMPS / CIMA y OpenFDA):**
   - Conectar el enriquecedor de fármacos con la API pública de **CIMA (Agencia Española de Medicamentos y Productos Sanitarios)** para auditar prospectos, fichas técnicas actualizadas y alertas de desabastecimiento o retirada de lotes en tiempo real.
   - Integrar **OpenFDA Adverse Event Reporting System (FAERS)** para computar señales post-comercialización de farmacovigilancia emergente.

2. **Expansión del Grafo de Farmacocinética (CYP450 y Aclaramiento Renal):**
   - Extender la base de conocimiento de 9 a más de 50 moléculas psicofarmacológicas, incluyendo estabilizadores del ánimo complejos (Ácido Valproico, Carbamazepina) y antipsicóticos de última generación (Cariprazina, Brexpiprazol, Lurasidona).
   - Incorporar algoritmos de dosificación según función renal mediante cálculo dinámico de filtrado glomerular (**CKD-EPI / Cockcroft-Gault**) para fármacos de estrecho margen terapéutico como el Litio o la Amisulprida.

3. **Voz Bidireccional en Tiempo Real (Full Audio Streaming):**
   - Sustituir la síntesis de voz estática por una canalización bidireccional basada en **WebRTC + Whisper Turbo** (transcripción en streaming con baja latencia) y síntesis neural expresiva (**XTTS-v2 / ElevenLabs**) para permitir consultas manos libres del profesional durante la guardia médica.

4. **Enrutamiento Híbrido de Inferencia (SLM Local vs. LLM Clínico Cloud):**
   - Diseñar un clasificador de complejidad en la capa de entrada:
     - Consultas factuales o posológicas simples se resuelven mediante el Small Language Model local (SLM 2B/7B) garantizando latencia sub-segundo y máxima privacidad de datos.
     - Casos de alta polifarmacia y comorbilidades complejas se delegan a modelos médicos de gran escala (ej. Llama-3-70B-Instruct o Med-PaLM) a través de Ollama Cloud o endpoints dedicados con cifrado extremo a extremo (HIPAA/GDPR compliance).

5. **Validación Clínica Formal y Benchmarking con Casos Reales:**
   - Crear una batería de evaluación clínica estructurada con 100+ casos complejos extraídos de convocatorias de examen de la especialidad (MIR Psiquiatría / Board Certifications).
   - Realizar una validación ciega inter-observador con facultativos especialistas en psiquiatría para contrastar el rendimiento del motor RAG frente al criterio del comité clínico.

---

## 🔒 Privacidad y Portabilidad Garantizadas

- **Cero Rutas de Usuario:** Ningún script, cuaderno o archivo de salida serializa rutas locales del tipo `/Users/...`, `/home/...` o `C:\...`. Todos los accesos se anclan dinámicamente a la raíz del repositorio mediante `pathlib.Path`.
- **Aislamiento de Configuración:** Los entornos de ejecución de Jupyter y Matplotlib están configurados para operar dentro de directorios temporales locales ignorados por Git (`.jupyter_tmp/`, `.matplotlib_tmp/`), asegurando que no se generen conflictos de permisos en el sistema operativo del usuario.

---

## ⚖️ Licencia

Distribuido bajo la Licencia MIT. Consulta el archivo [`LICENSE`](LICENSE) para más detalles.
