#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_pretraining_corpus.py
Script de automatización para generar el corpus médico estructurado en Markdown
dentro de pre_training/ para el proyecto Sofia.

Optimizado para:
1. Modalidad dual: Síntesis de voz (TTS / Asistentes orales) mediante 'Voice Snippets' y texto clínico detallado.
2. Ingestión y vectorización en bases de datos vectoriales con modelos de embeddings densos.
3. Filtrado por metadatos mediante YAML Frontmatter estandarizado.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PRE_TRAINING_DIR = BASE_DIR / "pre_training"

def write_doc(rel_path: str, content: str):
    target = PRE_TRAINING_DIR / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"✓ Generado: {rel_path} ({len(content)} caracteres)")

def build_all():
    print("Iniciando generación del corpus en pre_training/ ...")

    # =========================================================================
    # 00. MANIFIESTO Y ESPECIFICACIÓN
    # =========================================================================
    write_doc(
        "00_MANIFIESTO_Y_ESPECIFICACION_EMBEDDINGS.md",
        """---
id: "MANIF-EMBED-000"
tipo: "especificacion_tecnica"
titulo: "Manifiesto y Especificación Técnica del Corpus pre_training"
version: "1.0.0"
proyecto: "Sofia - Asistente Clínico y RAG de Psicofarmacología"
autor: "Equipo de Arquitectura y Farmacología Clínica"
---

# Manifiesto y Guía de Ingestión para Modelos de Embeddings y Sistemas de Voz

## 1. Propósito del Corpus
El directorio `pre_training/` reúne el conocimiento clínico estructurado del proyecto **Sofia** (Dra. Gabriela Jufe, Dr. Stephen Stahl, Guías CANMAT/ISBD, clasificación ATC de la OMS y catálogo de marcas y laboratorios). Este material ha sido convertido y curado con un estándar unificado de **modalidad dual (verbal y escrita)**:
1. **Comunicación Verbal (Voice AI / Speech-to-Text / TTS):** Cada archivo contiene un bloque formal `Voice Snippet` (Resumen Ejecutivo Verbalizable) redactado en prosa fluida, sin tablas multidimensionales ni abreviaturas telegráficas, listo para ser sintetizado a audio en menos de 40 segundos sin distorsión prosódica.
2. **Consultas Escritas y Recuperación Semántica (RAG / Embeddings):** Cada archivo profundiza en la farmacodinamia, farmacocinética, interacciones CYP450, cascadas de prescripción y algoritmos de decisión, calibrado en longitud semántica (400 a 900 tokens) para evitar truncamientos y maximizar la precisión de recuperación (*recall* y *MRR*).

---

## 2. Esquema Estándar de Metadatos (YAML Frontmatter)
Cada documento inicia obligatoriamente con un bloque YAML estandarizado que permite el filtrado híbrido determinista (*metadata filtering*) antes o después de la búsqueda vectorial densa:

| Campo | Tipo | Descripción | Ejemplo |
| :--- | :--- | :--- | :--- |
| `id` | `string` | Identificador único del chunk | `"MONO-SERTRALINA-001"` |
| `tipo` | `enum` | Tipo de entidad médica | `monografia_farmacologica`, `interaccion_cyp450`, `cascada_prescripcion`, `guia_clinica`, `alerta_seguridad` |
| `farmaco` / `farmacos_involucrados` | `list[string]` | Principios activos mencionados | `["Sertralina"]` o `["Haloperidol", "Biperideno"]` |
| `atc_codigo` / `atc_codigos` | `list[string]` | Códigos ATC oficiales de la OMS | `["N06AB06"]` |
| `enzimas_cyp` | `list[string]` | Citocromos hepáticos vinculados | `["CYP2D6", "CYP1A2", "CYP3A4"]` |
| `fuentes_primarias` | `list[string]` | Libros y directrices que avalan el texto | `["Jufe - Psicofarmacología Práctica", "Stahl"]` |
| `nivel_evidencia` | `string` | Grado de evidencia científica | `"Level 1"`, `"Level 2"`, `"Guía Clínica"`, `"Alerta Regulatoria"` |
| `nivel_riesgo` | `enum` | Severidad clínica de la alerta | `Bajo`, `Moderado`, `Alto`, `Crítico / Emergencia` |
| `palabras_clave` | `list[string]` | Descriptores DeCS/MeSH en español | `["depresión", "inhibición enzimática", "parkinsonismo"]` |

---

## 3. Estrategia de Ingestión y Pipeline de Vectorización

### Paso 1: Extracción de Payload y Texto para el Embedding
Para maximizar la densidad semántica, se recomienda componer el texto de entrada al modelo de embeddings combinando el título, las palabras clave y el cuerpo completo, preservando el YAML como metadatos de la base de datos vectorial:

```python
import yaml

def parse_markdown_chunk(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    parts = text.split("---", 2)
    frontmatter = yaml.safe_load(parts[1])
    body = parts[2].strip()
    
    # Texto optimizado para vectorización
    embedding_text = f"TÍTULO: {frontmatter.get('titulo', frontmatter.get('id'))}\\n" \\
                     f"ENTIDADES: {', '.join(frontmatter.get('farmacos_involucrados', []))}\\n" \\
                     f"PALABRAS CLAVE: {', '.join(frontmatter.get('palabras_clave', []))}\\n\\n" \\
                     f"{body}"
                     
    return frontmatter, embedding_text, body
```

### Paso 2: Modelos de Embeddings Recomendados
1. **Ollama Local (Entorno Sofia):** `embeddinggemma:latest` (o `bge-m3:latest` para embeddings densos multilingües con longitud de contexto de hasta 8192 tokens).
2. **Servicios Cloud / OpenAI:** `text-embedding-3-small` o `text-embedding-3-large` con dimensión configurada en 1536 o 3072.
3. **Métrica de Distancia Recomendada:** Distancia de Coseno (*Cosine Similarity*).

### Paso 3: Mapeo de Payload en Base de Datos Vectorial (Qdrant / Chroma / SQLite-vec)
Almacenar en el vector store:
- `vector`: Array flotante generado por el embedding.
- `payload`:
  - `id`: `frontmatter["id"]`
  - `tipo`: `frontmatter["tipo"]`
  - `farmacos`: `frontmatter.get("farmacos_involucrados", [frontmatter.get("farmaco")])`
  - `atc`: `frontmatter.get("atc_codigo", frontmatter.get("atc_codigos"))`
  - `voice_snippet`: Texto extraído de la sección `## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)`
  - `contenido_completo`: Texto íntegro del Markdown.
"""
    )

    # =========================================================================
    # 01. MONOGRAFÍAS FARMACOLÓGICAS (9 Principios Activos)
    # =========================================================================

    write_doc(
        "01_monografias_farmacologicas/sertralina.md",
        """---
id: "MONO-SERTRALINA-001"
tipo: "monografia_farmacologica"
farmaco: "Sertralina"
atc_codigo: "N06AB06"
atc_grupo: "Antidepresivos / Inhibidores Selectivos de la Recaptación de Serotonina (ISRS)"
enzimas_cyp: ["CYP2D6 (inhibición leve/moderada a dosis altas)", "CYP2B6", "CYP3A4"]
marcas_principales: ["Zoloft (Pfizer)", "Serlift (Roemmers)", "Aremis (Almirall)", "Aserin"]
fuentes_primarias: ["Jufe - Psicofarmacología Práctica", "Stahl - Prescriber's Guide"]
nivel_riesgo: "Moderado"
palabras_clave: ["sertralina", "zoloft", "isrs", "depresion", "ansiedad", "cyp2d6", "embarazo", "seguridad cardiovascular"]
---

# Ficha Clínica y Conversacional: Sertralina (N06AB06)

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "La sertralina es un antidepresivo inhibidor selectivo de la recaptación de serotonina, fármaco de primera línea para el trastorno depresivo mayor y cuadros de ansiedad como el pánico y el trastorno obsesivo-compulsivo. Su dosis inicial típica es de 25 a 50 miligramos diarios por la mañana, titulable hasta un máximo de 200 miligramos al día. Destaca por su seguridad cardiovascular comprobada y porque interfiere significativamente menos con el citocromo CYP2D6 que la paroxetina o la fluoxetina, lo que la convierte en una opción idónea en pacientes polimedicados. Comercialmente se encuentra como Zoloft de Pfizer o Serlift de Roemmers."

## 2. Perfil Neurofarmacológico y Mecanismo de Acción
Según la Dra. Gabriela Jufe y el Dr. Stephen Stahl, la sertralina actúa como un potente inhibidor del transportador de serotonina (SERT). A diferencia de otros ISRS, posee una característica farmacodinámica diferencial única: un bloqueo débil pero clínicamente relevante del transportador de dopamina (DAT) y afinidad por receptores sigma-1. Esta suave acción dopaminérgica contribuye a una mejoría temprana de la anhedonia, el enlentecimiento psicomotor y la energía en comparación con otros serotoninérgicos puros.

## 3. Farmacocinética e Interacciones Enzimáticas (CYP450)
- **Vías de Metabolización:** Es metabolizada a nivel hepático, principalmente por desmetilación (dismetilsertralina, metabolito menos activo).
- **Interacciones con Citocromos:** Presenta una inhibición débil a moderada de CYP2D6 únicamente cuando se alcanzan dosis elevadas (150-200 mg/día). A dosis terapéuticas habituales (50-100 mg/día), su impacto sobre el aclaramiento de otros fármacos metabolizados por CYP2D6 es bajo, lo que reduce el riesgo de toxicidad cruzada con betabloqueantes o antipsicóticos.
- **Riesgo Serotoninérgico:** Está estrictamente contraindicada su combinación con inhibidores de la monoaminooxidasa (IMAO) debido al riesgo inminente de síndrome serotoninérgico potencialmente letal. Se requiere un período de lavado de al menos 14 días al rotar entre estos grupos.

## 4. Dosificación, Rangos y Poblaciones Especiales
- **Dosis Inicial:** 25 mg/día durante los primeros 4 a 7 días en pacientes con alta sensibilidad o comorbilidad ansiosa para evitar agitación paradójica; 50 mg/día en depresión clásica.
- **Rango Terapéutico:** 50 a 200 mg/día en toma única matinal con alimentos (para mitigar náuseas).
- **Poblaciones Especiales:**
  - *Adultos Mayores:* No requiere ajuste drástico de inicio, aunque se aconseja comenzar en 25 mg/día y evaluar hiponatremia secundaria a secreción inadecuada de hormona antidiurética (SIADH).
  - *Embarazo y Lactancia:* Considerada por las guías clínicas como uno de los antidepresivos de elección en el período perinatal debido a su baja excreción en leche materna y favorable perfil de seguridad fetal.
  - *Cardiopatía:* Fármaco de primera elección en pacientes con infarto agudo de miocardio reciente o patología coronaria inestable (estudio SADHART).

## 5. Eventos Adversos: Frecuentes vs. Graves
- **Efectos Comunes y Transitorios:** Náuseas, dispepsia, diarrea o heces blandas (por receptores 5-HT3 y 5-HT4 intestinales), insomnio inicial, cefalea y disfunción sexual (anorgasmia o retraso eyaculatorio persistente).
- **Efectos Graves / Raros:** Sangrado gastrointestinal (por depleción plaquetaria de serotonina, potenciado si se combina con AINEs o aspirina), hiponatremia y síndrome serotoninérgico si se coadministra con tramadol, triptanes o hierba de San Juan.

## 6. Guía de Comunicación Clínica (Pregunta y Respuesta)
- **Pregunta:** "¿Por qué prescribir sertralina en lugar de fluoxetina en un anciano con polifarmacia?"
- **Respuesta:** "Porque la sertralina tiene una vida media más corta (26 horas frente a las semanas de la norfluoxetina) y una inhibición mucho menor sobre el citocromo CYP2D6, permitiendo retirar el fármaco rápidamente si surgen complicaciones sin alterar los niveles de sus otros tratamientos."

## 7. Referencias y Validación
- Dra. Gabriela Jufe: *Psicofarmacología Práctica*, Capítulo de Antidepresivos ISRS.
- Dr. Stephen Stahl: *Prescriber's Guide*, Monografía de Sertralina.
"""
    )

    write_doc(
        "01_monografias_farmacologicas/fluoxetina.md",
        """---
id: "MONO-FLUOXETINA-002"
tipo: "monografia_farmacologica"
farmaco: "Fluoxetina"
atc_codigo: "N06AB03"
atc_grupo: "Antidepresivos / Inhibidores Selectivos de la Recaptación de Serotonina (ISRS)"
enzimas_cyp: ["CYP2D6 (inhibidor potente)", "CYP2C9/CYP2C19 (inhibidor moderado)", "CYP3A4"]
marcas_principales: ["Prozac (Eli Lilly)", "Foxetin (Gador)", "Sarafem", "Adofen"]
fuentes_primarias: ["Jufe - Psicofarmacología Práctica", "Stahl - Prescriber's Guide"]
nivel_riesgo: "Alto (por interacciones enzimáticas prolongadas)"
palabras_clave: ["fluoxetina", "prozac", "isrs", "cyp2d6", "vida media larga", "bulimia", "activador", "norfluoxetina"]
---

# Ficha Clínica y Conversacional: Fluoxetina (N06AB03)

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "La fluoxetina es un antidepresivo ISRS con un perfil marcadamente activador, indicado en depresión mayor, bulimia nerviosa y trastorno disfórico premenstrual. Su dosis habitual va de 20 a 60 miligramos diarios. Su principal característica diferencial es su prolongadísima vida media: su metabolito activo, la norfluoxetina, permanece en el organismo entre 7 y 15 días, lo que minimiza el síndrome de discontinuación por olvido de dosis. Sin embargo, es un potente inhibidor del citocromo CYP2D6, por lo que eleva drásticamente los niveles de antipsicóticos y betabloqueantes. Si se desea cambiar a un IMAO, exige un período de lavado estricto de cinco semanas. Su marca pionera es Prozac de Eli Lilly."

## 2. Perfil Neurofarmacológico y Mecanismo de Acción
Stahl y Jufe destacan que la fluoxetina no solo inhibe el SERT, sino que también es antagonista de los receptores 5-HT2C. El bloqueo 5-HT2C desinhibe la liberación de noradrenalina y dopamina en la corteza prefrontal, confiriéndole propiedades energizantes y alertantes desde el inicio del tratamiento. Es ideal en pacientes con apatía, fatiga mental y fatiga psicomotora, pero puede inducir ansiedad paradójica inicial en individuos con perfil hiperactivado.

## 3. Farmacocinética e Interacciones Enzimáticas (CYP450)
- **Vida Media Prolongada:** La fluoxetina tiene una semivida de 2 a 4 días, pero su metabolito norfluoxetina persiste entre 7 y 15 días.
- **Inhibición Enzimática Extrema:** Es uno de los inhibidores más potentes conocidos del citocromo CYP2D6. Puede multiplicar por 3 a 5 veces las concentraciones de risperidona, haloperidol, aripiprazol, atomoxetina, antidepresivos tricíclicos y betabloqueantes lipofílicos (como metoprolol o propranolol).
- **Ventana de Lavado Mandatoria:** Para iniciar un IMAO se deben esperar como mínimo 5 semanas completas tras la última toma de fluoxetina para evitar toxicidad serotoninérgica letal.

## 4. Dosificación, Rangos y Poblaciones Especiales
- **Dosis Habitual:** 20 mg/día matinales. En bulimia nerviosa se requiere titular hasta 60 mg/día para lograr control de atracones y purgas.
- **Poblaciones Especiales:**
  - *Pediatría:* Es uno de los pocos antidepresivos con aprobación formal y evidencia sólida para depresión en niños y adolescentes a partir de los 8 años.
  - *Olvido de Dosis:* Es el fármaco de elección en pacientes con mala adherencia o tomas erráticas gracias a su efecto de autotaponamiento cinético.

## 5. Eventos Adversos: Frecuentes vs. Graves
- **Comunes:** Agitación e inquietud psicomotora los primeros días, insomnio, disminución del apetito y peso, náuseas y temblor fino.
- **Graves:** Síndrome serotoninérgico en polifarmacia, hiponatremia en adultos mayores, prolongación de intervalo QTc a dosis supraterapéuticas.

## 6. Guía de Comunicación Clínica (Pregunta y Respuesta)
- **Pregunta:** "¿Puedo combinar fluoxetina con risperidona para la depresión psicótica?"
- **Respuesta:** "Se puede usar con precaución, pero teniendo presente que la fluoxetina inhibirá el metabolismo de la risperidona por CYP2D6, lo que multiplicará sus niveles plasmáticos y disparará el riesgo de rigidez extrapiramidal e hiperprolactinemia. Se debe reducir la dosis de risperidona a la mitad."

## 7. Referencias y Validación
- Dra. Gabriela Jufe: *Psicofarmacología Práctica*, Capítulo de ISRS.
- Dr. Stephen Stahl: *Prescriber's Guide*, Monografía de Fluoxetina.
"""
    )

    write_doc(
        "01_monografias_farmacologicas/paroxetina.md",
        """---
id: "MONO-PAROXETINA-003"
tipo: "monografia_farmacologica"
farmaco: "Paroxetina"
atc_codigo: "N06AB02"
atc_grupo: "Antidepresivos / Inhibidores Selectivos de la Recaptación de Serotonina (ISRS)"
enzimas_cyp: ["CYP2D6 (inhibidor potente / sustrato)", "CYP3A4"]
marcas_principales: ["Paxil (GSK)", "Seroxat", "Aropax", "Motivan"]
fuentes_primarias: ["Jufe - Psicofarmacología Práctica", "Stahl - Prescriber's Guide"]
nivel_riesgo: "Alto (riesgo de interacciones, abstinencia y teratogenicidad)"
palabras_clave: ["paroxetina", "paxil", "isrs", "anticolinergico", "sedacion", "sindrome discontinuacion", "teratogenico", "cyp2d6"]
---

# Ficha Clínica y Conversacional: Paroxetina (N06AB02)

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "La paroxetina es el antidepresivo ISRS más potente sobre el transportador de serotonina, pero también el más sedante y anticolinérgico. Indicada en depresión severa, crisis de pánico y fobia social en dosis de 20 a 50 miligramos al día. Es un potentísimo inhibidor de su propia enzima metabolizadora, el citocromo CYP2D6, provocando una cinética no lineal e interacciones severas. Presenta dos advertencias clínicas críticas: un síndrome de discontinuación muy intenso con síntomas de rebote ante la suspensión brusca, y riesgo de malformaciones cardíacas congénitas si se administra durante el primer trimestre de gestación. Su marca más conocida es Paxil o Seroxat."

## 2. Perfil Neurofarmacológico y Mecanismo de Acción
Stahl remarca que además de inhibir el SERT con afinidad nanomolar extrema, la paroxetina posee una afinidad significativa por receptores muscarínicos colinérgicos (M1) y una leve afinidad por el transportador de noradrenalina (NET). Esta actividad anticolinérgica intrínseca explica por qué produce más somnolencia, boca seca, estreñimiento y letargia que cualquier otro ISRS, resultando beneficiosa en pacientes con ansiedad extrema o insomnio pertinaz, pero peligrosa en ancianos.

## 3. Farmacocinética e Interacciones Enzimáticas (CYP450)
- **Cinética No Lineal:** Inhibe potentemente al citocromo CYP2D6, que es su propia vía metabólica (inhibición dependiente de mecanismo). Pequeños aumentos de dosis generan incrementos desproporcionados en sangre.
- **Interacciones Críticas:** Al anular la actividad de CYP2D6, impide la activación metabólica de profármacos como el tamoxifeno (reduciendo su eficacia oncológica contra el cáncer de mama) o la codeína/tramadol (impidiendo su conversión a morfina/O-desmetiltramadol y aboliendo el efecto analgésico). A la vez, intoxica al paciente si se combina con fármacos que dependen de 2D6 para eliminarse.

## 4. Dosificación, Rangos y Poblaciones Especiales
- **Dosis Habitual:** 20 a 50 mg/día (formulaciones de liberación prolongada: 12.5 a 62.5 mg/día). Administrar preferentemente por la noche.
- **Poblaciones Especiales:**
  - *Adultos Mayores:* Relativamente desaconsejada por el riesgo de deterioro cognitivo, retención urinaria, empeoramiento del glaucoma y caídas derivado de su acción anticolinérgica.
  - *Embarazo (Categoría D):* Contraindicada de elección en mujeres en edad fértil sin anticoncepción eficaz debido al riesgo demostrado de defectos septales ventriculares y auriculares en el feto.

## 5. Eventos Adversos: Frecuentes vs. Graves
- **Comunes:** Sedación diurna, marcada ganancia ponderal, disfunción sexual muy severa (la más alta entre los ISRS), sequedad bucal y estreñimiento.
- **Graves:** Síndrome de discontinuación precoz e invalidante (mareos, sensaciones de descarga eléctrica o 'brain zaps', náuseas, llanto fácil e insomnio) que ocurre incluso tras olvidar una sola toma debido a su corta semivida y ausencia de metabolitos activos.

## 6. Guía de Comunicación Clínica (Pregunta y Respuesta)
- **Pregunta:** "¿Cómo suspender la paroxetina en un paciente que lleva años tomándola?"
- **Respuesta:** "Debe realizarse mediante una reducción hiperlenta y escalonada, disminuyendo entre 5 y 10 miligramos cada 2 a 4 semanas, e incluso utilizando formulación líquida o puenteando a fluoxetina si el paciente sufre descargas eléctricas cefálicas o crisis de pánico por discontinuación."

## 7. Referencias y Validación
- Dra. Gabriela Jufe: *Psicofarmacología Práctica*, Capítulo de ISRS.
- Dr. Stephen Stahl: *Prescriber's Guide*, Monografía de Paroxetina.
"""
    )

    write_doc(
        "01_monografias_farmacologicas/bupropion.md",
        """---
id: "MONO-BUPROPION-004"
tipo: "monografia_farmacologica"
farmaco: "Bupropión"
atc_codigo: "N06AX12"
atc_grupo: "Antidepresivos / Inhibidores de la Recaptación de Noradrenalina y Dopamina (IRND)"
enzimas_cyp: ["CYP2B6 (sustrato principal)", "CYP2D6 (inhibidor potente)"]
marcas_principales: ["Wellbutrin (GSK)", "Zyban (GSK)", "Odranal", "Elontril"]
fuentes_primarias: ["Jufe - Psicofarmacología Práctica", "Stahl - Prescriber's Guide"]
nivel_riesgo: "Alto (riesgo de crisis convulsivas y potente inhibición de CYP2D6)"
palabras_clave: ["bupropion", "wellbutrin", "irnd", "dopamina", "noradrenalina", "convulsiones", "bulimia", "cyp2d6", "tabaquismo", "sin disfuncion sexual"]
---

# Ficha Clínica y Conversacional: Bupropión (N06AX12)

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "El bupropión es un antidepresivo atípico perteneciente a la familia de los inhibidores de la recaptación de noradrenalina y dopamina. Se utiliza en el trastorno depresivo mayor, la prevención del trastorno afectivo estacional y como terapia de cesación tabáquica, en dosis de 150 a 300 miligramos diarios en formulaciones de liberación modificada. Su gran ventaja clínica es que no provoca aumento de peso ni causa disfunción sexual, siendo un activador de la energía y la concentración. Su principal contraindicación estricta son los antecedentes de epilepsia y los trastornos de la conducta alimentaria como anorexia o bulimia, debido al riesgo dosis-dependiente de inducir convulsiones. Además, es un potente inhibidor del citocromo CYP2D6. Se comercializa como Wellbutrin o Elontril."

## 2. Perfil Neurofarmacológico y Mecanismo de Acción
Tanto Jufe como Stahl subrayan que el bupropión no posee acción serotoninérgica directa. Su mecanismo reside en la inhibición del transportador de dopamina (DAT) y de noradrenalina (NET), junto con un antagonismo no competitivo de receptores nicotínicos de acetilcolina. Esta última propiedad modula las vías de recompensa mesolímbicas, lo que fundamenta su eficacia para suprimir el deseo compulsivo de nicotina sin generar síntomas de abstinencia.

## 3. Farmacocinética e Interacciones Enzimáticas (CYP450)
- **Metabolización por CYP2B6:** Es transformado por el citocromo CYP2B6 en varios metabolitos activos (hidroxibupropión, treohidrobupropión). Fármacos inductores como carbamazepina reducen sus niveles.
- **Inhibición de CYP2D6:** A pesar de no ser metabolizado por CYP2D6, el bupropión y sus metabolitos son **potentes inhibidores de CYP2D6**. Si se añade bupropión a un paciente tratado con metoprolol, flecainida, paroxetina o risperidona, las concentraciones de estos últimos pueden dispararse a niveles tóxicos.

## 4. Dosificación, Rangos y Poblaciones Especiales
- **Dosis Habitual:** Formulaciones XL (liberación extendida 24 horas): iniciar con 150 mg/día matinales; titular a 300 mg/día tras 2 a 4 semanas si la respuesta es parcial. Dosis máxima absoluta: 300 a 450 mg/día (nunca exceder 150 mg por toma única en formulaciones SR para evitar picos convulsivos).
- **Contraindicaciones Absolutas:**
  - Antecedentes de crisis epilépticas o patología con umbral convulsivo disminuido.
  - Diagnóstico actual o previo de bulimia nerviosa o anorexia nerviosa (alto riesgo de crisis convulsivas inducidas por desequilibrios hidroelectrolíticos).
  - Suspensión brusca de alcohol o de benzodiacepinas.

## 5. Eventos Adversos: Frecuentes vs. Graves
- **Comunes:** Insomnio (evitar tomas vespertinas), boca seca, cefalea, temblor fino de manos, sudoración y pérdida de peso moderada.
- **Graves:** Crisis tónico-clónicas generalizadas (incidencia menor al 0.1% a dosis de 300 mg XL, pero se multiplica por 10 al superar 450 mg), inducción de psicosis o agitación maníaca en pacientes vulnerables.

## 6. Guía de Comunicación Clínica (Pregunta y Respuesta)
- **Pregunta:** "¿Se puede añadir bupropión a un paciente que toma escitalopram pero presenta disfunción sexual y fatiga residual?"
- **Respuesta:** "Sí, es una combinación sinérgica y frecuente recomendada por las guías. El bupropión contrarresta la apatía y la disfunción sexual provocadas por el ISRS, pero se debe vigilar que la presión arterial no aumente y recordar que el bupropión inhibirá moderadamente el aclaramiento de fármacos que usen CYP2D6."

## 7. Referencias y Validación
- Dra. Gabriela Jufe: *Psicofarmacología Práctica*, Capítulo de Antidepresivos IRND.
- Dr. Stephen Stahl: *Prescriber's Guide*, Monografía de Bupropión.
"""
    )

    write_doc(
        "01_monografias_farmacologicas/litio.md",
        """---
id: "MONO-LITIO-005"
tipo: "monografia_farmacologica"
farmaco: "Litio (Carbonato de Litio)"
atc_codigo: "N05AN01"
atc_grupo: "Psicolépticos / Antipsicóticos y Estabilizadores del Ánimo"
enzimas_cyp: ["Ninguna (excreción renal exclusiva no hepática)"]
marcas_principales: ["Plenur (España)", "Eskalith", "Carbolit", "Ceglution (LatAm)"]
fuentes_primarias: ["Jufe - Psicofarmacología Práctica", "CANMAT/ISBD 2018 Guidelines", "Stahl"]
nivel_riesgo: "Crítico (rango terapéutico estrecho y riesgo de toxicidad letal)"
palabras_clave: ["litio", "carbonato de litio", "plenur", "bipolar", "mania", "litemia", "suicidio", "toxicidad renal", "aines", "tiazidas"]
---

# Ficha Clínica y Conversacional: Carbonato de Litio (N05AN01)

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "El carbonato de litio es el estabilizador del ánimo de referencia y estándar de oro para el tratamiento del trastorno bipolar, tanto en la manía aguda como en el mantenimiento preventivo a largo plazo, siendo el único fármaco con evidencia demostrada de reducción del riesgo de suicidio. Carece de metabolismo hepático y se excreta íntegramente por vía renal. Posee un rango terapéutico extremadamente estrecho, exigiendo monitorizar la litemia en sangre entre 0.6 y 1.0 miliequivalentes por litro. Niveles superiores a 1.5 desencadenan toxicidad grave con temblor grueso, ataxia y desorientación. Su interacción con antiinflamatorios como el ibuprofeno, diuréticos tiazídicos e inhibidores de la ECA es muy peligrosa porque reduce su eliminación renal provocando intoxicaciones agudas. Se comercializa como Plenur."

## 2. Perfil Neurofarmacológico y Mecanismo de Acción
Jufe y CANMAT explican que el litio es un catión monovalente que modula múltiples cascadas intracelulares de segundos mensajeros. Inhibe la enzima glucógeno sintasa quinasa 3 beta (GSK-3β) y la vía del fosfatidilinositol (inosina monofosfatasa), promoviendo factores neurotróficos como el BDNF y ejerciendo una acción neuroprotectora y antisuicida intrínseca que no comparten otros estabilizadores anticonvulsivantes.

## 3. Farmacocinética e Interacciones Farmacológicas Críticas
- **Excreción Renal Exclusiva:** El litio se filtra por el glomérulo y se reabsorbe en un 70-80% en el túbulo proximal renal, compitiendo directamente con el sodio.
- **Interacciones Fatales por Disminución de Excreción:**
  - *AINEs (Ibuprofeno, Naproxeno, Diclofenaco, Ketorolaco):* Inhiben las prostaglandinas renales, reducen la tasa de filtración glomerular y aumentan la litemia hasta en un 40-60%.
  - *Diuréticos Tiazídicos (Hidroclorotiazida):* Provocan depleción de sodio, lo que activa una reabsorción masiva compensatoria de sodio y litio en el túbulo proximal, desatando intoxicación severa.
  - *IECA / ARA-II (Enalapril, Losartán):* Reducen la presión intraglomerular elevando la litemia.
- **Interacciones Seguras para el Dolor:** Paracetamol o ácido acetilsalicílico a dosis analgésicas habituales son los analgésicos de elección porque no alteran la eliminación renal de litio.

## 4. Dosificación, Rangos y Monitorización
- **Rango Terapéutico en Plasma (Litemia):**
  - *Fase de Mantenimiento / Profilaxis:* 0.6 a 0.8 mEq/L (hasta 1.0 mEq/L en pacientes con recaídas frecuentes).
  - *Fase de Manía Aguda:* 0.8 a 1.1 mEq/L.
  - *Toma de Muestra:* Se extrae sangre a las 12 horas exactas tras la última toma de la noche previa (*litemia basal*).
- **Parámetros Obligatorios de Laboratorio (cada 6 meses):**
  - Creatinina sérica y tasa de filtración glomerular (evaluación de nefropatía intersticial crónica).
  - TSH y T4 libre (detección precoz de bocio e hipotiroidismo subclínico o franco inducido por litio).
  - Calcemia e ionograma sérico.

## 5. Eventos Adversos: Frecuentes vs. Intoxicación
- **Efectos Secundarios Comunes y Estables:** Temblor postural fino de manos, poliuria y polidipsia (por resistencia parcial a la ADH), aumento de peso, acné o psoriasis y sabor metálico.
- **Signos de Toxicidad por Litio (Alerta Médica):**
  - *Leve a Moderada (1.2 - 1.5 mEq/L):* Náuseas persistentes, vómitos, diarrea acuosa, temblor grosero, debilidad muscular y letargia.
  - *Grave (> 2.0 mEq/L):* Ataxia cerebelosa, disartria, clonus, hiperreflexia, confusión mental, convulsiones, coma y riesgo de secuelas neurológicas permanentes (síndrome SILENT).

## 6. Guía de Comunicación Clínica (Pregunta y Respuesta)
- **Pregunta:** "Un paciente en tratamiento con Plenur refiere dolor muscular y quiere tomar ibuprofeno. ¿Qué indicarle?"
- **Respuesta:** "Debe contraindicarse tajantemente el ibuprofeno. Los antiinflamatorios no esteroideos reducen la filtración renal de litio y pueden inducir una intoxicación aguda en pocos días. Se debe pautar paracetamol de un gramo como analgésico seguro."

## 7. Referencias y Validación
- Guías CANMAT / ISBD 2018 para Trastorno Bipolar: Fármaco de Primera Línea (Evidencia Level 1).
- Dra. Gabriela Jufe: *Psicofarmacología Práctica*, Capítulo de Estabilizadores del Ánimo.
"""
    )

    write_doc(
        "01_monografias_farmacologicas/clozapina.md",
        """---
id: "MONO-CLOZAPINA-006"
tipo: "monografia_farmacologica"
farmaco: "Clozapina"
atc_codigo: "N05AH02"
atc_grupo: "Psicolépticos / Antipsicóticos Atípicos de Segunda Generación"
enzimas_cyp: ["CYP1A2 (sustrato principal)", "CYP2D6", "CYP3A4"]
marcas_principales: ["Clozaril (Novartis)", "Leponex", "Sequax", "Nemea"]
fuentes_primarias: ["Jufe - Psicofarmacología Práctica", "Stahl - Prescriber's Guide"]
nivel_riesgo: "Crítico (riesgo de agranulocitosis, miocarditis y convulsiones)"
palabras_clave: ["clozapina", "leponex", "clozaril", "esquizofrenia resistente", "agranulocitosis", "neutropenia", "cyp1a2", "tabaquismo", "ileo paralitico", "miocarditis"]
---

# Ficha Clínica y Conversacional: Clozapina (N05AH02)

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "La clozapina es el antipsicótico atípico más eficaz de la psiquiatría moderna, reservado como estándar de oro para la esquizofrenia resistente al tratamiento y la conducta suicida persistente. Apenas causa síntomas extrapiramidales y no eleva la prolactina. Su uso exige un riguroso control hematológico mediante hemogramas seriados debido al riesgo de agranulocitosis o neutropenia grave potencialmente mortal. Es metabolizada principalmente por el citocromo CYP1A2; los hidrocarburos del humo del tabaco inducen esta enzima y reducen sus niveles a la mitad, por lo que un paciente que deja de fumar bruscamente puede intoxicarse de inmediato. Además, requiere monitorización de estreñimiento severo para evitar íleo paralítico fatal. Su nombre comercial pionero es Leponex o Clozaril."

## 2. Perfil Neurofarmacológico y Mecanismo de Acción
Stahl describe a la clozapina como el prototipo de antipsicótico atípico multirreceptorial (*broad-spectrum*). Posee una afinidad por receptores serotoninérgicos 5-HT2A sustancialmente superior a su afinidad por receptores dopaminérgicos D2. Se disocia con extrema rapidez del receptor D2 (baja ocupación estriatal del 40-60%), lo que explica la ausencia virtual de parkinsonismo secundario, distonías agudas y discinesia tardía. Tiene gran afinidad por receptores D4, 5-HT2C, H1 histaminérgicos, alfa-1 adrenérgicos y muscarínicos M1-M4.

## 3. Farmacocinética y Metabolismo por Citocromos (CYP450)
- **Vía Principal CYP1A2:** El citocromo 1A2 es responsable del 70% de su aclaramiento.
- **Efecto Crítico del Tabaquismo:** Los hidrocarburos aromáticos policíclicos presentes en el humo de los cigarrillos son inductores potentes de CYP1A2. Los pacientes fumadores requieren dosis hasta un 50% superiores. Si un paciente ingresa en un hospital psiquiátrico donde no puede fumar, los niveles de clozapina se duplican en cuestión de 48 a 72 horas, pudiendo sufrir sedación comatosa, mioclono y crisis convulsivas.
- **Inhibidores Catastróficos (Fluvoxamina y Ciprofloxacino):** La fluvoxamina o el ciprofloxacino inhiben potentemente CYP1A2 y multiplican los niveles de clozapina por 5 a 10 veces, lo que está formalmente contraindicado.

## 4. Dosificación, Titulación y Protocolo Hematológico
- **Titulación Lenta Obligatoria:** Iniciar con 12.5 mg una o dos veces al día; incrementar de 25 a 50 mg/día cada 48 horas hasta alcanzar 300 a 450 mg/día. Dosis máxima: 900 mg/día. Una titulación rápida causa hipotensión ortostática fulminante con síncope y convulsiones.
- **Protocolo Hematológico Mandatorio (Recuento Absoluto de Neutrófilos - RAN):**
  - *Inicio del tratamiento:* Exige RAN ≥ 1500/µL (≥ 1000/µL en pacientes con neutropenia étnica benigna).
  - *Frecuencia de Hemogramas:* Semanal durante las primeras 18 semanas; mensual a partir de entonces mientras dure el tratamiento.
  - *Umbral de Interrupción:* Si el RAN cae por debajo de 1000/µL, suspender inmediatamente y derivar a hematología.

## 5. Eventos Adversos: Frecuentes vs. Graves
- **Comunes:** Sialorrea nocturna intensa (hipersalivación paradójica por agonismo M4), sedación profunda, aumento de peso muy marcado, dislipidemia y taquicardia sinusal refleja.
- **Graves (Caja Negra):**
  - *Agranulocitosis:* Incidencia aproximada del 0.8%, concentrada en los primeros 6 meses.
  - *Miocarditis y Miocardiopatía:* Típicamente en el primer mes; vigilar troponinas, PCR y ecocardiograma ante disnea, taquicardia o fiebre.
  - *Hipomotilidad Gastrointestinal Severa / Íleo Paralítico:* Mata a más pacientes que la agranulocitosis por necrosis intestinal secundaria a constipación desatendida.

## 6. Guía de Comunicación Clínica (Pregunta y Respuesta)
- **Pregunta:** "Un paciente en tratamiento con Clozapina ingresa en planta y deja de fumar. ¿Qué ajuste debe hacerse?"
- **Respuesta:** "Se debe reducir la dosis de clozapina aproximadamente entre un 30% y un 50% de forma progresiva y monitorizar la clozapinemia, ya que el cese de los hidrocarburos del tabaco desactivará la inducción de CYP1A2 y sus concentraciones plasmáticas se dispararán a niveles tóxicos."

## 7. Referencias y Validación
- Dra. Gabriela Jufe: *Psicofarmacología Práctica*, Capítulo de Antipsicóticos Atípicos.
- Dr. Stephen Stahl: *Prescriber's Guide*, Monografía de Clozapina.
"""
    )

    write_doc(
        "01_monografias_farmacologicas/risperidona.md",
        """---
id: "MONO-RISPERIDONA-007"
tipo: "monografia_farmacologica"
farmaco: "Risperidona"
atc_codigo: "N05AX08"
atc_grupo: "Psicolépticos / Antipsicóticos Atípicos"
enzimas_cyp: ["CYP2D6 (sustrato principal a 9-hidroxirisperidona)", "CYP3A4"]
marcas_principales: ["Risperdal (Janssen)", "Ridal (Roemmers)", "Sequinan", "Diafusor"]
fuentes_primarias: ["Jufe - Psicofarmacología Práctica", "CANMAT Guidelines", "Stahl"]
nivel_riesgo: "Moderado a Alto (riesgo de hiperprolactinemia y síntomas extrapiramidales dosis-dependientes)"
palabras_clave: ["risperidona", "risperdal", "antipsicotico atipico", "prolactina", "galactorrea", "cyp2d6", "parkinsonismo dosis dependiente"]
---

# Ficha Clínica y Conversacional: Risperidona (N05AX08)

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "La risperidona es un antipsicótico atípico de segunda generación ampliamente utilizado en la esquizofrenia, los episodios maníacos del trastorno bipolar y la irritabilidad grave en el autismo. Su dosis habitual en adultos oscila entre 2 y 6 miligramos al día. Aunque es un atípico, tiene una afinidad dopaminérgica D2 muy elevada: a dosis superiores a 6 miligramos pierde su perfil atípico y provoca síntomas extrapiramidales similares a los fármacos clásicos. Además, es el antipsicótico que más eleva la hormona prolactina, causando amenorrea, galactorrea y disfunción sexual. Es metabolizada por el citocromo CYP2D6; si se combina con inhibidores potentes como fluoxetina o paroxetina, sus niveles plasmáticos aumentan marcadamente. Su marca de referencia es Risperdal de Janssen."

## 2. Perfil Neurofarmacológico y Mecanismo de Acción
Jufe y Stahl detallan que la risperidona combina un antagonismo potente sobre los receptores serotoninérgicos 5-HT2A con un bloqueo marcado de receptores D2 y alfa-1 adrenérgicos. En dosis bajas a moderadas (1 a 4 mg/día), el bloqueo 5-HT2A amortigua el impacto en la vía motora nigroestriatal. No obstante, al superar los 6 mg/día, la ocupación de receptores D2 estriatales supera el umbral del 80%, provocando acatisia, temblor y rigidez muscular.

## 3. Farmacocinética y Metabolismo Enzimático (CYP450)
- **Metabolito Activo (Paliperidona):** Es convertida en el hígado por el citocromo CYP2D6 en 9-hidroxirisperidona (paliperidona), que posee una potencia farmacológica casi idéntica a la molécula madre. El efecto clínico depende de la *fracción antipsicótica activa* (suma de risperidona + 9-hidroxirisperidona).
- **Polimorfismos e Interacciones de CYP2D6:**
  - En metabolizadores lentos de CYP2D6 o en presencia de inhibidores potentes (Paroxetina, Fluoxetina, Bupropión), la semivida de la risperidona se prolonga de 3 a más de 20 horas, incrementando el riesgo de parkinsonismo secundario si no se reduce la dosis.

## 4. Dosificación, Rangos y Poblaciones Especiales
- **Dosis Habitual:** Iniciar con 1 a 2 mg/día; titular en escalones de 1 mg cada 24-48 horas hasta una dosis óptima de 2 a 4 mg/día (dosis mayores a 6 mg raramente aportan mayor beneficio antipsicótico y disparan los efectos adversos motores).
- **Poblaciones Especiales:**
  - *Pediatría (Autismo y Conducta Disruptiva):* Dosis muy bajas (0.25 a 1 mg/día).
  - *Adultos Mayores / Demencia:* Dosis reducidas (0.5 a 1.5 mg/día). Advertencia de aumento de mortalidad por eventos cerebrovasculares (ictus) en demencia.

## 5. Eventos Adversos: Frecuentes vs. Graves
- **Hiperprolactinemia Marcada:** Ocurre en la gran mayoría de los pacientes por carecer de barrera hematoencefálica protectora a nivel de la eminencia media hipofisaria. Genera galactorrea, ginecomastia dolorosa, amenorrea completa y pérdida de densidad mineral ósea a largo plazo.
- **Síntomas Extrapiramidales Dosis-Dependientes:** Acatisia, temblor, distonía aguda de cuello/mandíbula a dosis elevadas.
- **Aumento Ponderal:** Moderado, junto con sedación e hipotensión ortostática inicial.

## 6. Guía de Comunicación Clínica (Pregunta y Respuesta)
- **Pregunta:** "¿Qué hacer si una mujer joven tratada con Risperdal desarrolla amenorrea y secreción láctea por el pezón?"
- **Respuesta:** "Se debe solicitar prolactina sérica para confirmar hiperprolactinemia inducida por risperidona. La estrategia clínica consiste en reducir la dosis, rotar a un antipsicótico ahorrador de prolactina como aripiprazol (agonista parcial D2) o añadir dosis bajas de aripiprazol como coadyuvante para normalizar la prolactina."

## 7. Referencias y Validación
- Dra. Gabriela Jufe: *Psicofarmacología Práctica*, Capítulo de Antipsicóticos Atípicos.
- Dr. Stephen Stahl: *Prescriber's Guide*, Monografía de Risperidona.
"""
    )

    write_doc(
        "01_monografias_farmacologicas/haloperidol.md",
        """---
id: "MONO-HALOPERIDOL-008"
tipo: "monografia_farmacologica"
farmaco: "Haloperidol"
atc_codigo: "N05AD01"
atc_grupo: "Psicolépticos / Antipsicóticos Típicos (Primera Generación / Butirofenonas)"
enzimas_cyp: ["CYP3A4 (sustrato principal)", "CYP2D6", "Glucuronidación"]
marcas_principales: ["Haldol (Janssen)", "Halopidol (Neuval)", "Aloperidin"]
fuentes_primarias: ["Jufe - Psicofarmacología Práctica", "Stahl - Prescriber's Guide"]
nivel_riesgo: "Alto (riesgo extrapiramidal severo, discinesia tardía y cascada con biperideno)"
palabras_clave: ["haloperidol", "haldol", "antipsicotico tipico", "bloqueo d2", "parkinsonismo", "distonia aguda", "biperideno", "cascada de prescripcion", "sindrome neuroleptico maligno"]
---

# Ficha Clínica y Conversacional: Haloperidol (N05AD01)

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "El haloperidol es el antipsicótico clásico o de primera generación más representativo, utilizado en urgencias psiquiátricas para el control de la agitación psicomotora aguda, cuadros psicóticos y delirium. Su dosis diaria en mantenimiento va de 2 a 10 miligramos. Ejerce un bloqueo potente e indiscriminado sobre los receptores dopaminérgicos D2 en todo el cerebro: al bloquear la vía mesolímbica calma el delirio, pero al bloquear la vía nigroestriatal desencadena con altísima frecuencia distonías agudas, rigidez y parkinsonismo. Este cuadro suele motivar erróneamente la prescripción de anticolinérgicos como el biperideno, generando una cascada de prescripción con riesgo de deterioro cognitivo y discinesia tardía. Su marca histórica es Haldol."

## 2. Perfil Neurofarmacológico y Mecanismo de Acción
Stahl y Jufe exponen que el haloperidol es un antagonista D2 de altísima afinidad y lenta disociación. Carece de propiedades antagonistas 5-HT2A significativas y tiene muy baja afinidad por receptores muscarínicos colinérgicos (M1) e histaminérgicos (H1). Esta falta de acción anticolinérgica intrínseca rompe el equilibrio dopamina-acetilcolina en los ganglios basales: la brusca caída funcional de dopamina libera una hiperactividad colinérgica no compensada, provocando espasmos musculares severos (distonías agudas) y rigidez en rueda dentada.

## 3. Farmacocinética e Interacciones
- **Metabolismo Hepático:** Es depurado por CYP3A4, CYP2D6 y glucuronidación. Inductores como carbamazepina o fenitoína reducen su concentración a la mitad; inhibidores de 2D6 o 3A4 pueden elevarla.
- **Riesgo Cardíaco por Vía Intravenosa:** La administración parenteral (especialmente endovenosa) exige monitorización electrocardiográfica estricta debido al riesgo de prolongación del intervalo QTc y taquicardia ventricular helicoidal (*Torsades de Pointes*).

## 4. Dosificación y Vías de Administración
- **Agitación Aguda / Emergencias:** 2.5 a 5 mg vía intramuscular (IM) o gotas orales, repetible según pauta médica.
- **Mantenimiento Crónico:** 2 a 10 mg/día por vía oral. En formulación decanoato (depot de liberación prolongada): 50 a 200 mg IM cada 4 semanas.

## 5. Eventos Adversos: Frecuentes vs. Graves
- **Trastornos Extrapiramidales (EPS) Frecuentes:**
  - *Distonía Aguda (primeras 48 h):* Crisis oculógiras, tortícolis espasmódica, opistótonos y laringoespasmo (urgencia médica).
  - *Parkinsonismo Secundario (semanas 1 a 4):* Rigidez plástica, bradicinesia, temblor de reposo, marcha festinante y facies inexpresiva.
  - *Acatisia:* Inquietud motora subjetiva angustiante con incapacidad para permanecer sentado.
- **Efectos Graves / Tardíos:**
  - *Discinesia Tardía:* Movimientos coreoatetoides involuntarios bucolinguales o de extremidades tras meses o años de uso, frecuentemente irreversible.
  - *Síndrome Neuroléptico Maligno (SNM):* Rigidez muscular extrema en "tubo de plomo", hipertermia maligna, inestabilidad autonómica y elevación masiva de creatina quinasa (CPK).

## 6. Guía de Comunicación Clínica (La Cascada de Prescripción)
- **Pregunta:** "Un paciente que toma 5 mg de haloperidol presenta temblor y rigidez en rueda dentada. ¿Debe recetársele biperideno?"
- **Respuesta:** "No debería ser la primera opción. Añadir biperideno inicia una cascada de prescripción, sumando toxicidad anticolinérgica, pérdida de memoria, estreñimiento severo y mayor riesgo futuro de discinesia tardía. La mejor práctica clínica consiste en reducir la dosis de haloperidol o rotar a un antipsicótico de segunda generación con bajo perfil extrapiramidal, como quetiapina u olanzapina."

## 7. Referencias y Validación
- Dra. Gabriela Jufe: *Psicofarmacología Práctica*, Capítulo de Antipsicóticos Típicos.
- Dr. Stephen Stahl: *Prescriber's Guide*, Monografía de Haloperidol.
"""
    )

    write_doc(
        "01_monografias_farmacologicas/lamotrigina.md",
        """---
id: "MONO-LAMOTRIGINA-009"
tipo: "monografia_farmacologica"
farmaco: "Lamotrigina"
atc_codigo: "N03AX09"
atc_grupo: "Antiepilépticos / Estabilizadores del Ánimo"
enzimas_cyp: ["Glucuronidación hepática por UGT1A4 (no dependiente de CYP450)"]
marcas_principales: ["Lamictal (GSK)", "Tradox", "Labileno", "Lamicstar"]
fuentes_primarias: ["Jufe - Psicofarmacología Práctica", "CANMAT/ISBD 2018 Guidelines", "Stahl"]
nivel_riesgo: "Alto (riesgo de Síndrome de Stevens-Johnson ante titulación rápida o combinación con valproato)"
palabras_clave: ["lamotrigina", "lamictal", "estabilizador del animo", "depresion bipolar", "acido valproico", "stevens johnson", "exantema", "ugt1a4"]
---

# Ficha Clínica y Conversacional: Lamotrigina (N03AX09)

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "La lamotrigina es un fármaco estabilizador del ánimo y antiepiléptico de primera línea para la prevención y el tratamiento de las fases depresivas del trastorno bipolar. A diferencia del litio o el valproato, no tiene eficacia para tratar la manía aguda, pero destaca por su excelente tolerancia metabólica, ya que no engorda ni seda. Su metabolismo depende de la glucuronidación hepática y no del sistema citocromo P450. Su aspecto de máxima seguridad clínica es el riesgo de exantemas cutáneos graves y potencialmente mortales, como el síndrome de Stevens-Johnson. Por esta razón, exige una titulación extraordinariamente lenta comenzando con 25 miligramos al día. Si se combina con ácido valproico, este duplica la vida media de la lamotrigina, obligando a reducir la dosis inicial y de mantenimiento a más de la mitad. Su marca comercial original es Lamictal de GSK."

## 2. Perfil Neurofarmacológico y Mecanismo de Acción
Stahl y CANMAT explican que la lamotrigina bloquea de forma voltaje-dependiente los canales de sodio sensibles a voltaje (NaV) en las neuronas presinápticas. Esta acción estabiliza la membrana neuronal y suprime la liberación excesiva del neurotransmisor excitatorio glutamato y aspartato, particularmente en circuitos córtico-límbicos hiperactivos en el trastorno bipolar, ejerciendo una acción estabilizadora orientada hacia el polo depresivo.

## 3. Farmacocinética e Interacción Vital con Ácido Valproico
- **Vía Metabólica:** Se depura por glucuronidación mediada por la enzima uridina 5'-difosfo-glucuronosiltransferasa (UGT1A4).
- **Interacción con Valproato (Inhibición Crítica):** El ácido valproico o divalproato de sodio inhibe potentemente la UGT1A4. Si se añade lamotrigina a un paciente que toma valproato, la vida media de la lamotrigina pasa de 25 a más de 60 horas. Si no se reduce la dosis a la mitad o menos, el riesgo de necrólisis epidérmica tóxica se multiplica drásticamente.
- **Interacción con Inductores (Carbamazepina / Anticonceptivos Orales):** La carbamazepina y los estrógenos contenidos en los anticonceptivos orales inducen la glucuronidación, reduciendo los niveles de lamotrigina a la mitad y provocando fallos terapéuticos.

## 4. Esquema de Titulación Mandatorio (Regla de Seguridad)
Para evitar el Síndrome de Stevens-Johnson, debe respetarse el siguiente cronograma:
- **Monoterapia (sin fármacos inductores ni inhibidores):**
  - Semanas 1 y 2: 25 mg/día.
  - Semanas 3 y 4: 50 mg/día.
  - Semana 5: 100 mg/día.
  - Semana 6 en adelante: Dosis terapéutica objetivo de 100 a 200 mg/día (hasta 400 mg/día).
- **En combinación con Ácido Valproico:**
  - Semanas 1 y 2: 25 mg **en días alternos** (cada 48 horas) o 12.5 mg/día.
  - Semanas 3 y 4: 25 mg/día.
  - Semanas 5 y 6: 50 mg/día.
  - Dosis de mantenimiento habitual: 100 mg/día (máximo 150 mg/día).

## 5. Eventos Adversos: Frecuentes vs. Erupción Cutánea Grave
- **Comunes y Benignos:** Cefalea leve, mareo, náuseas, visión borrosa diplópica y exantema cutáneo maculopapular benigno no confluente (ocurre en el 10% y remite espontáneamente).
- **Alerta de Síndrome de Stevens-Johnson / NET:**
  - Incidencia: Aproximadamente 1 de cada 1000 adultos (más frecuente en niños).
  - Signos de Alarma: Fiebre alta, afectación de mucosas (labios, boca, ojos, genitales), ampollas en la piel o dolor cutáneo intenso.
  - **Conducta:** Suspensión inmediata y permanente; nunca reintroducir el fármaco tras un episodio de Stevens-Johnson.

## 6. Guía de Comunicación Clínica (Pregunta y Respuesta)
- **Pregunta:** "Un paciente estable con Lamictal 100 mg/día olvida tomar la medicación durante 6 días seguidos por viaje. ¿Puede retomar sus 100 mg al regresar?"
- **Respuesta:** "Bajo ninguna circunstancia. Tras más de 5 vidas medias de interrupción (unos 5 días), el organismo ha eliminado el fármaco y se ha perdido la tolerancia cutánea. Reiniciar directamente con 100 mg expone al paciente a un riesgo inaceptable de Stevens-Johnson. Se debe reiniciar la titulación desde 25 mg/día siguiendo el protocolo."

## 7. Referencias y Validación
- Guías CANMAT / ISBD 2018: Primera línea en depresión bipolar (mantenimiento y prevención).
- Dra. Gabriela Jufe: *Psicofarmacología Práctica*, Capítulo de Antiepilépticos.
- Dr. Stephen Stahl: *Prescriber's Guide*, Monografía de Lamotrigina.
"""
    )

    # =========================================================================
    # 02. INTERACCIONES FARMACOLÓGICAS Y CITOCROMOS CYP450 (5 Archivos)
    # =========================================================================

    write_doc(
        "02_interacciones_farmacologicas_cyp450/interacciones_cyp2d6_inhibidores_sustratos.md",
        """---
id: "INTER-CYP2D6-001"
tipo: "interaccion_cyp450"
titulo: "Interacciones Críticas en el Citocromo CYP2D6: Inhibidores Potentes vs. Sustratos Psiquiátricos"
enzimas_cyp: ["CYP2D6"]
farmacos_involucrados: ["Paroxetina", "Fluoxetina", "Bupropión", "Risperidona", "Aripiprazol", "Haloperidol", "Tamoxifeno", "Tramadol"]
fuentes_primarias: ["Stahl - Prescriber's Guide", "Jufe - Psicofarmacología Práctica"]
nivel_riesgo: "Alto a Crítico"
palabras_clave: ["cyp2d6", "inhibicion enzimatica", "paroxetina", "fluoxetina", "bupropion", "risperidona", "profarmacos", "tamoxifeno", "toxicidad"]
---

# Interacciones en el Citocromo CYP2D6: Inhibidores Potentes y Sustratos Clínicos

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "El citocromo hepático CYP2D6 metaboliza cerca del veinticinco por ciento de los psicofármacos utilizados en la práctica clínica diaria, incluyendo antipsicóticos como la risperidona y el aripiprazol, y antidepresivos tricíclicos. Tres antidepresivos son inhibidores potentes de esta enzima: la paroxetina, la fluoxetina y el bupropión. Al coadministrar cualquiera de estos tres inhibidores con un sustrato de CYP2D6, se bloquea su eliminación hepática, multiplicando sus concentraciones en sangre y provocando toxicidad motora grave, arritmias o sedación comatosa. Asimismo, el bloqueo de CYP2D6 impide la activación terapéutica de profármacos esenciales como el tamoxifeno para el cáncer de mama o analgésicos como el tramadol y la codeína, dejándolos sin efecto."

## 2. Fisiopatología de la Vía Enzimática CYP2D6
El citocromo CYP2D6 no es inducible de forma clínicamente relevante, pero presenta un polimorfismo genético muy marcado en la población general (metabolizadores ultrarrápidos, extensos, intermedios y lentos). Cuando un paciente recibe un inhibidor potente como paroxetina o fluoxetina, su fenotipo funcional se convierte en el de un **metabolizador lento inducido por fármacos** (*fenoconversión*).

## 3. Matriz de Fármacos Involucrados
- **Inhibidores Potentes de CYP2D6 (Bloqueadores de la vía):**
  - Paroxetina (el inhibidor más potente por inactivación dependiente de mecanismo).
  - Fluoxetina y su metabolito norfluoxetina.
  - Bupropión y sus metabolitos activos (hidroxibupropión).
- **Sustratos Clínicos Críticos (Afectados por elevación plasmática):**
  - *Antipsicóticos:* Risperidona, Aripiprazol, Haloperidol, Olanzapina (vía secundaria).
  - *Antidepresivos Tricíclicos:* Nortriptilina, Desipramina, Imipramina (riesgo letal de arritmias ventriculares por prolongación de QRS/QTc).
  - *Betabloqueantes Cardiovasculares:* Metoprolol, Propranolol, Carvedilol (riesgo de bradicardia severa e hipotensión).
- **Profármacos que Dejan de Funcionar (Pérdida de eficacia terapéutica):**
  - *Tamoxifeno:* Requiere CYP2D6 para convertirse en endoxifeno; la paroxetina o fluoxetina reducen la supervivencia en cáncer de mama.
  - *Codeína y Tramadol:* Requieren CYP2D6 para bioactivarse a morfina y O-desmetiltramadol; su inhibición produce fracaso analgésico.

## 4. Protocolo de Manejo Clínico y Desprescripción
1. **Regla de Reducción:** Si es imprescindible combinar risperidona o aripiprazol con fluoxetina, paroxetina o bupropión, la dosis del antipsicótico debe reducirse de entrada entre un 50% y un 75%.
2. **Alternativas de Prescripción Segura:** Si el paciente requiere un antidepresivo pero ya toma sustratos sensibles de CYP2D6, sustituir por **Sertralina, Citalopram, Escitalopram o Venlafaxina**, los cuales presentan una inhibición de CYP2D6 despreciable a dosis habituales.
"""
    )

    write_doc(
        "02_interacciones_farmacologicas_cyp450/interacciones_cyp1a2_tabaco_clozapina.md",
        """---
id: "INTER-CYP1A2-002"
tipo: "interaccion_cyp450"
titulo: "Interacción entre Citocromo CYP1A2, Tabaquismo y Clozapina / Olanzapina"
enzimas_cyp: ["CYP1A2"]
farmacos_involucrados: ["Clozapina", "Olanzapina", "Humo de Tabaco", "Fluvoxamina", "Ciprofloxacino"]
fuentes_primarias: ["Stahl - Prescriber's Guide", "Jufe - Psicofarmacología Práctica"]
nivel_riesgo: "Crítico"
palabras_clave: ["cyp1a2", "tabaquismo", "clozapina", "olanzapina", "fluvoxamina", "ciprofloxacino", "induccion enzimatica", "toxicidad por clozapina"]
---

# Citocromo CYP1A2: Tabaquismo, Clozapina, Olanzapina y Fármacos Inhibidores

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "El citocromo hepático CYP1A2 es la vía metabólica principal de antipsicóticos clave como la clozapina y la olanzapina. Esta enzima tiene una particularidad de máxima relevancia clínica: es inducida potentemente por los hidrocarburos policíclicos del humo del tabaco, no por la nicotina. Por ello, los fumadores consumen sus niveles de clozapina un cincuenta por ciento más rápido. Si un paciente fumador ingresa al hospital o deja el tabaco repentinamente, sus niveles plasmáticos de clozapina se disparan en pocos días, pudiendo sufrir toxicidad extrema con convulsiones, neumonía por aspiración o coma. Asimismo, antibióticos como el ciprofloxacino o el antidepresivo fluvoxamina inhiben potentemente CYP1A2, estando formalmente contraindicados junto a clozapina."

## 2. Mecanismo Fisiopatológico: Humo vs. Nicotina
Es fundamental distinguir que la inducción de CYP1A2 es producida por los **hidrocarburos aromáticos policíclicos** generados en la combustión de la hoja de tabaco, y **no por la nicotina**.
- Por tanto, el uso de parches, chicles de nicotina o cigarrillos electrónicos (vapeo) **NO induce** el citocromo CYP1A2.
- Cuando un fumador pasa a vapear o a parches de nicotina, a efectos farmacocinéticos hepáticos es exactamente equivalente a haber dejado de fumar por completo: la actividad de CYP1A2 disminuye a su nivel basal en un plazo de 3 a 7 días.

## 3. Escenarios de Riesgo Clínico Extremo
1. **El Paciente que Deja de Fumar Bruscamente:**
   - La eliminación de clozapina u olanzapina se enlentece bruscamente.
   - Las concentraciones en sangre de clozapina pueden aumentar entre un 50% y un 100%.
   - Consecuencias: Sedación profunda, mioclonías, ataxia, crisis convulsivas dosis-dependientes y riesgo de broncoaspiración por sialorrea.
   - *Conducta:* Reducir la dosis de clozapina un 30-40% inmediatamente al dejar el tabaco y monitorizar niveles plasmáticos.
2. **El Paciente Estable que Empieza a Fumar o Recae:**
   - La inducción enzimática acelera la degradación del fármaco.
   - Consecuencias: Pérdida de eficacia antipsicótica y recaída delirante o alucinatoria grave.
3. **Coadministración de Inhibidores Potentes (Fluvoxamina o Ciprofloxacino):**
   - Fluvoxamina es el inhibidor más potente de CYP1A2. Eleva la clozapina entre 5 y 10 veces. La asociación sin monitorización extrema está formalmente proscrita.

## 4. Protocolo de Seguridad en Unidades de Hospitalización
- Todo ingreso psiquiátrico en régimen libre de humo exige interrogar el hábito tabáquico previo (número de cigarrillos diarios).
- En pacientes que toman clozapina, programar sistemáticamente una reducción del 30% en la dosis domiciliaria durante los primeros 3 a 5 días de internamiento libre de humo.
"""
    )

    write_doc(
        "02_interacciones_farmacologicas_cyp450/interacciones_lamotrigina_acido_valproico.md",
        """---
id: "INTER-LAM-VAL-003"
tipo: "interaccion_cyp450"
titulo: "Interacción Farmacocinética Sinérgica: Lamotrigina y Ácido Valproico (Glucuronidación UGT1A4)"
enzimas_cyp: ["UGT1A4 (Glucuronidación Hepática)"]
farmacos_involucrados: ["Lamotrigina", "Ácido Valproico / Divalproato de Sodio"]
fuentes_primarias: ["CANMAT/ISBD 2018", "Jufe - Psicofarmacología Práctica", "Stahl"]
nivel_riesgo: "Crítico (Riesgo letal de Síndrome de Stevens-Johnson / NET)"
palabras_clave: ["lamotrigina", "valproato", "divalproato", "depakote", "lamictal", "stevens johnson", "ugt1a4", "titulacion lenta"]
---

# Interacción Farmacocinética Vital: Lamotrigina y Ácido Valproico

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "La combinación de lamotrigina con ácido valproico es una estrategia clínica muy potente para el trastorno bipolar complejo, pero entraña un peligro farmacocinético vital. El ácido valproico inhibe intensamente la enzima hepática responsable de eliminar la lamotrigina, duplicando su vida media en el cuerpo de veinticinco a más de sesenta horas. Si la lamotrigina se inicia a las dosis habituales en un paciente que ya recibe valproato, sus niveles se acumulan de forma explosiva, multiplicando el riesgo de desencadenar el síndrome de Stevens-Johnson o necrólisis epidérmica tóxica, una reacción dermatológica potencialmente mortal. Para combinar ambos fármacos con total seguridad, es mandatorio iniciar la lamotrigina a dosis ultrabajas de veinticinco miligramos en días alternos y titularla con extrema lentitud."

## 2. Mecanismo Fisiopatológico Molecular
La lamotrigina no utiliza la vía de los citocromos P450, sino que se depura por conjugación con ácido glucurónico a través de la enzima UDP-glucuronosiltransferasa (especialmente la isoforma UGT1A4).
El ácido valproico (y sus sales como el divalproato de sodio) es un inhibidor competitivo directo y potente de la enzima UGT1A4.
- **Resultado cinético:** El aclaramiento renal de la lamotrigina disminuye en más de un 50%. La vida media se prolonga de 25-30 horas hasta 60-70 horas.

## 3. Pauta Mandatoria de Titulación en Presencia de Valproato
Cuando un paciente ya está en tratamiento con ácido valproico (en cualquier dosis) y se prescribe lamotrigina:

| Período | Pauta en Monoterapia | **Pauta Mandatoria con Ácido Valproico** |
| :--- | :--- | :--- |
| **Semanas 1 y 2** | 25 mg diarios | **25 mg en días alternos (un día sí, un día no)** |
| **Semanas 3 y 4** | 50 mg diarios | **25 mg diarios** |
| **Semanas 5 y 6** | 100 mg diarios | **50 mg diarios** |
| **Mantenimiento** | 100 a 200 mg diarios | **100 mg diarios (máximo 150 mg)** |

## 4. Escenario Inverso: ¿Qué ocurre si se suspende el Valproato?
Si un paciente estable con ambos fármacos interrumpe el ácido valproico:
- La inhibición de la glucuronidación desaparece en 48 horas.
- La vida media de la lamotrigina se reduce a la mitad.
- Los niveles de lamotrigina caen en picado, provocando recaídas depresivas inmediatas si no se duplica la dosis de lamotrigina.

## 5. Criterios de Alarma y Derivación Urgente
Cualquier erupción en la piel, especialmente si se acompaña de fiebre, adenopatías, úlceras en labios o aftas orales, exige la retirada inmediata del tratamiento y acudir a urgencias médicas hospitalarias.
"""
    )

    write_doc(
        "02_interacciones_farmacologicas_cyp450/interacciones_litio_aines_diureticos_ieca.md",
        """---
id: "INTER-LIT-RENAL-004"
tipo: "interaccion_cyp450"
titulo: "Interacciones Renales Críticas del Carbonato de Litio: AINEs, Tiazidas e Inhibidores del SRAA"
enzimas_cyp: ["Ninguna (Manejo hemodinámico y tubular renal)"]
farmacos_involucrados: ["Carbonato de Litio", "Ibuprofeno", "Naproxeno", "Hidroclorotiazida", "Enalapril", "Losartán"]
fuentes_primarias: ["Jufe - Psicofarmacología Práctica", "CANMAT Guidelines", "Stahl"]
nivel_riesgo: "Crítico (Intoxicación por litemia tóxica y fallo renal agudo)"
palabras_clave: ["litio", "litemia", "aines", "ibuprofeno", "tiazidas", "hidroclorotiazida", "ieca", "enalapril", "intoxicacion renal"]
---

# Toxicidad Renal del Litio: Interacciones con AINEs, Diuréticos e IECA/ARA-II

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "El carbonato de litio no sufre metabolismo en el hígado; se filtra en los glomérulos y se reabsorbe en los túbulos renales compitiendo con el sodio. Por ello, cualquier medicamento que altere la función del riñón o disminuya el sodio puede provocar una acumulación letal de litio. Los antiinflamatorios comunes como el ibuprofeno o el naproxeno, los diuréticos tiazídicos como la hidroclorotiazida y los antihipertensivos IECA como el enalapril reducen drásticamente la eliminación del litio, elevando su concentración en sangre hacia niveles tóxicos. La intoxicación por litio causa temblor grueso, vómitos, pérdida de equilibrio, convulsiones y daño renal irreversible. Ante cuadros de dolor en pacientes tratados con litio, el analgésico de elección siempre debe ser el paracetamol."

## 2. Mecanismo Fisiológico de la Excreción Renal de Litio
El ion litio (Li+) es tratado fisiológicamente por el riñón de manera idéntica al ion sodio (Na+). Se reabsorbe en un 70-80% en el túbulo contorneado proximal. Todo factor o fármaco que genere depleción de sodio o reduzca la perfusión renal induce a las células tubulares a reabsorber con avidez tanto sodio como litio, reteniendo este último en el torrente sanguíneo.

## 3. Las Tres Familias Farmacológicas Peligrosas
1. **Antiinflamatorios No Esteroideos (AINEs):**
   - *Fármacos:* Ibuprofeno, naproxeno, ketoprofeno, diclofenaco, celecoxib.
   - *Mecanismo:* Bloquean la síntesis de prostaglandinas renales vasodilatadoras, disminuyendo el flujo plasmático renal y la filtración glomerular.
   - *Impacto:* Elevan la litemia entre un 30% y un 60% en pocos días.
2. **Diuréticos Tiazídicos (La interacción más peligrosa):**
   - *Fármacos:* Hidroclorotiazida, clortalidona, indapamida.
   - *Mecanismo:* Bloquean la reabsorción de sodio en el túbulo distal, forzando una depleción salina que estimula una hiperreabsorción compensatoria proximal masiva de litio.
   - *Impacto:* Reducen el aclaramiento de litio en un 50%.
3. **Inhibidores de la ECA y Antagonistas del Receptor de Angiotensina (ARA-II):**
   - *Fármacos:* Enalapril, ramipril, lisinopril, losartán, valsartán.
   - *Mecanismo:* Vasodilatación preferente de la arteriola eferente renal, reduciendo la presión intraglomerular.
   - *Impacto:* Incrementos insidiosos y tardíos de la litemia (a menudo a las 2 a 4 semanas de iniciado el antihipertensivo).

## 4. Opciones Terapéuticas Seguras
- **Para el Dolor o Inflamación:** Paracetamol (acetaminofén) a dosis habituales de 500 a 1000 mg cada 8 horas, o aspirina en dosis analgésicas puntuales. Ninguno altera la hemodinámica tubular del litio.
- **Para la Hipertensión Arterial:** Los betabloqueantes o los antagonistas de los canales de calcio dihidropiridínicos (como el amlodipino) no interfieren con la litemia.
"""
    )

    write_doc(
        "02_interacciones_farmacologicas_cyp450/matriz_clinica_cyp450_psicofarmacos.md",
        """---
id: "INTER-MATRIZ-CYP-005"
tipo: "interaccion_cyp450"
titulo: "Matriz Clínica de Referencia: Sistema Citocromo P450 en Psicofarmacología"
enzimas_cyp: ["CYP1A2", "CYP2D6", "CYP3A4", "CYP2C19", "CYP2B6"]
farmacos_involucrados: ["Paroxetina", "Fluoxetina", "Bupropión", "Clozapina", "Olanzapina", "Sertralina", "Risperidona", "Carbamazepina", "Valproato"]
fuentes_primarias: ["Stahl - Prescriber's Guide", "Jufe - Psicofarmacología Práctica"]
nivel_riesgo: "Alto"
palabras_clave: ["cyp450", "sustratos", "inhibidores", "inductores", "polifarmacia", "interacciones", "psiquiatria"]
---

# Matriz Clínica de Referencia del Sistema Citocromo P450 en Psiquiatría

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "El sistema del citocromo P450 hepático es el motor central que determina la concentración de los psicofármacos en el organismo. En psiquiatría destacan cinco enzimas principales. El citocromo CYP2D6 metaboliza la mayoría de los antidepresivos y antipsicóticos, y es fuertemente inhibido por paroxetina, fluoxetina y bupropión. El citocromo CYP1A2 depura la clozapina y la olanzapina, siendo inducido por el humo del tabaco e inhibido por la fluvoxamina. El citocromo CYP3A4 metaboliza benzodiacepinas y estabilizadores, y sufre una inducción masiva por la carbamazepina, reduciendo a la mitad la eficacia de los tratamientos combinados. Conocer estas vías permite anticipar toxicidades y evitar fracasos terapéuticos en polifarmacia."

## 2. Mapa Enzimático por Familia de Citocromos

### Citocromo CYP1A2
- **Sustratos Principales:** Clozapina, Olanzapina, Duloxetina, Agomelatina.
- **Inductores Potentes:** Hidrocarburos del humo del tabaco, Carbamazepina, Carne asada al carbón.
- **Inhibidores Potentes:** Fluvoxamina, Ciprofloxacino.

### Citocromo CYP2D6
- **Sustratos Principales:** Risperidona, Aripiprazol, Haloperidol, Antidepresivos Tricíclicos (Nortriptilina, Imipramina), Venlafaxina, Metoprolol, Atomoxetina.
- **Inductores:** Vía no inducible de forma clínicamente relevante.
- **Inhibidores Potentes:** Paroxetina, Fluoxetina, Bupropión, Quinidina.
- **Inhibidores Moderados:** Sertralina (a dosis altas > 150 mg/día), Duloxetina.

### Citocromo CYP3A4
- **Sustratos Principales:** Quetiapina, Aripiprazol, Alprazolam, Clonazepam, Midazolam, Lurasidona, Buspirona.
- **Inductores Potentes:** Carbamazepina, Fenitoína, Hierba de San Juan (*Hypericum perforatum*), Rifampicina.
- **Inhibidores Potentes:** Ketoconazol, Itraconazol, Claritromicina, Zumo de Pomelo (*Grapefruit*).

### Citocromo CYP2B6
- **Sustratos Principales:** Bupropión, Metadona, Ketamina.
- **Inductores:** Carbamazepina, Fenobarbital.
- **Inhibidores:** Ticlopidina, Clopidogrel.

## 3. Reglas de Oro en Prescripción Psiquiátrica
1. **Regla de la Carbamazepina (El autoinductor universal):** La carbamazepina induce su propio metabolismo y el de CYP3A4 y CYP1A2. Al añadir carbamazepina a casi cualquier antipsicótico o antidepresivo, sus niveles caen un 50%.
2. **Regla del Fármaco Limpio:** Si un paciente polimedicado con enfermedad cardiovascular o metabólica requiere un antidepresivo, los fármacos con menor perfil de interferencia sobre citocromos son **Sertralina, Escitalopram y Citalopram**.
"""
    )

    # =========================================================================
    # 03. CASCADAS DE PRESCRIPCIÓN Y DESPRESCRIPCIÓN (3 Archivos)
    # =========================================================================

    write_doc(
        "03_cascadas_prescripcion_y_desprescripcion/cascada_haloperidol_biperideno_parkinsonismo.md",
        """---
id: "CASC-HAL-BIP-001"
tipo: "cascada_prescripcion"
titulo: "Cascada de Prescripción Clásica: Haloperidol a Biperideno por Parkinsonismo Secundario"
farmacos_involucrados: ["Haloperidol", "Biperideno", "Trihexifenidilo"]
fuentes_primarias: ["Jufe - Psicofarmacología Práctica", "Stahl - Prescriber's Guide"]
nivel_riesgo: "Alto"
palabras_clave: ["cascada de prescripcion", "haloperidol", "biperideno", "parkinsonismo secundario", "carga anticolinergica", "discinesia tardia", "desprescripcion"]
---

# Cascada de Prescripción: Haloperidol inducido y Biperideno

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "Una cascada de prescripción ocurre cuando el efecto adverso de un primer fármaco es diagnosticado erróneamente como una nueva enfermedad, recetándose un segundo medicamento para tratarlo. El ejemplo clásico en psiquiatría es el uso de haloperidol: debido a su potente bloqueo de dopamina, produce rigidez, temblor y lentitud motora en rueda dentada. En lugar de disminuir la dosis o cambiar a un antipsicótico atípico, con frecuencia se receta biperideno. Esta cascada encubre el problema y expone al paciente a una peligrosa carga anticolinérgica con pérdidas de memoria, confusión, estreñimiento severo, retención urinaria y un mayor riesgo de desarrollar discinesia tardía irreversible. La desprescripción razonada exige reducir el haloperidol y retirar progresivamente el anticolinérgico."

## 2. Anatomía de la Cascada Clínica
1. **Fármaco Inicial:** Haloperidol (Antipsicótico típico de alta potencia).
2. **Efecto Adverso Generado:** Parkinsonismo secundario farmacológico (bloqueo > 80% de receptores D2 en la vía dopaminérgica nigroestriatal).
3. **Interpretación Errónea / Conducta Reactiva:** Se asume que el temblor o la rigidez requieren farmacoterapia obligatoria, prescribiendo **Biperideno** (Akineton) o Trihexifenidilo.
4. **Daño Iatrogénico Acumulativo:**
   - *Carga Anticolinérgica:* Deterioro neurocognitivo, fallos ejecutivos y de memoria episódica, sedación, midriasis con riesgo de glaucoma de ángulo cerrado, retención aguda de orina e íleo paralítico.
   - *Riesgo Psiquiátrico:* El biperideno genera potencial de abuso, euforia y descompensaciones psicóticas con alucinaciones tóxicas en adultos mayores.
   - *Empeoramiento Motor Tardío:* Los anticolinérgicos aumentan la sensibilidad de los receptores D2 estriatales desnervados, acelerando la aparición de **discinesia tardía**.

## 3. Algoritmo de Desprescripción y Resolución
```mermaid
flowchart TD
    A["Paciente con Haloperidol presenta Temblor/Rigidez"] --> B["¿Identificado como efecto adverso farmacológico?"]
    B -->|Sí| C["Paso 1: ¿Se puede reducir la dosis de Haloperidol?"]
    C -->|Sí| D["Reducir dosis a < 4 mg/día y evaluar 1 semana"]
    C -->|No / Recaída| E["Paso 2: Rotar a Antipsicótico Atípico\n(Quetiapina, Olanzapina o Aripiprazol)"]
    D --> F["Paso 3: Retirada gradual del Biperideno\n(Reducir 1-2 mg cada semana)"]
    E --> F
    F --> G["Resolución exitosa sin carga anticolinérgica"]
```

## 4. Guía de Interrogatorio Clínico Verbal
- "Doctor, el temblor y la lentitud que presenta este paciente comenzaron dos semanas después de subir el haloperidol. Antes de añadir biperideno, ¿hemos considerado reducir la dosis o rotar a quetiapina para proteger su rendimiento cognitivo y evitar una discinesia tardía?"
"""
    )

    write_doc(
        "03_cascadas_prescripcion_y_desprescripcion/cascada_antipsicoticos_sindrome_metabolico.md",
        """---
id: "CASC-METAB-002"
tipo: "cascada_prescripcion"
titulo: "Cascada Metabólica por Antipsicóticos Atípicos: De Olanzapina/Clozapina a Polifarmacia Cardiometabólica"
farmacos_involucrados: ["Olanzapina", "Clozapina", "Quetiapina", "Metformina", "Estatinas", "Antihipertensivos"]
fuentes_primarias: ["Jufe - Psicofarmacología Práctica", "Stahl"]
nivel_riesgo: "Alto"
palabras_clave: ["sindrome metabolico", "olanzapina", "clozapina", "aumento de peso", "resistencia a insulina", "metformina", "desprescripcion"]
---

# Cascada de Prescripción Metabólica por Antipsicóticos de Segunda Generación

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "Los antipsicóticos atípicos con alta afinidad histaminérgica H1 y serotoninérgica 5-HT2C, especialmente la olanzapina y la clozapina, inducen un aumento de peso drástico, hiperfagia y resistencia periférica a la insulina. Con frecuencia, este deterioro metabólico desencadena una cascada en la que se incorporan metformina, estatinas para la dislipidemia y fármacos antihipertensivos, transformando un tratamiento psiquiátrico en un cuadro de polifarmacia cardiometabólica compleja. El abordaje proactivo exige monitorizar peso, glucemia y lípidos desde el primer mes, interviniendo tempranamente con ajustes dietéticos, cambio a antipsicóticos metabólicamente neutros como aripiprazol o lurasidona, o coadyuvancia precoz antes de consolidar el daño vascular."

## 2. Mecanismo Fisiopatológico de la Ganancia Ponderal
El bloqueo simultáneo de receptores H1 histaminérgicos y 5-HT2C en los núcleos hipotalámicos inhibe la señalización de saciedad mediada por la leptina, generando hiperfagia compulsiva por carbohidratos. Adicionalmente, alteran la secreción pancreática de insulina de forma directa, elevando los triglicéridos séricos y promoviendo esteatosis hepática antes incluso de un aumento masivo de peso en la báscula.

## 3. Protocolo de Monitorización Metabólica Estricta
- **Basal (antes de iniciar):** Peso, IMC, perímetro abdominal, presión arterial, glucemia basal en ayunas y perfil lipídico completo.
- **A las 4 y 8 semanas:** Control de peso e IMC (una ganancia superior al 5% del peso basal en el primer mes es un marcador altamente predictivo de obesidad futura severa).
- **A los 3 meses y anual:** Glucemia, hemoglobina glicosilada (HbA1c) y perfil lipídico.

## 4. Estrategia de Intervención y Reversión
1. **Rotación Estratégica:** Si la eficacia psicopatológica lo permite, sustituir por antipsicóticos con neutralidad metabólica comprobada: **Aripiprazol, Lurasidona, Ziprasidona o Cariprazina**.
2. **Coadyuvancia con Metformina:** Iniciar metformina (500 a 850 mg dos veces al día) en etapas tempranas si el cambio de antipsicótico está contraindicado por riesgo de recaída psicótica grave.
"""
    )

    write_doc(
        "03_cascadas_prescripcion_y_desprescripcion/protocolo_conversacional_deteccion_cascadas.md",
        """---
id: "CASC-PROT-003"
tipo: "cascada_prescripcion"
titulo: "Protocolo Conversacional y Algoritmo Clínico para la Detección de Cascadas de Prescripción"
farmacos_involucrados: ["Todos los psicofármacos"]
fuentes_primarias: ["Jufe - Psicofarmacología Práctica", "Criterios STOPP/START"]
nivel_riesgo: "Moderado a Alto"
palabras_clave: ["cascada de prescripcion", "desprescripcion", "iatrogenia", "polifarmacia", "anamnesis farmacologica", "asistente de voz"]
---

# Protocolo Conversacional para Detección de Cascadas de Prescripción

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "En medicina y psiquiatría rige un axioma fundamental de seguridad: cualquier síntoma nuevo que aparezca tras iniciar, cambiar o aumentar la dosis de un fármaco debe considerarse un efecto adverso hasta que se demuestre lo contrario. El protocolo de detección de cascadas interroga la cronología exacta de cada prescripción. Si un paciente comenzó a tomar un protector gástrico, un anticolinérgico, un antihipertensivo o un ansiolítico poco después de un ajuste farmacológico previo, el asistente clínico alerta al médico sobre una posible cascada de prescripción, recomendando desescalar o sustituir el fármaco causal en lugar de cronificar la polifarmacia."

## 2. Las Tres Preguntas Clave del Algoritmo
Ante cualquier paciente polimedicado que refiera un nuevo motivo de consulta:
1. **¿Cuándo comenzó exactamente el síntoma actual?**
2. **¿Qué modificaciones se hicieron en su medicación en las 8 semanas previas a la aparición del síntoma?**
3. **¿Puede la farmacología del fármaco causal explicar este nuevo cuadro clínico?**

## 3. Catálogo de Cascadas Frecuentes en Salud Mental
- **Antipsicótico (Haloperidol, Risperidona) → Parkinsonismo → Biperideno** (Carga anticolinérgica).
- **ISRS (Fluoxetina, Sertralina) → Acatisia / Inquietud motora → Benzodiacepinas** (Sedación y adicción).
- **ISRS (Paroxetina) → Disfunción sexual → Sildenafilo** (Riesgo cardiovascular).
- **Antipsicótico Atípico (Olanzapina) → Hiperglucemia / Obesidad → Metformina / Estatinas**.
- **Carbonato de Litio → Temblor fino → Propranolol** (Bradicardia, broncoespasmo).
- **Inhibidor de la Colinesterasa (Donepezilo) → Incontinencia urinaria de urgencia → Oxibutinina** (Antagonismo del efecto procognitivo).
"""
    )

    # =========================================================================
    # 04. GUÍAS CLÍNICAS Y ALGORITMOS DE DECISIÓN CANMAT/ISBD (5 Archivos)
    # =========================================================================

    write_doc(
        "04_guias_clinicas_y_algoritmos_decision/canmat_bipolar_mania_aguda.md",
        """---
id: "GUIA-CANMAT-MANIA-001"
tipo: "guia_clinica"
patologia: "Trastorno Bipolar I y II"
fase: "Manía Aguda"
fuentes_primarias: ["CANMAT/ISBD 2018 Guidelines for Bipolar Disorder"]
nivel_evidencia: "Level 1"
nivel_riesgo: "Alto (Emergencia Psiquiátrica)"
palabras_clave: ["canmat", "bipolar", "mania aguda", "litio", "valproato", "quetiapina", "aripiprazol", "primera linea"]
---

# Guía CANMAT / ISBD 2018: Tratamiento de la Manía Bipolar Aguda

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "Para el tratamiento de la manía aguda en el trastorno bipolar, las guías oficiales CANMAT e ISBD 2018 establecen como intervenciones de primera línea con nivel de evidencia uno tanto la monoterapia como la terapia combinada. En monoterapia, los fármacos recomendados son el carbonato de litio, el divalproato de sodio y antipsicóticos atípicos como la quetiapina, el aripiprazol, la risperidona y la asenapina. En cuadros de manía severa con agitación psicomotora o síntomas psicóticos, la combinación de litio o valproato junto a un antipsicótico atípico es significativamente superior a la monoterapia en velocidad de respuesta. Cualquier antidepresivo debe ser suspendido de inmediato."

## 2. Jerarquía de Evidencia CANMAT 2018 para Manía Aguda

### Primera Línea (Evidencia Level 1)
- **Monoterapia:**
  - Carbonato de Litio (Litemia objetivo: 0.8 a 1.1 mEq/L).
  - Divalproato de Sodio / Ácido Valproico (Nivel plasmático: 50 a 125 µg/mL).
  - Quetiapina (Dosis de 400 a 800 mg/día).
  - Aripiprazol (Dosis de 15 a 30 mg/día).
  - Risperidona (Dosis de 2 a 6 mg/día).
  - Asenapina (Dosis de 10 a 20 mg/día sublingual).
  - Cariprazina y Paliperidona.
- **Terapia Combinada de Primera Línea:**
  - **Litio o Divalproato + Antipsicótico Atípico** (Quetiapina, Aripiprazol, Risperidona o Asenapina). Esta combinación ofrece la mayor tasa de remisión sintomática a corto plazo.

### Segunda Línea (Evidencia Level 2)
- Olanzapina en monoterapia (muy eficaz pero desplazada a segunda línea por su severo impacto metabólico y ganancia ponderal a medio plazo).
- Carbamazepina (eficaz pero compleja por interacciones enzimáticas masivas de CYP3A4).
- Haloperidol (eficaz para control rápido pero con alto riesgo extrapiramidal y de viraje a depresión post-manía).

## 3. Medidas Inmediatas Obligatorias
1. **Retirada Inmediata de Antidepresivos:** Si el paciente venía recibiendo cualquier antidepresivo (ISRS, dual, tricíclico), suspenderlo de forma inmediata para detener el estímulo maníaco.
2. **Evaluación de Factores Desencadenantes:** Descartar consumo de sustancias psicoactivas (cocaína, anfetaminas), esteroides sistémicos o privación severa de sueño.
"""
    )

    write_doc(
        "04_guias_clinicas_y_algoritmos_decision/canmat_bipolar_depresion_y_viraje.md",
        """---
id: "GUIA-CANMAT-DEP-002"
tipo: "guia_clinica"
patologia: "Trastorno Bipolar I y II"
fase: "Depresión Bipolar"
fuentes_primarias: ["CANMAT/ISBD 2018 Guidelines for Bipolar Disorder"]
nivel_evidencia: "Level 1 / Alerta de Contraindicación (Level 1 de Daño)"
nivel_riesgo: "Crítico"
palabras_clave: ["depresion bipolar", "canmat", "viraje maniaco", "quetiapina", "lurasidona", "lamotrigina", "isrs contraindicados", "ciclos rapidos"]
---

# Guía CANMAT / ISBD 2018: Depresión Bipolar y Alerta Crítica de Viraje a Manía

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "El tratamiento de la depresión bipolar según las guías CANMAT e ISBD difiere radicalmente de la depresión unipolar. Los tratamientos de primera línea basados en evidencia nivel uno son la quetiapina en monoterapia a trescientos miligramos al día, la lurasidona sola o combinada con litio o valproato, y la lamotrigina. La alerta clínica más crítica de esta guía es una contraindicación formal con evidencia de nivel uno de daño: los antidepresivos tradicionales en monoterapia están formalmente desaconsejados por su alto riesgo de inducir un viraje maníaco agudo, desestabilizar la enfermedad hacia ciclos rápidos y aumentar el riesgo suicida."

## 2. Jerarquía Terapéutica CANMAT para Depresión Bipolar

### Primera Línea (Evidencia Level 1 y Level 2)
1. **Quetiapina (Monoterapia):** Dosis objetivo de 300 mg/día al acostarse (robusta eficacia demostrada en estudios BOLDER I y II, tanto en Bipolar I como Bipolar II).
2. **Lurasidona (Monoterapia o Add-on a Litio/Valproato):** Dosis de 20 a 120 mg/día (debe administrarse con una comida de al menos 350 calorías; excelente neutralidad metabólica).
3. **Litio (Monoterapia):** Manteniendo litemia entre 0.8 y 1.0 mEq/L.
4. **Lamotrigina (Monoterapia o Add-on):** Requiere titulación lenta a 100-200 mg/día.
5. **Combinación Olanzapina + Fluoxetina (OFC):** Aprobada pero con reserva por síndrome metabólico.

### Contraindicación Formal / No Recomendado (Alerta de Seguridad)
> [!CAUTION]
> **Antidepresivos en Monoterapia (ISRS, IRSN, Tricíclicos):**
> - **Nivel de Evidencia:** Level 1 de Daño (*Harm*).
> - **Justificación:** No han demostrado eficacia superior al placebo en depresión bipolar si no van acompañados de un estabilizador, y disparan la tasa de viraje a manía o hipomanía, acelerando la ciclación rápida.

## 3. Protocolo de Manejo si se Decide Usar un Antidepresivo
Solo se considerará añadir un antidepresivo (preferentemente bupropión o sertralina, evitando tricíclicos y venlafaxina) si se cumplen dos condiciones estrictas:
1. El paciente se encuentra bajo cobertura óptima y confirmada con un estabilizador del ánimo de primera línea (Litio en rango terapéutico o Divalproato).
2. Se retira el antidepresivo de forma escalonada una vez alcanzada la remisión del episodio depresivo.
"""
    )

    write_doc(
        "04_guias_clinicas_y_algoritmos_decision/canmat_bipolar_mantenimiento.md",
        """---
id: "GUIA-CANMAT-MANT-003"
tipo: "guia_clinica"
patologia: "Trastorno Bipolar I y II"
fase: "Mantenimiento y Prevención de Recaídas a Largo Plazo"
fuentes_primarias: ["CANMAT/ISBD 2018 Guidelines for Bipolar Disorder"]
nivel_evidencia: "Level 1"
nivel_riesgo: "Moderado a Alto"
palabras_clave: ["canmat", "mantenimiento", "bipolar", "litio", "quetiapina", "valproato", "lamotrigina", "prevencion de suicidio"]
---

# Guía CANMAT / ISBD 2018: Fase de Mantenimiento y Profilaxis del Trastorno Bipolar

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "La fase de mantenimiento del trastorno bipolar busca prevenir nuevos episodios maníacos y depresivos, reducir la polaridad predominante y proteger al paciente contra el riesgo de suicidio. Las guías CANMAT sitúan al carbonato de litio en la cúspide de la primera línea terapéutica gracias a su capacidad preventiva bipolar global y a su demostrada acción reductora de la mortalidad por suicidio. Otras opciones de primera línea en monoterapia son la quetiapina, el divalproato de sodio y la lamotrigina, teniendo en cuenta que la lamotrigina previene principalmente las recaídas depresivas pero no protege contra la manía. En pacientes con recaídas frecuentes se aconseja la combinación sinérgica de litio con quetiapina o valproato."

## 2. Fármacos de Primera Línea en Mantenimiento y su Polaridad Protectora

| Fármaco | Nivel Evidencia | Prevención de Manía | Prevención de Depresión | Observación Clínica Clave |
| :--- | :--- | :--- | :--- | :--- |
| **Carbonato de Litio** | **Level 1** | **Excelente (+++)** | **Buena (++)** | **Estándar de oro; prevención probada de suicidio** |
| **Quetiapina** | **Level 1** | **Excelente (+++)** | **Excelente (+++)** | Eficaz en ambas polaridades; vigilar peso y sedación |
| **Divalproato de Sodio** | **Level 1** | **Excelente (+++)** | Moderada (+) | Eficaz en manías mixtas y cicladores rápidos |
| **Lamotrigina** | **Level 1** | Pobre (-) | **Excelente (+++)** | Elección en pacientes con predominio depresivo |
| **Aripiprazol** | **Level 1** | **Excelente (+++)** | Pobre (-) | Excelente tolerancia metabólica |

## 3. Pautas Clínicas de Retirada o Continuidad
- El tratamiento de mantenimiento debe prolongarse indefinidamente tras dos o más episodios severos de trastorno bipolar.
- Si se suspende el litio de forma brusca, la tasa de recaída maníaca en los siguientes 6 meses supera el 90%. Cualquier suspensión debe ser pactada, muy lenta y gradual a lo largo de un mínimo de 3 a 6 meses.
"""
    )

    write_doc(
        "04_guias_clinicas_y_algoritmos_decision/canmat_tdm_fase_aguda_primera_linea.md",
        """---
id: "GUIA-CANMAT-TDM-AGUDA-004"
tipo: "guia_clinica"
patologia: "Trastorno Depresivo Mayor (TDM)"
fase: "Fase Aguda (Primera Línea de Tratamiento)"
fuentes_primarias: ["CANMAT 2016 Guidelines for Major Depressive Disorder"]
nivel_evidencia: "Level 1"
nivel_riesgo: "Moderado"
palabras_clave: ["canmat", "tdm", "depresion mayor", "primera linea", "isrs", "irsn", "sertralina", "escitalopram", "bupropion"]
---

# Guía CANMAT 2016: Tratamiento de Primera Línea en el Trastorno Depresivo Mayor

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "Para el tratamiento del episodio depresivo mayor agudo en adultos, las guías CANMAT establecen como opciones de primera línea con evidencia de nivel uno a varios antidepresivos de segunda generación. Entre ellos figuran los inhibidores selectivos de recaptación de serotonina como citalopram, escitalopram y sertralina; los inhibidores duales de serotonina y noradrenalina como venlafaxina y duloxetina; el bupropión y la mirtazapina. La elección del fármaco no se basa en diferencias mayores de eficacia global, sino en el perfil de tolerabilidad, la presencia de síntomas diana como insomnio o pérdida de apetito, las enfermedades médicas concomitantes y la evitación de interacciones enzimáticas."

## 2. Fármacos de Primera Línea Recomendados (Evidencia Level 1)
- **ISRS:**
  - Citalopram (Dosis habitual: 20 a 40 mg/día; precaución por intervalo QT en > 60 años).
  - Escitalopram (Dosis habitual: 10 a 20 mg/día; el más selectivo sobre el transportador SERT).
  - Sertralina (Dosis habitual: 50 a 200 mg/día; perfil seguro en cardiopatía e interacciones bajas).
- **IRSN (Duales):**
  - Venlafaxina (Dosis habitual: 75 a 225 mg/día; vigilar tensión arterial a dosis altas).
  - Duloxetina (Dosis habitual: 60 a 120 mg/día; de elección si coexiste dolor crónico o fibromialgia).
- **Otras Familias:**
  - Bupropión (Dosis habitual: 150 a 300 mg XL; preserva la función sexual y peso).
  - Mirtazapina (Dosis habitual: 15 a 45 mg; potente sedante y estimulante del apetito).
  - Vortioxetina (Dosis habitual: 10 a 20 mg; beneficio procognitivo demostrado).

## 3. Criterios de Evaluación y Tiempo de Respuesta
- El tiempo adecuado para evaluar la eficacia de un antidepresivo a dosis terapéutica plena es de **4 a 6 semanas**.
- Si a las 2 a 4 semanas no se observa una reducción de al menos el 20-25% de los síntomas en escalas validadas (como MADRS o PHQ-9), la probabilidad de alcanzar la remisión con ese mismo esquema es menor al 15%, debiendo optimizarse la dosis o planificar una rotación.
"""
    )

    write_doc(
        "04_guias_clinicas_y_algoritmos_decision/canmat_tdm_resistencia_y_potenciacion.md",
        """---
id: "GUIA-CANMAT-TDM-RESIST-005"
tipo: "guia_clinica"
patologia: "Trastorno Depresivo Mayor (TDM)"
fase: "Depresión Resistente y Estrategias de Potenciación (Add-on)"
fuentes_primarias: ["CANMAT 2016 Guidelines for Major Depressive Disorder"]
nivel_evidencia: "Level 1"
nivel_riesgo: "Alto"
palabras_clave: ["depresion resistente", "canmat", "potenciacion", "add on", "aripiprazol", "litio", "quetiapina", "cambio vs potenciacion"]
---

# Guía CANMAT 2016: Estrategias en Depresión Resistente al Tratamiento

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "Cuando un paciente con trastorno depresivo mayor no responde tras dos ensayos adecuados con antidepresivos de primera línea a dosis plenas, nos encontramos ante una depresión resistente. Las guías CANMAT recomiendan dos estrategias principales: el cambio de antidepresivo o la potenciación mediante adición de un segundo fármaco. Si el paciente ha presentado una respuesta parcial, la potenciación es la estrategia preferida con evidencia de nivel uno. Los fármacos potenciadores con mayor respaldo científico son los antipsicóticos atípicos a dosis bajas, especialmente el aripiprazol de dos a cinco miligramos y la quetiapina, así como la adición clásica de carbonato de litio en dosis moderadas."

## 2. Algoritmo: ¿Cuándo Cambiar y Cuándo Potenciar?
- **Cambio de Antidepresivo (*Switch*):**
  - Indicado si la respuesta al primer fármaco ha sido **nula (menos del 25% de mejoría)** o si han surgido efectos adversos intolerables.
  - Se puede rotar a otro fármaco de la misma clase (ISRS a ISRS) o preferentemente a una clase con mecanismo complementario (de ISRS a Dual, Mirtazapina o Bupropión).
- **Potenciación (*Augmentation / Add-on*):**
  - Indicada si ha habido una **respuesta parcial (entre 25% y 50% de mejoría)** y el fármaco actual es bien tolerado.

## 3. Fármacos de Potenciación de Primera Línea (Evidencia Level 1)
1. **Aripiprazol (Add-on):**
   - Dosis baja de inicio: 2 a 5 mg/día (máximo 10 mg/día).
   - Su agonismo parcial D2 estimula la neurotransmisión dopaminérgica en áreas límbicas sin inducir somnolencia ni ganancia de peso masiva.
2. **Quetiapina (Add-on):**
   - Dosis de 150 a 300 mg/día de liberación prolongada.
   - Excelente para insomnio rebelde, ansiedad psíquica intensa y rumiación depresiva.
3. **Carbonato de Litio (Add-on):**
   - Estrategia clásica de potenciación; requiere litemia objetivo entre 0.5 y 0.8 mEq/L.
   - Aporta protección antisuicida probada en pacientes deprimidos con ideación recurrente.
"""
    )

    # =========================================================================
    # 05. ALERTAS DE SEGURIDAD Y CAJA NEGRA (4 Archivos)
    # =========================================================================

    write_doc(
        "05_alertas_seguridad_y_caja_negra/alerta_agranulocitosis_clozapina.md",
        """---
id: "ALERTA-CLOZ-AGRAN-001"
tipo: "alerta_seguridad"
titulo: "Alerta Crítica de Caja Negra: Agranulocitosis y Neutropenia Grave Inducida por Clozapina"
farmacos_involucrados: ["Clozapina"]
fuentes_primarias: ["AEMPS", "FDA Boxed Warning", "Jufe - Psicofarmacología Práctica"]
nivel_riesgo: "Crítico / Emergencia Médica"
palabras_clave: ["clozapina", "agranulocitosis", "neutropenia", "ran", "hemograma", "infeccion", "caja negra"]
---

# Alerta Crítica de Seguridad: Agranulocitosis por Clozapina

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "Alerta crítica de seguridad médica: La clozapina conlleva una advertencia de caja negra por riesgo de agranulocitosis, una caída brusca de los glóbulos blancos neutrófilos por debajo de quinientos por microlitro que deja al paciente indefenso ante infecciones bacterianas fulminantes. Es obligatorio realizar hemogramas con recuento absoluto de neutrófilos cada semana durante las primeras dieciocho semanas y mensualmente después. Cualquier síntoma infeccioso, especialmente fiebre, dolor de garganta, úlceras orales o malestar general súbito, debe considerarse una agranulocitosis hasta que se demuestre lo contrario, exigiendo la suspensión inmediata de la clozapina y un hemograma urgente de emergencia."

## 2. Definiciones Hematológicas Mandatorias
- **Recuento Absoluto de Neutrófilos (RAN):**
  - *Rango Normal Seguro:* RAN ≥ 1500 /µL (≥ 1.5 x 10^9 /L).
  - *Neutropenia Leve:* RAN 1000 a 1499 /µL.
  - *Neutropenia Moderada:* RAN 500 a 999 /µL.
  - *Agranulocitosis Severa:* **RAN < 500 /µL** (riesgo inminente de sepsis y muerte).

## 3. Protocolo de Actuación ante Sospecha Infecciosa
```
┌────────────────────────────────────────────────────────┐
│ Paciente con Clozapina refiere FIEBRE o DOLOR DE FARINGE │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
    ┌──────────────────────────────────────────────────┐
    │ 1. NO esperar a la próxima cita rutinaria.       │
    │ 2. Suspender la dosis inmediata de Clozapina.    │
    │ 3. Hemograma completo con RAN URGENTE (< 2 horas)│
    └───────────────────────┬──────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
     RAN < 1000 /µL                  RAN ≥ 1500 /µL
┌───────────────────────┐       ┌────────────────────────┐
│ RETIRADA DEFINITIVA   │       │ Continuar Clozapina y  │
│ Aislamiento protector │       │ tratar cuadro infeccioso│
│ Derivación Hematología│       │ común                  │
└───────────────────────┘       └────────────────────────┘
```
"""
    )

    write_doc(
        "05_alertas_seguridad_y_caja_negra/alerta_sindrome_stevens_johnson_lamotrigina.md",
        """---
id: "ALERTA-LAM-SJS-002"
tipo: "alerta_seguridad"
titulo: "Alerta Crítica Dermatológica: Síndrome de Stevens-Johnson y Necrólisis Epidérmica Tóxica por Lamotrigina"
farmacos_involucrados: ["Lamotrigina", "Ácido Valproico"]
fuentes_primarias: ["FDA Boxed Warning", "AEMPS", "CANMAT Guidelines", "Jufe"]
nivel_riesgo: "Crítico / Emergencia Médica"
palabras_clave: ["lamotrigina", "stevens johnson", "net", "exantema", "reaccion medicamentosa", "desprendimiento epidermico", "caja negra"]
---

# Alerta Crítica: Síndrome de Stevens-Johnson por Lamotrigina

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "Alerta clínica de seguridad: La lamotrigina presenta una advertencia de caja negra por riesgo de reacciones dermatológicas graves y potencialmente mortales, como el síndrome de Stevens-Johnson y la necrólisis epidérmica tóxica. Estas reacciones suelen manifestarse durante las primeras dos a ocho semanas de tratamiento y se relacionan directamente con una titulación demasiado rápida o con la coadministración de ácido valproico sin reducir la dosis. Los signos cardinales de peligro son la aparición de fiebre, erupción con ampollas o dianas en la piel y afectación dolorosa de mucosas en boca, ojos o genitales. Ante estos signos, la lamotrigina debe interrumpirse inmediatamente y de forma permanente; nunca debe volver a administrarse a un paciente que haya sufrido esta reacción."

## 2. Diferenciación: Exantema Benigno vs. Stevens-Johnson

| Signo Clínico | Exantema Maculopapular Benigno (10%) | **Síndrome de Stevens-Johnson / NET** |
| :--- | :--- | :--- |
| **Inicio Temporal** | Semanas 2 a 4 | Primeras semanas tras dosis alta |
| **Fiebre y Malestar** | Ausente o muy leve | **Fiebre alta, odinofagia, postración** |
| **Mucosas (Boca/Ojos/Genitales)** | **Completamente respetadas** | **Afectación erosiva de 2 o más mucosas** |
| **Lesiones en Piel** | Pápulas aisladas en tronco | **Dianas atípicas, ampollas y signo de Nikolsky (+)** |
| **Conducta** | Monitorización estrecha | **SUSPENSIÓN INMEDIATA + INGRESO EN QUEMADOS** |

## 3. Factores de Riesgo Primarios
1. Incumplimiento de la pauta de titulación lenta (iniciar a 50 o 100 mg/día directamente).
2. Coadministración de ácido valproico sin haber reducido la dosis de inicio a 25 mg en días alternos.
3. Antecedente previo de erupción alérgica con otros anticonvulsivantes aromáticos (carbamazepina).
"""
    )

    write_doc(
        "05_alertas_seguridad_y_caja_negra/alerta_toxicidad_litemia_litio.md",
        """---
id: "ALERTA-LIT-TOX-003"
tipo: "alerta_seguridad"
titulo: "Alerta Crítica Toxicológica: Intoxicación Aguda y Rango Terapéutico Estrecho del Litio"
farmacos_involucrados: ["Carbonato de Litio"]
fuentes_primarias: ["Jufe - Psicofarmacología Práctica", "CANMAT 2018", "Stahl"]
nivel_riesgo: "Crítico / Emergencia Médica"
palabras_clave: ["litio", "litemia", "intoxicacion por litio", "ataxia", "hemodialisis", "rango estrecho", "neurotoxicidad"]
---

# Alerta Crítica Toxicológica: Intoxicación por Carbonato de Litio

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "Alerta toxicológica médica: El carbonato de litio tiene una ventana terapéutica extremadamente estrecha; la diferencia entre una dosis eficaz y una dosis tóxica es mínima. Una litemia superior a 1.2 miliequivalentes por litro ya produce toxicidad leve con temblor grueso, diarrea y letargia. Si los niveles superan los dos miliequivalentes por litro, el paciente entra en un cuadro neurológico grave con ataxia, disartria, rigidez, mioclonías y confusión mental. Concentraciones mayores a dos punto cinco pueden provocar convulsiones, coma, daño cerebral irreversible y muerte. Si se sospecha intoxicación, se debe suspender el litio de inmediato, medir litemia urgente, forzar hidratación con suero fisiológico para barrer el litio por el riñón y preparar hemodiálisis de emergencia si los niveles son críticos."

## 2. Estratificación de la Intoxicación según la Litemia

```
LITEMIA PLASMÁTICA (mEq/L)
│
├── 0.6 - 1.0 mEq/L ───► [RANGO TERAPÉUTICO SEGURO]
│
├── 1.2 - 1.5 mEq/L ───► [TOXICIDAD LEVE A MODERADA]
│                         Temblor grosero de manos, náuseas, vómitos, diarrea acuosa,
│                         debilidad muscular y somnolencia.
│
├── 1.5 - 2.5 mEq/L ───► [TOXICIDAD GRAVE]
│                         Ataxia cerebelosa franca, habla arrastrada (disartria),
│                         nistagmo, hiperreflexia, clonus y confusión mental.
│
└── > 2.5 mEq/L ───────► [TOXICIDAD POTENCIALMENTE LETAL]
                          Convulsiones tónico-clónicas, arritmias, fracaso renal agudo,
                          coma. Requiere HEMODIÁLISIS URGENTE.
```

## 3. Criterios de Hemodiálisis de Emergencia
- Litemia > 4.0 mEq/L independientemente del estado clínico.
- Litemia > 2.5 mEq/L con presencia de síntomas neurológicos graves (coma, convulsiones) o insuficiencia renal aguda que impida la excreción espontánea del catión.
"""
    )

    write_doc(
        "05_alertas_seguridad_y_caja_negra/alerta_viraje_maniaco_isrs_bipolar.md",
        """---
id: "ALERTA-ISRS-VIRAJE-004"
tipo: "alerta_seguridad"
titulo: "Alerta de Práctica Clínica: Riesgo de Viraje Maníaco y Ciclación Rápida por Antidepresivos en Trastorno Bipolar"
farmacos_involucrados: ["Sertralina", "Fluoxetina", "Paroxetina", "Venlafaxina", "Antidepresivos Tricíclicos"]
fuentes_primarias: ["CANMAT/ISBD 2018 Guidelines", "Jufe - Psicofarmacología Práctica"]
nivel_riesgo: "Alto a Crítico"
palabras_clave: ["viraje maniaco", "bipolar", "isrs", "antidepresivos contraindicados", "ciclos rapidos", "canmat", "manic switch"]
---

# Alerta Clínica: Viraje Maníaco y Desestabilización por Antidepresivos

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "Alerta de práctica médica en salud mental: El uso de antidepresivos en pacientes con trastorno bipolar, en especial cuando se administran en monoterapia sin la cobertura previa de un estabilizador del ánimo, está contraindicado por el alto riesgo de provocar un viraje maníaco. Esto significa transformar súbitamente una depresión en un episodio de manía o hipomanía aguda con euforia patológica, insomnio sin cansancio, verborrea, hipersexualidad y conducta temeraria. Además, los antidepresivos pueden acelerar la enfermedad hacia un patrón de ciclos rápidos y agravar el riesgo suicida por estados mixtos disfóricos. La directriz de seguridad exige suspender inmediatamente el antidepresivo y optimizar el estabilizador o antipsicótico atípico."

## 2. Signos de Alarma Temprana de Viraje Maníaco
Cuando un paciente diagnosticado de depresión comienza a tomar un antidepresivo y en los días siguientes presenta:
1. **Reducción drástica de la necesidad de dormir:** Despierta tras 2 o 3 horas sintiéndose con desbordante energía.
2. **Aceleración del pensamiento y del lenguaje (Taquipsiquia y taquilalia):** Habla imparable y salta de una idea a otra sin coherencia.
3. **Hiperactividad no productiva y gastos desmedidos:** Inicia múltiples proyectos incoherentes o realiza compras compulsivas fuera de su patrón habitual.
4. **Irritabilidad extrema o desinhibición conductual:** Explosiones de ira o pérdida del pudor social.

## 3. Conducta Inmediata ante el Viraje
- **Paso 1:** Descontinuar de inmediato el fármaco antidepresivo causante.
- **Paso 2:** Introducir o titular un estabilizador de acción antimaníaca rápida de primera línea según CANMAT (Quetiapina, Aripiprazol o Carbonato de Litio).
- **Paso 3:** Si existe agitación psicomotora severa, coadyuvar transitoriamente con benzodiacepinas de alta potencia (como clonazepam o lorazepam) en espera del efecto antimaníaco pleno.
"""
    )

    print("\n¡Generación completada con éxito! Todos los archivos fueron creados.")

if __name__ == "__main__":
    build_all()
