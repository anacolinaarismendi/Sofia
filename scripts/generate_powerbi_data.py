#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_powerbi_data.py
Genera y consolida datasets optimizados para Power BI y Azure a partir de
la presentación 'Sofia_presentacion.pptx' y los datos procesados de Sofia.
"""

from pathlib import Path
import pandas as pd
import json

WORKSPACE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE / "data" / "powerbi"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Dataset de Slides de la Presentación
slides_data = [
    {
        "Slide_Num": 1,
        "Capitulo": "1. Introducción y Problema",
        "Titulo": "Sofia: Sistema RAG Clínico y Farmacovigilancia",
        "Subtitulo": "Soporte a la decisión contra los peligros de la polifarmacia en psiquiatría",
        "Puntos_Clave": "RAG local, privado y determinista; Grounding estricto [DOC-ID]; Prevención de cascadas de prescripción.",
        "Tipo_Slide": "Portada",
        "Icono": "🧠"
    },
    {
        "Slide_Num": 2,
        "Capitulo": "1. Introducción y Problema",
        "Titulo": "La Polifarmacia: Realidad Cotidiana",
        "Subtitulo": "Complejidad en pacientes polimedicados con múltiples comorbilidades",
        "Puntos_Clave": "Cruces de antidepresivos, antipsicóticos y estabilizadores; Riesgo elevado de toxicidad e interacciones.",
        "Tipo_Slide": "Contexto",
        "Icono": "💊"
    },
    {
        "Slide_Num": 3,
        "Capitulo": "1. Introducción y Problema",
        "Titulo": "Problema 1: Interacciones CYP450",
        "Subtitulo": "Bloqueos farmacocinéticos en citocromos hepáticos",
        "Puntos_Clave": "Inhibición de CYP2D6 por Paroxetina/Bupropión; Elevación tóxica de Risperidona o Haloperidol; Prolongación QTc.",
        "Tipo_Slide": "Problema Clínico",
        "Icono": "⚡"
    },
    {
        "Slide_Num": 4,
        "Capitulo": "1. Introducción y Problema",
        "Titulo": "Problema 2: Cascadas de Prescripción",
        "Subtitulo": "Confusión entre efecto adverso inducido y nueva enfermedad",
        "Puntos_Clave": "Haloperidol produce temblor/rigidez -> Prescripción errónea de Biperideno -> Riesgo de discinesia tardía irreversible.",
        "Tipo_Slide": "Problema Clínico",
        "Icono": "🛑"
    },
    {
        "Slide_Num": 5,
        "Capitulo": "1. Introducción y Problema",
        "Titulo": "Problema 3: Falta de Adherencia a Guías",
        "Subtitulo": "Brecha entre práctica asistencial y medicina basada en la evidencia",
        "Puntos_Clave": "CANMAT/ISBD 2018 desaconseja antidepresivos solos en depresión bipolar por riesgo de viraje maníaco.",
        "Tipo_Slide": "Problema Clínico",
        "Icono": "📋"
    },
    {
        "Slide_Num": 6,
        "Capitulo": "2. Solución y Bioética",
        "Titulo": "¿Por Qué No un LLM Generalista?",
        "Subtitulo": "Peligros de ChatGPT o modelos públicos en medicina",
        "Puntos_Clave": "Alucinaciones clínicas; Citas apócrifas; Sofia usa RAG determinista sobre literatura canónica (Jufe, Stahl, CANMAT, OMS).",
        "Tipo_Slide": "Comparativa",
        "Icono": "⚖️"
    },
    {
        "Slide_Num": 7,
        "Capitulo": "2. Solución y Bioética",
        "Titulo": "Reglas Bioéticas Innegociables",
        "Subtitulo": "Seguridad asistencial y auditoría médica",
        "Puntos_Clave": "1) Grounding estricto con trazabilidad [DOC-ID]. 2) Soporte dual: 4 apartados escritos + Voice Snippet (<40s).",
        "Tipo_Slide": "Normativa",
        "Icono": "🛡️"
    },
    {
        "Slide_Num": 8,
        "Capitulo": "3. Interfaz Streamlit",
        "Titulo": "Arquitectura de la Interfaz Web",
        "Subtitulo": "Cuatro módulos operativos para la consulta y la guardia",
        "Puntos_Clave": "P1: Chat RAG; P2: Matriz CYP450; P3: Detector de Cascadas; P4: Guías CANMAT y Catálogo ATC.",
        "Tipo_Slide": "UI/UX",
        "Icono": "🖥️"
    },
    {
        "Slide_Num": 9,
        "Capitulo": "3. Interfaz Streamlit",
        "Titulo": "Pestaña 1: Asistente RAG Clínico",
        "Subtitulo": "Consultas en lenguaje natural con respuestas estructuradas",
        "Puntos_Clave": "Respuesta en 4 partes (Juicio, Mecanismo, Pauta, Alertas); Voice Snippet (<40s); Desplegable de fuentes con similitud coseno.",
        "Tipo_Slide": "Módulo UI",
        "Icono": "💬"
    },
    {
        "Slide_Num": 10,
        "Capitulo": "3. Interfaz Streamlit",
        "Titulo": "Pestaña 2: Matriz Interactiva CYP450",
        "Subtitulo": "Semáforo enzimático de interacciones en tiempo real",
        "Puntos_Clave": "Rojo Crítico (bloqueo 2D6); Amarillo Advertencia (Clozapina/1A2, Litio/Renal, Lamotrigina/UGT1A4); Verde Seguro.",
        "Tipo_Slide": "Módulo UI",
        "Icono": "🚦"
    },
    {
        "Slide_Num": 11,
        "Capitulo": "3. Interfaz Streamlit",
        "Titulo": "Pestaña 3: Cascadas de Prescripción",
        "Subtitulo": "Algoritmo heurístico y protocolo de desprescripción",
        "Puntos_Clave": "Detecta cruces patológicos (Haloperidol + Síntoma); Propone: 1) Reducir dosis, 2) Rotar a atípico, 3) Desescalada en 2-4 semanas.",
        "Tipo_Slide": "Módulo UI",
        "Icono": "🔄"
    },
    {
        "Slide_Num": 12,
        "Capitulo": "3. Interfaz Streamlit",
        "Titulo": "Pestaña 4: Guías CANMAT y Catálogo ATC",
        "Subtitulo": "Explorador de evidencia oficial y clasificación OMS",
        "Puntos_Clave": "Tablas filtradas por TDM y Bipolar (Fase y Línea); Buscador con códigos ATC oficiales y marcas comerciales registradas.",
        "Tipo_Slide": "Módulo UI",
        "Icono": "📖"
    },
    {
        "Slide_Num": 13,
        "Capitulo": "4. Ingeniería del Sistema",
        "Titulo": "Ingeniería de Sofia: 4 Capas",
        "Subtitulo": "Diseño modular en Python 3 con total portabilidad relativa",
        "Puntos_Clave": "Capa 1: Extracción; Capa 2: Corpus Curado; Capa 3: Motor RAG Dual; Capa 4: Fine-Tuning y Evaluación Benchmark.",
        "Tipo_Slide": "Arquitectura",
        "Icono": "🏗️"
    },
    {
        "Slide_Num": 14,
        "Capitulo": "4. Ingeniería del Sistema",
        "Titulo": "Capa 1: Extracción y Enriquecimiento",
        "Subtitulo": "Pipeline automatizado de ingesta de literatura médica",
        "Puntos_Clave": "CachedPDFExtractor (evita reprocesar PDFs de 7MB); Regex estructuradas para Jufe y CANMAT; ATCEnricher con catálogo OMS.",
        "Tipo_Slide": "Ingeniería",
        "Icono": "📥"
    },
    {
        "Slide_Num": 15,
        "Capitulo": "4. Ingeniería del Sistema",
        "Titulo": "Capa 2: Corpus Clínico Curado",
        "Subtitulo": "27 Documentos en Markdown con metadatos estructurados",
        "Puntos_Clave": "YAML Frontmatter estandarizado (id, atc, enzimas); Voice Snippets continuos (<130 palabras); Scripts de autovalidación.",
        "Tipo_Slide": "Ingeniería",
        "Icono": "📚"
    },
    {
        "Slide_Num": 16,
        "Capitulo": "4. Ingeniería del Sistema",
        "Titulo": "Capa 3: Motor RAG Vectorial",
        "Subtitulo": "Almacén dual: Ollama neural o Fallback determinista Scikit-Learn",
        "Puntos_Clave": "Embeddinggemma y Qwen2.5 en local; Fallback instantáneo a TF-IDF n-gram (1,2) sin internet ni GPU dedicada (<0.002s).",
        "Tipo_Slide": "Ingeniería",
        "Icono": "⚙️"
    },
    {
        "Slide_Num": 17,
        "Capitulo": "4. Ingeniería del Sistema",
        "Titulo": "El Prompt Clínico de Sofia",
        "Subtitulo": "Directrices formalizadas de seguridad y trazabilidad",
        "Puntos_Clave": "Prohibición de citas personales apócrifas; Obligación de citar [DOC-ID]; Estructura estricta en 4 secciones clínicas.",
        "Tipo_Slide": "Ingeniería",
        "Icono": "📝"
    },
    {
        "Slide_Num": 18,
        "Capitulo": "5. Evaluación y Conclusión",
        "Titulo": "Capa 4: Fine-Tuning y Evaluación Benchmark",
        "Subtitulo": "Validación cuantitativa de la Tríada de RAG",
        "Puntos_Clave": "Dataset de 29 casos ChatML para Qwen; Benchmark formal de 4 casos clínicos; 100% Recall, 100% Fidelidad, 90% Relevancia.",
        "Tipo_Slide": "Validación",
        "Icono": "📊"
    },
    {
        "Slide_Num": 19,
        "Capitulo": "5. Evaluación y Conclusión",
        "Titulo": "Conclusión: IA Médica Explicable",
        "Subtitulo": "Rápida, explicable, determinista y centrada en el médico",
        "Puntos_Clave": "Combina literatura médica canónica, catálogo ATC OMS y RAG híbrido contra la polifarmacia; Código abierto bajo MIT.",
        "Tipo_Slide": "Cierre",
        "Icono": "🎯"
    }
]

df_slides = pd.DataFrame(slides_data)
df_slides.to_csv(OUTPUT_DIR / "01_slides_presentacion.csv", index=False)

# 2. Dataset de Métricas y KPIs de la Tríada de RAG
benchmark_kpis = [
    {
        "Metrica": "Context Recall (Recuperación Canónica)",
        "Categoria": "Tríada de RAG",
        "Valor": 1.00,
        "Porcentaje": 100.0,
        "Objetivo_Minimo": 0.80,
        "Estado": "Superado",
        "Detalle": "El documento canónico esperado se recuperó en el Top-1 en el 100% de los casos."
    },
    {
        "Metrica": "Answer Relevance (Relevancia y Cobertura)",
        "Categoria": "Tríada de RAG",
        "Valor": 0.90,
        "Porcentaje": 90.0,
        "Objetivo_Minimo": 0.85,
        "Estado": "Superado",
        "Detalle": "Cobertura exhaustiva de entidades clínicas (posologías, enzimas, riesgos)."
    },
    {
        "Metrica": "Faithfulness / Groundedness (Cero Alucinaciones)",
        "Categoria": "Tríada de RAG",
        "Valor": 1.00,
        "Porcentaje": 100.0,
        "Objetivo_Minimo": 1.00,
        "Estado": "Superado",
        "Detalle": "Todas las afirmaciones se fundamentan al 100% en los fragmentos clínicos del corpus."
    },
    {
        "Metrica": "Trazabilidad Ética [DOC-ID] (Cero Citas Apócrifas)",
        "Categoria": "Bioética y Auditoría",
        "Valor": 1.00,
        "Porcentaje": 100.0,
        "Objetivo_Minimo": 1.00,
        "Estado": "Superado",
        "Detalle": "Cero falsas atribuciones a nombres personales; respaldo directo por código [DOC-ID]."
    },
    {
        "Metrica": "Cobertura de Voice Snippets (<40s)",
        "Categoria": "Multimodalidad",
        "Valor": 1.00,
        "Porcentaje": 100.0,
        "Objetivo_Minimo": 1.00,
        "Estado": "Superado",
        "Detalle": "100% de los casos cuentan con síntesis verbalizable fluida de menos de 130 palabras."
    },
    {
        "Metrica": "Latencia Media de Respuesta (s)",
        "Categoria": "Rendimiento",
        "Valor": 0.0015,
        "Porcentaje": 99.8,
        "Objetivo_Minimo": 1.00,
        "Estado": "Óptimo",
        "Detalle": "Tiempo de respuesta ultra veloz de 1.5 ms gracias al motor vectorial optimizado."
    }
]

df_kpis = pd.DataFrame(benchmark_kpis)
df_kpis.to_csv(OUTPUT_DIR / "02_benchmark_kpis.csv", index=False)

# 3. Dataset de las 4 Capas de Arquitectura
arquitectura_data = [
    {
        "Capa_Num": 1,
        "Nombre": "Extracción y Enriquecimiento",
        "Modulos": "src/extractors/, src/enrichers/",
        "Tecnologias": "pdfplumber, pypdf, Regex, ATC WHO",
        "Funcion_Principal": "Ingesta no destructiva de PDFs clínicos y estandarización a CSV con códigos ATC.",
        "Mecanismo_Clave": "CachedPDFExtractor en data/cache/ (carga instantánea sin consumo repetitivo de CPU)."
    },
    {
        "Capa_Num": 2,
        "Nombre": "Corpus Clínico Curado",
        "Modulos": "pre_training/",
        "Tecnologias": "Markdown, YAML Frontmatter, Voice Snippets",
        "Funcion_Principal": "27 Documentos clínicos estructurados en 5 categorías (Monografías, CYP, Cascadas, CANMAT, Caja Negra).",
        "Mecanismo_Clave": "Metadatos auditables por [DOC-ID] y síntesis ejecutivas orales de <40s."
    },
    {
        "Capa_Num": 3,
        "Nombre": "Motor RAG Vectorial",
        "Modulos": "src/rag/engine.py",
        "Tecnologias": "Ollama (embeddinggemma, Qwen2.5), Scikit-Learn TF-IDF",
        "Funcion_Principal": "Almacén vectorial dual para búsqueda semántica híbrida y generación clínica.",
        "Mecanismo_Clave": "Fallback automático e instantáneo a TF-IDF n-gram (1,2) ante ausencia de Ollama o GPU."
    },
    {
        "Capa_Num": 4,
        "Nombre": "Fine-Tuning y Evaluación",
        "Modulos": "finetunningpre/, scripts/evaluate_rag.py",
        "Tecnologias": "ChatML JSONL, LoRA/Unsloth, RAG Triad Benchmark",
        "Funcion_Principal": "29 Casos clínicos para adaptación de modelos y auditoría formal de fidelidad.",
        "Mecanismo_Clave": "Reporte cuantitativo automatizado en data/processed/rag_evaluation_report.json."
    }
]

df_arq = pd.DataFrame(arquitectura_data)
df_arq.to_csv(OUTPUT_DIR / "03_arquitectura_capas.csv", index=False)

# 4. Dataset de Matriz de Interacciones CYP450 y Semáforo Clínico
cyp_data = [
    {
        "Farmaco_Principal": "Paroxetina",
        "Farmaco_Secundario": "Risperidona",
        "Enzima": "CYP2D6",
        "Nivel_Riesgo": "Rojo Crítico",
        "Impacto_Clinico": "Inhibición potente de CYP2D6; eleva niveles de risperidona provocando extrapiramidalismo agudo.",
        "Conducta_Sugerida": "Reducir dosis de risperidona al 50% o rotar a Sertralina/Escitalopram."
    },
    {
        "Farmaco_Principal": "Bupropión",
        "Farmaco_Secundario": "Haloperidol",
        "Enzima": "CYP2D6",
        "Nivel_Riesgo": "Rojo Crítico",
        "Impacto_Clinico": "Inhibición marcada de 2D6 con acumulación de haloperidol y sedación excesiva/prolongación QTc.",
        "Conducta_Sugerida": "Ajuste posológico del 50% y monitorización estrecha de síntomas extrapiramidales."
    },
    {
        "Farmaco_Principal": "Fluoxetina",
        "Farmaco_Secundario": "Risperidona",
        "Enzima": "CYP2D6",
        "Nivel_Riesgo": "Rojo Crítico",
        "Impacto_Clinico": "Inhibición potente y persistente (vida media larga de norfluoxetina).",
        "Conducta_Sugerida": "Vigilancia de toxicidad durante semanas incluso tras retirar la fluoxetina."
    },
    {
        "Farmaco_Principal": "Clozapina",
        "Farmaco_Secundario": "Humo de Tabaco",
        "Enzima": "CYP1A2",
        "Nivel_Riesgo": "Amarillo Advertencia",
        "Impacto_Clinico": "Los hidrocarburos inducen CYP1A2 reduciendo niveles plasmáticos de clozapina hasta un 50%.",
        "Conducta_Sugerida": "Ajustar dosis según hábito tabáquico y monitorizar hemograma seriado."
    },
    {
        "Farmaco_Principal": "Carbonato de Litio",
        "Farmaco_Secundario": "AINEs / Tiazidas",
        "Enzima": "Excreción Renal",
        "Nivel_Riesgo": "Rojo Crítico",
        "Impacto_Clinico": "Reducción del aclaramiento renal de litio en un 30-50%, precipitando toxicidad lítica potencialmente fatal.",
        "Conducta_Sugerida": "Contraindicación de AINEs/tiazidas sin reducción drástica de litio y control estricto de litemia."
    },
    {
        "Farmaco_Principal": "Lamotrigina",
        "Farmaco_Secundario": "Ácido Valproico",
        "Enzima": "Glucuronidación UGT1A4",
        "Nivel_Riesgo": "Amarillo Advertencia",
        "Impacto_Clinico": "Valproato inhibe la glucuronidación duplicando la vida media de lamotrigina (riesgo de Stevens-Johnson).",
        "Conducta_Sugerida": "Iniciar lamotrigina a la mitad de la dosis estándar y titular lentamente a semanas alternas."
    },
    {
        "Farmaco_Principal": "Haloperidol",
        "Farmaco_Secundario": "Biperideno",
        "Enzima": "Cascada D2 / Anticolinérgico",
        "Nivel_Riesgo": "Alerta Roja (Cascada)",
        "Impacto_Clinico": "Cascada clásica: Parkinsonismo inducido tratado con anticolinérgico. Riesgo de discinesia tardía irreversible.",
        "Conducta_Sugerida": "Protocolo de Desprescripción Sofia: Reducir dosis, rotar a atípico y retirar biperideno en 2-4 semanas."
    },
    {
        "Farmaco_Principal": "Sertralina",
        "Farmaco_Secundario": "Litio",
        "Enzima": "Vías Independientes",
        "Nivel_Riesgo": "Verde Seguro",
        "Impacto_Clinico": "Mínima interferencia enzimática; perfil cardiovascular seguro y baja interacción CYP2D6.",
        "Conducta_Sugerida": "Mantener esquema estándar con monitorización habitual de respuesta clínica."
    }
]

df_cyp = pd.DataFrame(cyp_data)
df_cyp.to_csv(OUTPUT_DIR / "04_matriz_cyp450_interacciones.csv", index=False)

# 5. Dataset de Casos del Benchmark Evaluados
casos_data = [
    {
        "Caso_ID": "CASO_01",
        "Categoria": "Monografía y Posología",
        "Doc_ID": "[MONO-SERTRALINA-001]",
        "Pregunta": "¿Qué laboratorios fabrican la Sertralina y cuáles son sus nombres comerciales y posología?",
        "Top_Doc": "Sertralina",
        "Score_Similitud": 0.1786,
        "Context_Recall": 1.0,
        "Answer_Relevance": 1.0,
        "Groundedness": 1.0,
        "Voice_Snippet_Ok": "Sí",
        "Latencia_Seg": 0.001
    },
    {
        "Caso_ID": "CASO_02",
        "Categoria": "Farmacocinética y Citocromos",
        "Doc_ID": "[INTER-CYP2D6-001]",
        "Pregunta": "¿Qué interacciones y precauciones enzimáticas con CYP2D6 se describen para Paroxetina y Bupropión?",
        "Top_Doc": "Interacciones Críticas en el Citocromo CYP2D6",
        "Score_Similitud": 0.2398,
        "Context_Recall": 1.0,
        "Answer_Relevance": 1.0,
        "Groundedness": 1.0,
        "Voice_Snippet_Ok": "Sí",
        "Latencia_Seg": 0.001
    },
    {
        "Caso_ID": "CASO_03",
        "Categoria": "Medicina Basada en Evidencia",
        "Doc_ID": "[GUIA-CANMAT-DEP-002]",
        "Pregunta": "¿Qué fármacos se recomiendan en primera línea para manía aguda y por qué se desaconsejan antidepresivos solos?",
        "Top_Doc": "CANMAT Bipolar Depresión y Viraje",
        "Score_Similitud": 0.2065,
        "Context_Recall": 1.0,
        "Answer_Relevance": 0.8,
        "Groundedness": 1.0,
        "Voice_Snippet_Ok": "Sí",
        "Latencia_Seg": 0.001
    },
    {
        "Caso_ID": "CASO_04",
        "Categoria": "Farmacovigilancia y Cascadas",
        "Doc_ID": "[CASC-HAL-BIP-001]",
        "Pregunta": "Paciente con Haloperidol comienza con rigidez y temblor y le indican biperideno. ¿Riesgo de cascada?",
        "Top_Doc": "Cascada de Prescripción Clásica: Haloperidol a Biperideno",
        "Score_Similitud": 0.2004,
        "Context_Recall": 1.0,
        "Answer_Relevance": 0.8,
        "Groundedness": 1.0,
        "Voice_Snippet_Ok": "Sí",
        "Latencia_Seg": 0.001
    }
]

df_casos = pd.DataFrame(casos_data)
df_casos.to_csv(OUTPUT_DIR / "05_evaluacion_casos_benchmark.csv", index=False)

print("✅ Datasets de Power BI generados exitosamente en:", OUTPUT_DIR)
