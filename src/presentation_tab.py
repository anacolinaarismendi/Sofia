"""
presentation_tab.py
Módulo de Presentación Ejecutiva y Dashboard EDA para Sofia en Streamlit.
Permite exponer los 11 gráficos del análisis exploratorio, métricas de corpus,
farmacovigilancia y benchmark RAG de forma fluida y profesional.
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path


def load_benchmark_data():
    ruta = Path("data/processed/rag_evaluation_report.json")
    if not ruta.exists():
        return None
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            report = json.load(f)
        casos = report.get("detalle_por_caso", [])
        filas = []
        for c in casos:
            filas.append({
                "Caso ID": c.get("caso_id"),
                "Categoría": c.get("categoria"),
                "Pregunta": c.get("pregunta"),
                "Fuera Dominio": "Sí" if c.get("es_fuera_de_alcance") else "No",
                "Top Doc Recuperado": c.get("top_doc_id", "N/D"),
                "Recall@3": f"{c.get('context_recall@3', 0.0)*100:.0f}%",
                "MRR": f"{c.get('mrr', 0.0):.2f}",
                "Groundedness": f"{c.get('groundedness_score', 0.0)*100:.0f}%",
                "Cita [DOC-ID]": "Válida" if c.get("trazabilidad_etica_valida") else "No",
                "Latencia (ms)": f"{c.get('latencia_recuperacion_segundos', 0.0)*1000:.1f}"
            })
        return pd.DataFrame(filas)
    except Exception:
        return None


def render_presentation_tab():
    st.markdown("### 📊 Presentación Ejecutiva y Hallazgos del EDA")
    st.caption("Recorrido interactivo por la arquitectura del corpus, farmacovigilancia, citocromos CYP450, guías CANMAT y evaluación del motor RAG.")

    diapositivas = [
        "1. 🏛️ Resumen y Calidad del Corpus",
        "2. 💊 Cobertura Farmacológica y Fusión ATC",
        "3. 🧬 Seguridad Médica y Citocromos CYP450",
        "4. 📋 Directrices CANMAT y Normalización",
        "5. 🎯 Evaluación Formal del RAG (N=22)",
        "🎙️ Speech de Exposición (5 Minutos)"
    ]

    col_nav, _ = st.columns([4, 1])
    diapositiva_sel = col_nav.radio(
        "Navegación de Diapositivas:",
        options=diapositivas,
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # DIAPOSITIVA 1: Resumen y Calidad del Corpus
    # -------------------------------------------------------------------------
    if diapositiva_sel == diapositivas[0]:
        st.markdown("#### 1. Arquitectura del Corpus RAG e Integridad Estructural")
        st.markdown("**Pregunta clave:** *¿Qué contiene Sofia y cómo se garantiza la calidad de los datos sin fuga ni nulos?*")

        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
        kpi1.metric("Documentos Padre", "244", "0% nulos")
        kpi2.metric("Chunks Semánticos", "702", "100% < 600 tok")
        kpi3.metric("Fármacos Jufe", "57", "Nivel 5 OMS")
        kpi4.metric("Guías CANMAT", "67", "Recomendaciones")
        kpi5.metric("Eventos Jufe", "505", "40% graves")
        kpi6.metric("Eventos Stahl", "188", "38.3% graves")

        st.markdown("<br>", unsafe_allow_html=True)
        col_img1, col_img2 = st.columns([1.1, 1])
        with col_img1:
            st.markdown("**Gráfico 5: Composición del Corpus por Fuente y Tipo**")
            st.image("assets/eda_05_distribucion_fuentes_tipos.png", use_container_width=True)
            st.caption("CANMAT 31.6%, Marcas 27%, Jufe 23.4%, Stahl 9% y Pre-training 9%.")
        with col_img2:
            st.markdown("**Gráfico 1: Auditoría de Valores Nulos e Integridad**")
            st.image("assets/eda_01_mapa_nulos.png", use_container_width=True)
            st.caption("0% de valores nulos en campos mandatorios de todas las tablas estructuradas.")

        st.markdown("**Gráfico 6: Distribución de Tokens en Documentos Padre vs. Chunks Semánticos**")
        st.image("assets/eda_06_tokens_chunks_distribucion.png", use_container_width=True)
        st.success("✅ **Hallazgo de Ingeniería NLP:** Mediana de 68 tokens por chunk y máximo absoluto de 590 tokens. El 100% de los fragmentos se procesan sin truncamiento léxico en modelos de embeddings.")

        with st.expander("🎙️ Notas para la oradora (Diapositiva 1)"):
            st.markdown("""
            > *"El primer mandamiento de Sofia es el rigor: auditamos el 100% de las tablas con cero valores nulos.
            > Consolidamos 244 documentos padre balanceados entre guías de evidencia (CANMAT), práctica clínica (Jufe) y farmacocinética (Stahl).
            > A nivel de NLP, descompusimos el corpus en 702 chunks con una mediana de 68 tokens, garantizando que el 100% de los fragmentos caben holgadamente en la ventana de contexto sin pérdida de información médica."*
            """)

    # -------------------------------------------------------------------------
    # DIAPOSITIVA 2: Cobertura Farmacológica y Fusión ATC
    # -------------------------------------------------------------------------
    elif diapositiva_sel == diapositivas[1]:
        st.markdown("#### 2. Cobertura Farmacológica ATC (OMS) y Fusión Multifuente")
        st.markdown("**Pregunta clave:** *¿Cómo unifica Sofia tres fuentes heterogéneas bajo un marco canónico común?*")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Núcleo Común", "19 Fármacos", "Compartidos por 3 fuentes")
        kpi2.metric("Inclusión de Stahl", "100%", "22 de 22 en Jufe")
        kpi3.metric("CANMAT Exclusivo", "1 Fármaco", "Armodafinilo")
        kpi4.metric("Mediana Eventos IQR", "8 eventos/fármaco", "Homogéneo")

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown("**Gráfico 7: Cobertura por Familia Terapéutica ATC (OMS)**")
            st.image("assets/eda_07_cobertura_atc_familias.png", use_container_width=True)
            st.caption("Predominio de Antidepresivos (N06A), Antipsicóticos (N05A) y Estabilizadores (N03A).")
        with col_img2:
            st.markdown("**Gráfico 8: Solapamiento Interfuente (Diagrama de Venn 3-Way)**")
            st.image("assets/eda_08_venn_coocurrencia_fuentes.png", use_container_width=True)
            st.caption("Núcleo común de 19 psicofármacos esenciales y 100% de Stahl contenido en Jufe.")

        st.markdown("**Gráfico 2: Detección de Outliers en Farmacovigilancia (IQR sobre Jufe)**")
        st.image("assets/eda_02_iqr_efectos_adversos.png", use_container_width=True)
        st.info("🔍 **Criterio Clínico Estricto:** Fármacos multirreceptoriales como Clozapina (24 eventos) y Haloperidol (21 eventos) sobrepasan el límite matemático IQR pero se mantuvieron al 100% para evitar silencios iatrogénicos en seguridad.")

        with st.expander("🎙️ Notas para la oradora (Diapositiva 2)"):
            st.markdown("""
            > *"Estandarizamos toda la base bajo la clasificación ATC oficial de la OMS.
            > Mediante análisis de solapamiento trifásico identificamos un núcleo común de 19 moléculas esenciales compartidas por Jufe, Stahl y CANMAT.
            > En el análisis de outliers por IQR, la mediana fue de 8 efectos por fármaco. Aunque Clozapina y Haloperidol superaron el límite estadístico, aplicamos criterio clínico de no exclusión: en farmacovigilancia, un valor atípico representa un perfil toxicológico que el médico debe conocer."*
            """)

    # -------------------------------------------------------------------------
    # DIAPOSITIVA 3: Seguridad Médica y Citocromos CYP450
    # -------------------------------------------------------------------------
    elif diapositiva_sel == diapositivas[2]:
        st.markdown("#### 3. Farmacovigilancia, Toxicidad y Matriz Enzimática CYP450")
        st.markdown("**Pregunta clave:** *¿Qué interacciones metabólicas complejas y toxicidades previene el sistema?*")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Eventos Jufe", "505", "40.0% graves")
        kpi2.metric("Eventos Stahl", "188", "38.3% graves")
        kpi3.metric("Variables Enzimáticas", "15 flags", "Inhibe / Sustrato / Induce")
        kpi4.metric("Inhibidores 2D6 Potentes", "3 fármacos", "Fluox, Parox, Buprop")

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown("**Gráfico 9: Severidad de Eventos Adversos (Jufe vs. Stahl)**")
            st.image("assets/eda_09_severidad_eventos_adversos.png", use_container_width=True)
            st.caption("Consistencia metodológica: ambas fuentes reflejan ~39%-40% de toxicidad severa frente a síntomas leves.")
        with col_img2:
            st.markdown("**Gráfico 10: Matriz de Interacciones Enzimáticas CYP450 (Stahl)**")
            st.image("assets/eda_10_matriz_cyp450_stahl.png", use_container_width=True)
            st.caption("Mapeo de 15 variables de citocromos (CYP1A2, 2C9, 2C19, 2D6, 3A4) en 22 psicofármacos.")

        st.warning("⚠️ **Prevención de Cascadas de Prescripción:** Sofia detecta colisiones críticas: inhibición potente de CYP2D6 frente a Risperidona/Haloperidol, duplicación de Lamotrigina por Ácido Valproico, y la cascada clásica Haloperidol ➔ Parkinsonismo ➔ Biperideno.")

        with st.expander("🎙️ Notas para la oradora (Diapositiva 3)"):
            st.markdown("""
            > *"Clasificamos más de 690 eventos adversos bajo vocabulario canónico. Tanto Jufe como Stahl coinciden de forma consistente en que ~40% corresponden a eventos graves o de riesgo vital.
            > Nuestra matriz CYP450 cruza 15 variables enzimáticas para prevenir colisiones críticas: inhibición potente de 2D6 por Paroxetina/Fluoxetina, inducción masiva por Carbamazepina y cascadas donde un efecto adverso se confunde con una nueva patología."*
            """)

    # -------------------------------------------------------------------------
    # DIAPOSITIVA 4: Directrices CANMAT y Normalización
    # -------------------------------------------------------------------------
    elif diapositiva_sel == diapositivas[3]:
        st.markdown("#### 4. Directrices CANMAT 2018 y Estabilización Textual")
        st.markdown("**Pregunta clave:** *¿Cómo se estructura la evidencia clínica oficial y cómo se normalizó la varianza textual?*")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Guías Bipolar", "38 recomendaciones", "Manía, depresión y mant.")
        kpi2.metric("Guías TDM", "29 recomendaciones", "Aguda y resistencia")
        kpi3.metric("Skewness log1p", "2.22 ➔ 0.29", "Varianza estabilizada")
        kpi4.metric("Correlación char vs tok", "r = 0.99", "Lineal perfecta")

        st.markdown("**Gráfico 11: Directrices CANMAT 2018 por Patología y Nivel de Evidencia**")
        st.image("assets/eda_11_directrices_canmat_evidencia.png", use_container_width=True)
        st.caption("Predominio de Nivel 1 en primeras líneas; formalización de alertas negativas Level 1 (prohibición de antidepresivos en monoterapia en depresión bipolar).")

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown("**Gráfico 3: Distribución Textual y log1p**")
            st.image("assets/eda_03_distribucion_longitud_log.png", use_container_width=True)
            st.caption("Estabilización de varianza mediante transformación logarítmica log(1+x).")
        with col_img2:
            st.markdown("**Gráfico 4: Matriz de Correlación de Métricas Cuantitativas**")
            st.image("assets/eda_04_matriz_correlacion.png", use_container_width=True)
            st.caption("Validación de la heurística tokens ≈ caracteres / 4 (r=0.99) sin colinealidades espurias.")

        with st.expander("🎙️ Notas para la oradora (Diapositiva 4)"):
            st.markdown("""
            > *"Tabulamos 67 recomendaciones oficiales de CANMAT con predominio de evidencia Nivel 1 en fases agudas y formalización de alertas de daño (evidencia Level 1 negativa para antidepresivos solos en depresión bipolar).
            > A nivel estadístico, la longitud de documentos presentaba un skewness de 2.22 que estabilizamos a 0.29 con log1p, confirmando además una correlación lineal r=0.99 entre longitud y tokens sin sesgos ocultos."*
            """)

    # -------------------------------------------------------------------------
    # DIAPOSITIVA 5: Evaluación Formal del RAG (Benchmark N=22)
    # -------------------------------------------------------------------------
    elif diapositiva_sel == diapositivas[4]:
        st.markdown("#### 5. Evaluación Formal del Motor RAG (Benchmark N=22)")
        st.markdown("**Pregunta clave:** *¿Qué métricas formales respaldan la precisión, fidelidad y control de alucinaciones?*")

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Context Recall@3", "100.0%", "Top 3 recuperación")
        kpi2.metric("MRR", "1.000", "Ranking óptimo")
        kpi3.metric("Groundedness", "95.5%", "Fidelidad factual")
        kpi4.metric("Tasa Abstención", "100.0%", "Consultas fuera dominio")
        kpi5.metric("Latencia Media", "1.6 ms", "Recuperación léxica")

        st.success("🎯 **Trazabilidad Ética y Citas [DOC-ID]:** En el 100% de los casos evaluados, cada recomendación clínica cita documentos válidos recuperados del contexto, eliminando alucinaciones médicas.")

        df_bench = load_benchmark_data()
        if df_bench is not None:
            st.markdown("##### 📋 Auditoría Detallada de los 22 Casos Clínicos Centinela")
            st.dataframe(df_bench, use_container_width=True, hide_index=True)
        else:
            st.info("El reporte completo de benchmark está disponible en data/processed/rag_evaluation_report.json.")

        with st.expander("🎙️ Notas para la oradora (Diapositiva 5)"):
            st.markdown("""
            > *"El sistema se validó mediante un benchmark formal de 22 escenarios centinela estructurados por especialistas.
            > Los resultados son concluyentes: 100% de Recall en top 3, MRR de 1.0, 95.5% de fidelidad factual y 100% de citas verificables con identificador [DOC-ID].
            > Ante preguntas fuera de dominio médico, Sofia se abstiene con un 100% de seguridad en lugar de alucinar."*
            """)

    # -------------------------------------------------------------------------
    # PESTAÑA EXTRA: Speech Completo de Exposición (5 Minutos)
    # -------------------------------------------------------------------------
    elif diapositiva_sel == diapositivas[5]:
        st.markdown("#### 🎙️ Guion Completo de Exposición Oral (5 Minutos)")
        st.caption("Guion con acotaciones visuales, tiempos y énfasis para ensayar o leer en tu presentación.")

        st.markdown("""
        ### ⏱️ Minuto 0:00 – 0:45 | Introducción y el Problema
        > *"Buenos días a todos. En psiquiatría y salud mental, una decisión farmacológica errónea puede provocar una urgencia médica o una complicación iatrogénica grave. Sin embargo, cuando los clínicos intentan consultar modelos de inteligencia artificial comercial, se enfrentan a un riesgo inaceptable: **las alucinaciones médicas**.
        > Para resolver este problema creamos **Sofia: un asistente clínico y sistema RAG especializado en psicofarmacología basada en la evidencia**. Sofia no improvisa: opera bajo una política estricta de Grounding, respondiendo únicamente a partir de fuentes canónicas consolidadas y trazando cada afirmación con identificadores documentales auditables."*

        ---

        ### ⏱️ Minuto 0:45 – 1:45 | Diapositiva 1: Corpus e Integridad
        > *"El primer mandamiento de Sofia es el rigor: auditamos el 100% de las tablas con **cero valores nulos**. Consolidamos **244 documentos padre** balanceados entre las guías CANMAT (31.6%), el catálogo de equivalencias comerciales (27%), las monografías clínicas de Jufe (23.4%) y la farmacocinética de Stahl.
        > A nivel de procesamiento de lenguaje natural, descompusimos estos documentos en **702 fragmentos semánticos o chunks**. Como muestra la distribución, la mediana se sitúa en apenas **68 tokens** y el valor máximo absoluto en **590 tokens**. Esto significa que el 100% de los fragmentos caben holgadamente dentro de la ventana de contexto de los modelos de embeddings, eliminando por completo cualquier truncamiento léxico."*

        ---

        ### ⏱️ Minuto 1:45 – 2:45 | Diapositiva 2: Cobertura ATC y Fusión Multifuente
        > *"Estandarizamos toda la base farmacológica mediante la clasificación oficial **ATC de la Organización Mundial de la Salud**. En nuestro gráfico observamos cobertura de 20 antidepresivos, 15 antipsicóticos y 8 estabilizadores del ánimo.
        > Mediante análisis de solapamiento trifásico identificamos un **núcleo común de 19 psicofármacos esenciales** —como Litio, Clozapina, Haloperidol o Sertralina— compartidos por Jufe, Stahl y CANMAT. Además, el 100% de los fármacos con perfil cinético de Stahl están contenidos en Jufe.
        > En la detección de valores atípicos mediante el método IQR, observamos una mediana de 8 eventos por fármaco. Aunque Clozapina y Haloperidol superaron el límite estadístico con más de 20 eventos, aplicamos un **criterio clínico estricto de no exclusión**: en farmacovigilancia, un dato atípico no es basura estadística, es un perfil toxicológico que el médico necesita conocer."*

        ---

        ### ⏱️ Minuto 2:45 – 3:45 | Diapositiva 3: Seguridad y Citocromos CYP450
        > *"Clasificamos más de 690 eventos adversos bajo vocabulario canónico. Tanto Jufe como Stahl coinciden de forma consistente en que aproximadamente un **40% de los eventos reportados corresponden a toxicidad severa o de riesgo vital**, lo que valida la coherencia metodológica de los extractores.
        > A la derecha observamos la **Matriz Enzimática del Citocromo P450**: Sofia cruza 15 variables enzimáticas para prevenir colisiones críticas en urgencias: por ejemplo, la potente inhibición de CYP2D6 por Fluoxetina o Paroxetina cuando se coadministra con Risperidona, o el riesgo de duplicación plasmática de Lamotrigina al interactuar con Ácido Valproico. Además, Sofia incorpora un detector determinista de **cascadas de prescripción** para alertar cuando un síntoma nuevo es en realidad un efecto adverso mal manejado."*

        ---

        ### ⏱️ Minuto 3:45 – 4:30 | Diapositiva 4: Guías CANMAT y Normalización
        > *"Tabulamos **67 directrices de las guías CANMAT e ISBD 2018**. Observamos el predominio de evidencia de **Nivel 1** en primeras líneas, así como alertas de daño de máximo nivel que prohíben, por ejemplo, el uso de antidepresivos en monoterapia en depresión bipolar por riesgo inminente de viraje maníaco.
        > A nivel estadístico, la longitud de documentos presentaba un sesgo pronunciado de 2.22 que estabilizamos a 0.29 con la transformación log1p, confirmando una correlación lineal casi perfecta (r=0.99) entre longitud y tokens sin colinealidades espurias."*

        ---

        ### ⏱️ Minuto 4:30 – 5:00 | Diapositiva 5: Benchmark RAG y Cierre
        > *"Todo este pipeline de datos se sometió a un **benchmark formal sobre 22 escenarios clínicos estructurados**:
        > - **100% de Context Recall@3**: en todos los casos el documento médico exacto se recuperó en el top 3.
        > - **MRR de 1.0**: ordenación óptima de búsqueda.
        > - **95.5% de Fidelidad factual**: garantizando respuestas sin alucinaciones.
        > - **100% de Tasa de Abstención**: declarando su límite ante consultas fuera de dominio médico.
        > 
        > En conclusión: Sofia no es un chat improvisado; es un sistema robusto de soporte clínico donde cada número, cada chunk y cada interacción ha sido auditado desde la ciencia de datos y la medicina basada en la evidencia. Muchas gracias."*
        """)
