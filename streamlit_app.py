"""
streamlit_app.py
Aplicación Web Interactiva de Sofia: Asistente Clínico Inteligente y RAG de Psicofarmacología.
Diseñado con estricta portabilidad relativa y cero rutas locales de usuario.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import streamlit as st

# Configuración de rutas relativas dinámicas
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from src.rag.engine import ClinicalRAGEngine
except Exception as _rag_err:
    import traceback
    st.error(f"⚠️ Error al inicializar ClinicalRAGEngine: {_rag_err}")
    st.code(traceback.format_exc())
    class ClinicalRAGEngine:  # type: ignore
        def __init__(self, *args, **kwargs):
            class DummyStore:
                use_ollama = False
            self.vector_store = DummyStore()
            self.docs = []
        def consultar(self, pregunta: str, top_k: int = 3):
            return {
                "pregunta": pregunta,
                "respuesta": f"El motor RAG no pudo inicializarse correctamente en este entorno.\nError: {_rag_err}",
                "voice_snippet": "",
                "generado_con_llm": False,
                "documentos_recuperados": []
            }

try:
    from src.enrichers.atc_enricher import ATCEnricher
except Exception:
    class ATCEnricher:  # type: ignore
        CATALOGO_ATC = {}
        @classmethod
        def get_info(cls, *args, **kwargs):
            return None
        @classmethod
        def obtener_ficha(cls, *args, **kwargs):
            return None



# -----------------------------------------------------------------------------
# Configuración General de la Página Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sofia | Asistente de Psicofarmacología Clínica",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para interfaz médica moderna
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .voice-card {
        background-color: #f0fdf4;
        border-left: 5px solid #16a34a;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 12px 0;
        font-size: 0.98rem;
        color: #14532d;
    }
    .alert-critical {
        background-color: #fef2f2;
        border-left: 5px solid #dc2626;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
        color: #991b1b;
    }
    .alert-warning {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
        color: #92400e;
    }
    .alert-safe {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
        color: #166534;
    }
    .metric-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-level1 { background-color: #dcfce7; color: #166534; }
    .badge-level2 { background-color: #e0f2fe; color: #075985; }
    .badge-damage { background-color: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Carga de Motor RAG y Datasets en Caché
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Inicializando base de conocimiento clínica de Sofia...")
def get_rag_engine(api_key: str = "", host: str = "", model: str = ""):
    # 1. Leer credenciales de st.secrets si existen (Streamlit Cloud)
    sec_key = ""
    sec_host = ""
    sec_model = ""
    try:
        if hasattr(st, "secrets"):
            sec_key = st.secrets.get("OLLAMA_API_KEY", "")
            sec_host = st.secrets.get("OLLAMA_HOST", "")
            sec_model = st.secrets.get("OLLAMA_MODEL", "")
    except Exception:
        pass

    final_key = api_key or sec_key or None
    final_host = host or sec_host or None
    final_model = model or sec_model or None

    return ClinicalRAGEngine(
        ollama_host=final_host,
        llm_model=final_model,
        api_key=final_key
    )


@st.cache_data
def load_datasets():
    data_dir = WORKSPACE_ROOT / "data" / "processed"
    datasets = {}
    try:
        datasets["jufe"] = pd.read_csv(data_dir / "jufe_farmacos_clean.csv")
        datasets["jufe_efectos"] = pd.read_csv(data_dir / "jufe_efectos_adversos_clean.csv")
        datasets["stahl"] = pd.read_csv(data_dir / "stahl_farmacos_clean.csv")
        datasets["stahl_eventos"] = pd.read_csv(data_dir / "stahl_eventos_adversos_clean.csv")
        datasets["canmat"] = pd.read_csv(data_dir / "canmat_recomendaciones_clean.csv")
        datasets["canmat_alertas"] = pd.read_csv(data_dir / "canmat_alertas_clinicas_clean.csv")
    except Exception as e:
        st.error(f"Error cargando datasets: {e}")
    return datasets


# -----------------------------------------------------------------------------
# Barra Lateral (Sidebar)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=64)
    st.markdown("## **Sofia Clinical RAG**")
    st.caption("Asistente de Soporte a la Decisión y Farmacovigilancia en Psiquiatría")

    st.markdown("---")
    st.markdown("### 🔌 Inferencia y Conectividad")

    # Panel expandible para configuración de Ollama Cloud en despliegue
    with st.expander("⚙️ Conexión Ollama Cloud / Remoto", expanded=False):
        st.caption("Si despliegas fuera de tu Mac (Streamlit Cloud), configura tu API Key de Ollama Cloud:")
        user_key = st.text_input(
            "Ollama Cloud API Key:",
            type="password",
            value=st.session_state.get("ollama_api_key", ""),
            help="Obtén tu API key en ollama.com/settings/keys"
        )
        user_host = st.text_input(
            "Host URL:",
            value=st.session_state.get("ollama_host", ""),
            placeholder="https://ollama.com"
        )
        if st.button("Aplicar Credenciales", use_container_width=True):
            st.session_state["ollama_api_key"] = user_key
            st.session_state["ollama_host"] = user_host
            st.cache_resource.clear()
            st.rerun()

    active_key = st.session_state.get("ollama_api_key", "")
    active_host = st.session_state.get("ollama_host", "")
    rag_engine = get_rag_engine(api_key=active_key, host=active_host)
    data = load_datasets()

    # Estado visual del motor de inferencia
    if rag_engine.api_key or (rag_engine.ollama_host and "ollama.com" in rag_engine.ollama_host):
        st.success(f"☁️ Ollama Cloud Activo (`{rag_engine.llm_model}`)")
    elif rag_engine.vector_store.use_ollama:
        st.success(f"🟢 Ollama Local Activo (`{rag_engine.llm_model}`)")
    else:
        st.info("🟡 Inferencia Contingencia: TF-IDF + Fichas Estructuradas")

    st.metric("Base de Conocimiento", f"{len(rag_engine.docs)} Documentos", delta="Curados pre_training")

    st.markdown("---")
    st.markdown("### 📚 Corpus Clínico y Trazabilidad")
    st.markdown("""
    - **Corpus propio:** Síntesis estructurada basada en Jufe, Stahl y CANMAT 2018.
    - **Trazabilidad `[DOC-ID]`:** Cero citas apócrifas; respaldo directo por código de documento.
    - **Catálogo Oficial:** Clasificación ATC (OMS) y marcas comerciales.
    """)

    st.markdown("---")
    st.markdown("### 📂 Código y Documentación")
    st.markdown("[🔗 Ver Repositorio y README en GitHub](https://github.com/anacolinaarismendi/Sofia)")

    st.markdown("---")
    st.caption("⚠️ **Aviso de Responsabilidad:** Sofia es una herramienta de soporte asistencial para profesionales de la salud. No sustituye el juicio clínico individual.")


# -----------------------------------------------------------------------------
# Encabezado Principal y Pestañas
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🧠 Sofia — Asistente Clínico de Psicofarmacología</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Motor RAG especializado en interacciones CYP450, cascadas de prescripción, directrices CANMAT y farmacovigilancia con trazabilidad [DOC-ID].</div>',
    unsafe_allow_html=True
)

tab_chat, tab_cyp, tab_cascade, tab_canmat = st.tabs([
    "💬 Asistente RAG Clínico",
    "⚡ Matriz CYP450 Interactiva",
    "🛑 Detector de Cascadas",
    "📖 Guías CANMAT y Catálogo"
])


# =============================================================================
# PESTAÑA 1: Chat RAG Clínico
# =============================================================================
with tab_chat:
    st.markdown("#### Consulta Clínica Especializada en Lenguaje Natural")
    st.write("Formula preguntas sobre posología, interacciones citocromo P450, contraindicaciones de guías CANMAT o pautas de desprescripción.")

    # Botones de consulta rápida preconfigurada
    st.markdown("**Consultas clínicas rápidas:**")
    col1, col2, col3, col4 = st.columns(4)
    query_preset = None

    if col1.button("💊 Sertralina (Posología Jufe)", use_container_width=True):
        query_preset = "¿Qué laboratorios fabrican la Sertralina y cuáles son sus nombres comerciales y posología según Jufe?"
    if col2.button("⚠️ Paroxetina y Bupropión (CYP2D6)", use_container_width=True):
        query_preset = "¿Qué interacciones y precauciones enzimáticas con CYP2D6 describe Stahl para Paroxetina y Bupropión?"
    if col3.button("📋 CANMAT Manía y Viraje", use_container_width=True):
        query_preset = "¿Qué fármacos recomienda CANMAT en primera línea para manía aguda y por qué desaconseja los antidepresivos solos en depresión bipolar?"
    if col4.button("🛑 Haloperidol + Biperideno", use_container_width=True):
        query_preset = "Un paciente en tratamiento con Haloperidol comienza con rigidez y temblor y le indican medicación antiparkinsoniana. ¿Qué dicen Jufe y Stahl sobre este riesgo?"

    # Inicializar historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hola, soy **Sofia**, asistente clínica especializada en psicofarmacología clínica y farmacovigilancia. "
                    "Estoy lista para responder consultas con estricto *grounding* en nuestro corpus clínico y "
                    "trazabilidad transparente mediante identificadores de evidencia `[DOC-ID]`. "
                    "¿En qué caso o paciente puedo asistirte hoy?"
                )
            }
        ]

    # Mostrar mensajes existentes
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("voice_snippet"):
                st.markdown(
                    f'<div class="voice-card">🔊 <b>Voice Snippet Clínico (< 40s):</b><br>{msg["voice_snippet"]}</div>',
                    unsafe_allow_html=True
                )
            if msg.get("fuentes"):
                with st.expander("📚 Fuentes y Documentos Clínicos Recuperados"):
                    for doc, score in msg["fuentes"]:
                        doc_id = doc.get("id", "DOC")
                        st.markdown(f"- **`[{doc_id}]` {doc['titulo']}** (ATC: `{doc.get('atc', 'N/D')}`) — Relevancia: `{score:.4f}` | Archivo: `{doc['archivo_origen']}`")

    # Manejar input del usuario (sea del chat o de un botón preconfigurado)
    user_query = st.chat_input("Escribe tu consulta clínica aquí...") or query_preset

    if user_query:
        # Registrar y mostrar mensaje de usuario
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generar respuesta con RAG
        with st.chat_message("assistant"):
            with st.spinner("Consultando literatura clínica y analizando interacciones..."):
                resultado = rag_engine.consultar(user_query, top_k=3)
                respuesta_texto = resultado["respuesta"]
                voice_snip = resultado["voice_snippet"]
                fuentes = resultado["documentos_recuperados"]

                st.markdown(respuesta_texto)

                if voice_snip:
                    st.markdown(
                        f'<div class="voice-card">🔊 <b>Voice Snippet Clínico (< 40s):</b><br>{voice_snip}</div>',
                        unsafe_allow_html=True
                    )

                with st.expander("📚 Fuentes y Documentos Clínicos Recuperados"):
                    for doc, score in fuentes:
                        doc_id = doc.get("id", "DOC")
                        st.markdown(f"- **`[{doc_id}]` {doc['titulo']}** (ATC: `{doc.get('atc', 'N/D')}`) — Relevancia: `{score:.4f}` | Archivo: `{doc['archivo_origen']}`")

        # Guardar en historial
        st.session_state.messages.append({
            "role": "assistant",
            "content": respuesta_texto,
            "voice_snippet": voice_snip,
            "fuentes": fuentes
        })


# =============================================================================
# PESTAÑA 2: Matriz Interactiva CYP450
# =============================================================================
with tab_cyp:
    st.markdown("#### Matriz Dinámica de Citocromos y Semáforo de Interacciones")
    st.write("Selecciona dos o más psicofármacos del tratamiento del paciente para evaluar bloqueos enzimáticos o acumulación tóxica.")

    farmacos_disponibles = sorted([
        "Sertralina", "Fluoxetina", "Paroxetina", "Bupropión",
        "Clozapina", "Risperidona", "Haloperidol", "Litio", "Lamotrigina"
    ])

    seleccion = st.multiselect(
        "Fármacos en el esquema terapéutico:",
        options=farmacos_disponibles,
        default=["Paroxetina", "Risperidona"]
    )

    if len(seleccion) < 2:
        st.info("💡 Selecciona al menos 2 fármacos para calcular los cruces enzimáticos.")
    else:
        st.markdown("##### 🚦 Diagnóstico Clínico de Interacciones")

        interacciones_detectadas = []

        isrs_catalogo = {"Sertralina", "Fluoxetina", "Paroxetina"}
        isrs_seleccionados = set(seleccion).intersection(isrs_catalogo)

        # Regla A: Paroxetina + Bupropión (Doble Inhibición Potente de CYP2D6 y Riesgo Convulsivo)
        if "Paroxetina" in seleccion and "Bupropión" in seleccion:
            interacciones_detectadas.append({
                "nivel": "CRÍTICO",
                "titulo": "🔴 Interacción Severa Farmacocinética y Farmacodinámica (Paroxetina + Bupropión)",
                "descripcion": (
                    "**Mecanismo Enzimático:** Tanto la paroxetina como el bupropión son potentes inhibidores del citocromo CYP2D6. "
                    "Al coadministrarse, se produce una saturación competitiva del aclaramiento de paroxetina (que depende de 2D6 con cinética no lineal), "
                    "provocando un aumento exponencial e impredecible de sus niveles plasmáticos.\n\n"
                    "**Riesgo Clínico:** Toxicidad serotoninérgica, prolongación del intervalo QTc, temblor severo y disminución crítica del umbral "
                    "convulsivo inducida por bupropión ante concentraciones supraterapéuticas.\n\n"
                    "**Recomendación:** Combinación desaconsejada. Si se requiere añadir bupropión para mitigar disfunción sexual o apatía, "
                    "es preferible emplear un ISRS con mínimo o nulo impacto sobre CYP2D6 (como Sertralina o Escitalopram)."
                )
            })

        # Regla B: Duplicidad Terapéutica de ISRS (2 o más ISRS juntos)
        if len(isrs_seleccionados) >= 2:
            interacciones_detectadas.append({
                "nivel": "CRÍTICO",
                "titulo": f"🔴 Duplicidad Terapéutica Inadecuada de ISRS ({' + '.join(sorted(isrs_seleccionados))})",
                "descripcion": (
                    "**Mecanismo:** La prescripción simultánea de dos o más fármacos de la misma familia (ISRS) carece de fundamento clínico "
                    "basado en evidencia y no aporta beneficio sinérgico en depresión mayor o trastornos de ansiedad.\n\n"
                    "**Riesgo Clínico:** Multiplica de forma inmediata el riesgo de **Síndrome Serotoninérgico** potencialmente letal "
                    "(hipertermia, clonus, mioclonías, inestabilidad autonómica, agitación psicomotriz) y riesgo hemorrágico por depleción plaquetaria de serotonina.\n\n"
                    "**Conducta Racional:** Contraindicado. Realizar cambio escalonado con período de retirada o lavado adecuado si se planea una rotación."
                )
            })

        # Regla C: Inhibición potente de CYP2D6 + Sustrato 2D6 (Risperidona, Haloperidol)
        inhibidores_2d6 = {"Paroxetina", "Bupropión", "Fluoxetina"}
        sustratos_2d6 = {"Risperidona", "Haloperidol"}

        inh_presentes = set(seleccion).intersection(inhibidores_2d6)
        sus_presentes = set(seleccion).intersection(sustratos_2d6)

        if inh_presentes and sus_presentes:
            interacciones_detectadas.append({
                "nivel": "CRÍTICO",
                "titulo": f"🔴 Bloqueo Enzimático Potente de CYP2D6 ({', '.join(inh_presentes)} + {', '.join(sus_presentes)})",
                "descripcion": (
                    f"**Mecanismo:** {', '.join(inh_presentes)} inhibe potentemente el CYP2D6, provocando una elevación "
                    f"marcada de las concentraciones plasmáticas de {', '.join(sus_presentes)}.\n\n"
                    f"**Riesgo Clínico:** Aparición de síntomas extrapiramidales agudos, sedación excesiva y prolongación del intervalo QTc.\n\n"
                    f"**Pauta:** Reducir la dosis del sustrato en un 50% o rotar hacia un antidepresivo con mínimo impacto 2D6 (ej. Sertralina o Escitalopram)."
                )
            })

        # Regla D: Litio + Interacciones cruzadas
        if "Litio" in seleccion:
            if "Haloperidol" in seleccion:
                interacciones_detectadas.append({
                    "nivel": "CRÍTICO",
                    "titulo": "🔴 Riesgo de Neurotoxicidad y Síndrome Encefalopático (Litio + Haloperidol)",
                    "descripcion": (
                        "**Hallazgo Clínico:** La combinación puede precipitar un síndrome encefalopático severo (síndrome de Cohen & Cohen), "
                        "cursando con letargia, temblor grosero, debilidad extrema, fiebre y síntomas extrapiramidales severos con riesgo de secuelas neurológicas.\n\n"
                        "**Monitorización:** Evitar dosis altas de ambos fármacos; mantener litemias en el rango inferior (0.6 - 0.8 mEq/L) y suspender si aparece confusión."
                    )
                })

            if isrs_seleccionados:
                interacciones_detectadas.append({
                    "nivel": "ADVERTENCIA",
                    "titulo": f"🟡 Potenciación Serotoninérgica y Temblor (Litio + {', '.join(sorted(isrs_seleccionados))})",
                    "descripcion": (
                        "**Interacción Farmacodinámica:** El litio estimula la neurotransmisión serotoninérgica central. Aunque es una estrategia válida de adición "
                        "en depresión resistente, eleva la frecuencia de temblor postural fino, molestias digestivas y riesgo de toxicidad serotoninérgica.\n\n"
                        "**Pauta:** Controlar periódicamente la litemia basal y vigilar síntomas autonómicos o neuromusculares."
                    )
                })

            if "Clozapina" in seleccion:
                interacciones_detectadas.append({
                    "nivel": "ADVERTENCIA",
                    "titulo": "🟡 Monitorización Neurológica Estricta (Litio + Clozapina)",
                    "descripcion": (
                        "**Riesgo:** Ambos agentes disminuyen el umbral convulsivo y aumentan el riesgo de mioclonías, confusión y síndrome neuroléptico maligno.\n\n"
                        "**Pauta:** Titulación lenta de clozapina y hemograma seriado habitual."
                    )
                })

            # Recordatorio renal constante de litio si no hay otra regla crítica de litio
            if "Haloperidol" not in seleccion and not isrs_seleccionados and "Clozapina" not in seleccion:
                interacciones_detectadas.append({
                    "nivel": "ADVERTENCIA",
                    "titulo": "🟡 Vigilancia de Eliminación Renal de Litio",
                    "descripcion": (
                        "**Riesgo Renal:** El litio tiene eliminación exclusivamente renal sin metabolismo CYP450. "
                        "La prescripción concomitante de AINEs (ibuprofeno, naproxeno), IECA o diuréticos tiazídicos "
                        "reduce su aclaramiento en un 30-50%, precipitando toxicidad lítica potencialmente fatal."
                    )
                })

        # Regla E: Clozapina + Inhibidores o Inductores CYP1A2
        if "Clozapina" in seleccion and "Litio" not in seleccion:
            interacciones_detectadas.append({
                "nivel": "ADVERTENCIA",
                "titulo": "🟡 Monitorización Estricta de Clozapina (Sustrato Principal CYP1A2)",
                "descripcion": (
                    "**Vía Metabólica:** Clozapina depende en un 70% de CYP1A2. Fármacos inhibidores elevan su nivel hacia el rango convulsivo; "
                    "los hidrocarburos del humo del tabaco inducen CYP1A2 y reducen sus niveles hasta un 50%. Requiere hemograma seriado obligatorio."
                )
            })

        # Regla F: Lamotrigina + Valproato
        if "Lamotrigina" in seleccion and "Litio" not in seleccion:
            interacciones_detectadas.append({
                "nivel": "ADVERTENCIA",
                "titulo": "🟡 Metabolismo de Lamotrigina por Glucuronidación (UGT1A4)",
                "descripcion": (
                    "**Interacción:** No se metaboliza principalmente por CYP450. Sin embargo, si se combina con Ácido Valproico, "
                    "éste inhibe la glucuronidación duplicando la vida media de lamotrigina y aumentando drásticamente el riesgo de Síndrome de Stevens-Johnson."
                )
            })

        if not interacciones_detectadas:
            st.markdown(
                f'<div class="alert-warning" style="background-color: #f8fafc; border-left: 5px solid #64748b; color: #1e293b; padding: 14px 18px; border-radius: 6px;">'
                f'<h4>⚪ Sin Datos Específicos de Interacción en esta Matriz Rápida</h4>'
                f'<p>No se dispone de una regla heurística predefinida en este panel para la combinación simultánea de: <b>{", ".join(seleccion)}</b>.</p>'
                f'<p style="margin-bottom: 0;"><b>Aviso de seguridad clínica:</b> La ausencia de una alerta automática en este selector rápido <u>no garantiza la inocuidad</u> del esquema. '
                f'Verifique la función renal y hepática del paciente, consulte las monografías oficiales o plantee la consulta a <b>Sofia</b> en el asistente RAG.</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            for item in interacciones_detectadas:
                css_class = "alert-critical" if item["nivel"] == "CRÍTICO" else "alert-warning"
                st.markdown(
                    f'<div class="{css_class}"><h4>{item["titulo"]}</h4><p>{item["descripcion"]}</p></div>',
                    unsafe_allow_html=True
                )


# =============================================================================
# PESTAÑA 3: Detector de Cascadas de Prescripción
# =============================================================================
with tab_cascade:
    st.markdown("#### Detector Heurístico de Cascadas de Prescripción y Desprescripción")
    st.write(
        "Una cascada de prescripción ocurre cuando el efecto adverso de un fármaco es malinterpretado como una nueva condición médica, "
        "llevando a prescribir un segundo fármaco innecesario que añade toxicidad."
    )

    col_farmaco, col_sintoma = st.columns(2)

    farmaco_paciente = col_farmaco.selectbox(
        "Fármaco Primario del Paciente:",
        options=[
            "Haloperidol (Antipsicótico Típico)",
            "Risperidona (Antipsicótico Atípico)",
            "Clozapina (Antipsicótico Atípico)",
            "Litio (Estabilizador del Ánimo)",
            "Paroxetina (ISRS)"
        ]
    )

    sintoma_emergente = col_sintoma.selectbox(
        "Nuevo Síntoma o Signo Clínico Emergente:",
        options=[
            "Rigidez muscular, temblor de reposo o acatisia",
            "Aumento de peso marcado y dislipidemia",
            "Sialorrea nocturna profusa (babeo)",
            "Poliuria, polidipsia o temblor fino distal",
            "Disfunción sexual (anorgasmia o retardo eyaculatorio)"
        ]
    )

    if st.button("🔍 Evaluar Cascada de Prescripción", use_container_width=True):
        if "Haloperidol" in farmaco_paciente and "Rigidez" in sintoma_emergente:
            st.markdown("""
            <div class="alert-critical">
                <h3>🚨 ALERTA ROJA: CASCADA DE PRESCRIPCIÓN CLÁSICA DETECTADA</h3>
                <p><b>Fármaco Causal:</b> Haloperidol (Antipsicótico Típico de alta potencia D2).</p>
                <p><b>Error Típico:</b> Prescribir Biperideno o Trihexifenidilo para tratar el "Parkinsonismo".</p>
                <p><b>Riesgo Severo Documentado por Jufe y Stahl:</b> La adición de anticolinérgicos aumenta el riesgo de <b>discinesia tardía irreversible</b>, deterioro cognitivo, boca seca y retención urinaria.</p>
                <hr>
                <h4>📋 Protocolo de Desprescripción Sofia:</h4>
                <ol>
                    <li><b>Reducir la dosis</b> de Haloperidol a la menor dosis antipsicótica efectiva.</li>
                    <li><b>Rotar</b> hacia un antipsicótico atípico con menor perfil extrapiramidal (ej. Quetiapina o Aripiprazol).</li>
                    <li>Si se usó Biperideno, realizar <b>retirada gradual</b> (desescalada lenta en 2-4 semanas) para evitar rebote colinérgico.</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        elif "Clozapina" in farmaco_paciente and "Sialorrea" in sintoma_emergente:
            st.markdown("""
            <div class="alert-warning">
                <h3>⚠️ EFECTO ADVERSO PROPIO DE CLOZAPINA (Agonismo M4)</h3>
                <p>La sialorrea nocturna de Clozapina es paradójica y secundaria a agonismo de receptores muscarínicos M4 o disfunción deglutoria.</p>
                <p><b>Manejo Racional:</b> No prescribir anticolinérgicos sistémicos de alta potencia. Usar medidas locales (atropina sublingual en gotas al acostarse) o ajustar la dosis nocturna.</p>
            </div>
            """, unsafe_allow_html=True)
        elif "Litio" in farmaco_paciente and "Poliuria" in sintoma_emergente:
            st.markdown("""
            <div class="alert-warning">
                <h3>⚠️ DIABETES INSÍPIDA NEFROGÉNICA INDUCIDA POR LITIO</h3>
                <p>El litio interfiere con la acuaporina-2 en el túbulo colector renal.</p>
                <p><b>Alerta Crítica:</b> ¡NUNCA prescribir diuréticos tiazídicos para la poliuria sin reducir drásticamente el litio! Las tiazidas reducen la excreción renal de litio provocando intoxicación letal.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-safe">
                <h3>✅ EFECTO ADVERSO SUSCEPTIBLE DE AJUSTE POSOLÓGICO</h3>
                <p>El síntoma reportado es compatible con el perfil farmacológico del tratamiento base. Evaluar reducción de dosis o titulación más gradual antes de añadir fármacos correctores.</p>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PESTAÑA 4: Explorador CANMAT y Catálogo
# =============================================================================
with tab_canmat:
    st.markdown("#### Explorador de Recomendaciones Oficiales CANMAT 2018")

    filtro_patologia = st.radio(
        "Seleccionar Patología:",
        options=["Trastorno Depresivo Mayor (TDM)", "Trastorno Bipolar"],
        horizontal=True
    )

    df_canmat_filtrado = data["canmat"][data["canmat"]["patologia"] == filtro_patologia]

    st.dataframe(
        df_canmat_filtrado[["fase", "linea", "estrategia", "farmaco", "atc", "nivel_evidencia", "comentarios"]],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.markdown("#### 🔍 Catálogo Rápido de Marcas y Códigos ATC (OMS)")

    busqueda_farmaco = st.text_input("Buscar principio activo:", value="Sertralina")
    info_atc = ATCEnricher.get_info(busqueda_farmaco)

    if info_atc:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Principio Activo", info_atc["nombre_oficial"])
        col_b.metric("Código ATC", info_atc["atc_codigo"])
        col_c.metric("Grupo Terapéutico", info_atc["atc_nivel3"])

        st.markdown(f"**Marcas comerciales registradas:** `{', '.join(info_atc['marcas_comerciales'])}`")
        st.caption(f"Clasificación OMS: {info_atc['atc_nivel1']} → {info_atc['atc_nivel2']} → {info_atc['atc_nivel4']}")
    else:
        st.warning(f"No se encontró información en el catálogo base para '{busqueda_farmaco}'.")
