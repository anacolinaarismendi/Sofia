"""
streamlit_app.py
Aplicación Web Interactiva de Sofia: Asistente Clínico Inteligente y RAG de Psicofarmacología.
Diseñado con estricta portabilidad relativa y cero rutas locales de usuario.
"""

import re
import sys
import traceback
from html import escape
from pathlib import Path
import pandas as pd
import streamlit as st

# set_page_config DEBE ser el primer comando ejecutable de Streamlit
st.set_page_config(
    page_title="Sofia | Asistente de Psicofarmacología Clínica",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de rutas relativas dinámicas
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Fallback seguro para ClinicalRAGEngine
try:
    from src.rag.engine import ClinicalRAGEngine
except Exception as _rag_err:
    RAG_INIT_ERROR = type(_rag_err).__name__
    # El detalle completo va al log del servidor, no a la interfaz pública
    traceback.print_exc(file=sys.stderr)
    st.error(
        f"⚠️ No se pudo inicializar el motor RAG ({RAG_INIT_ERROR}). "
        "Consulta los logs del servidor para ver el detalle."
    )

    class ClinicalRAGEngine:  # type: ignore
        def __init__(self, *args, **kwargs):
            class DummyStore:
                use_ollama = False
            self.vector_store = DummyStore()
            self.docs = []
            self.api_key = None
            self.ollama_host = None
            self.llm_model = "no disponible"
            self.kb_obsoleta = False

        def consultar(self, pregunta: str, top_k: int = 3):
            return {
                "pregunta": pregunta,
                "respuesta": f"El motor RAG no pudo inicializarse correctamente en este entorno ({RAG_INIT_ERROR}).",
                "voice_snippet": "",
                "generado_con_llm": False,
                "modo_respuesta": "error",
                "documentos_recuperados": []
            }


# Fallback seguro para ATCEnricher
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
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Carga de Motor RAG y Datasets en Caché
# -----------------------------------------------------------------------------
@st.cache_resource(ttl=600, show_spinner="Inicializando base de conocimiento clínica de Sofia...")
def get_rag_engine(api_key: str = "", host: str = "", model: str = ""):
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
    archivos = {
        "jufe": "jufe_farmacos_clean.csv",
        "jufe_efectos": "jufe_efectos_adversos_clean.csv",
        "stahl": "stahl_farmacos_clean.csv",
        "stahl_eventos": "stahl_eventos_adversos_clean.csv",
        "canmat": "canmat_recomendaciones_clean.csv",
        "canmat_alertas": "canmat_alertas_clinicas_clean.csv",
    }
    datasets = {}
    for clave, nombre in archivos.items():
        try:
            ruta = data_dir / nombre
            if ruta.exists():
                datasets[clave] = pd.read_csv(ruta)
        except Exception as e:
            st.warning(f"No se pudo cargar {nombre}: {e}")
    return datasets


def _md_a_html(texto: str) -> str:
    """Convierte negritas Markdown y saltos de párrafo a HTML limpio para tarjetas de alerta."""
    parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
    bloques_html = []
    for parrafo in parrafos:
        p_html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", parrafo)
        p_html = p_html.replace("\n", "<br>")
        bloques_html.append(f"<p>{p_html}</p>")
    return "".join(bloques_html)


def render_modo_respuesta(modo):
    """Indica visiblemente cómo se generó la respuesta (LLM, plantilla sin LLM, sin cobertura o error)."""
    if modo == "llm":
        st.caption("🤖 Respuesta generada por el modelo de lenguaje a partir de los documentos recuperados.")
    elif modo == "contingencia":
        st.warning(
            "🟡 **Modo contingencia – sin modelo de lenguaje.** Esto es una plantilla fija rellenada con el documento "
            "recuperado por coincidencia léxica; no es un razonamiento clínico sobre el caso. Consulta el documento citado."
        )
    elif modo == "sin_cobertura":
        st.info("⚪ Consulta fuera del alcance del corpus clínico de Sofia.")
    elif modo == "error":
        st.error("🔴 La consulta no pudo procesarse.")


def render_fuentes(fuentes):
    """Muestra el desplegable de fuentes recuperadas de manera segura y defensiva."""
    if not fuentes:
        return
    with st.expander("📚 Fuentes y Documentos Clínicos Recuperados"):
        for item in fuentes:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                doc, score = item
            else:
                doc, score = item, 0.0

            if not isinstance(doc, dict):
                continue

            doc_id = doc.get("id", "DOC")
            titulo = doc.get("titulo") or "Sin título"
            atc = doc.get("atc") or "N/D"
            archivo = doc.get("archivo_origen") or "N/D"
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)

            st.markdown(
                f"- **`[{doc_id}]` {titulo}** (ATC: `{atc}`) "
                f"— Relevancia: `{score_str}` | Archivo: `{archivo}`"
            )


# -----------------------------------------------------------------------------
# Barra Lateral (Sidebar)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=64)
    st.markdown("## **Sofia Clinical RAG**")
    st.caption("Asistente de Soporte a la Decisión y Farmacovigilancia en Psiquiatría")

    st.markdown("---")
    st.markdown("### 🔌 Inferencia y Conectividad")

    with st.expander("⚙️ Conexión Ollama Cloud / Remoto", expanded=False):
        st.caption("Si despliegas fuera de tu entorno local (Streamlit Cloud), configura tu API Key:")
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
        if st.button("Aplicar Credenciales", width="stretch"):
            st.session_state["ollama_api_key"] = user_key
            st.session_state["ollama_host"] = user_host
            # No se vacía la caché global (afectaría a otras sesiones): la clave de caché
            # incluye las credenciales, así que cambiarlas crea un motor nuevo.
            st.rerun()

    active_key = st.session_state.get("ollama_api_key", "")
    active_host = st.session_state.get("ollama_host", "")
    rag_engine = get_rag_engine(api_key=active_key, host=active_host)
    data = load_datasets()

    # Inspección segura de atributos del motor
    _api_key = getattr(rag_engine, "api_key", None)
    _host = getattr(rag_engine, "ollama_host", None) or ""
    _modelo = getattr(rag_engine, "llm_model", "N/D")
    _usa_ollama = getattr(getattr(rag_engine, "vector_store", None), "use_ollama", False)
    _modo_idx = getattr(getattr(rag_engine, "vector_store", None), "modo_indice", "N/D")

    if _api_key or "ollama.com" in _host:
        st.success(f"☁️ Ollama Cloud configurado (`{_modelo}`)")
    elif _usa_ollama:
        st.success(f"🟢 Ollama Local Activo (`{_modelo}`)")
    else:
        st.info("🟡 Inferencia Contingencia: TF-IDF + Fichas Estructuradas")
    st.caption(
        "Recuperación: "
        + ("semántica densa (embeddings)" if _modo_idx == "ollama" else "léxica TF-IDF")
        + ". La disponibilidad del LLM se indica en cada respuesta."
    )

    num_docs = len(getattr(rag_engine, "docs", []))
    st.metric("Base de Conocimiento", f"{num_docs} Documentos", delta="pre_training + CANMAT + marcas")
    if getattr(rag_engine, "kb_obsoleta", False):
        st.warning("⚠️ Hay documentos de pre_training/ más recientes que la base de conocimiento indexada. Regenera rag_knowledge_base.json.")


    st.markdown("---")
    st.markdown("### 📚 Corpus Clínico y Trazabilidad")
    st.markdown("""
    - **Corpus propio:** Síntesis estructurada basada en Jufe, Stahl y CANMAT 2018.
    - **Trazabilidad `[DOC-ID]`:** Cero citas apócrifas; respaldo directo por código de evidencia.
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

    st.markdown("**Consultas clínicas rápidas:**")
    col1, col2, col3, col4 = st.columns(4)
    query_preset = None

    if col1.button("💊 Sertralina (Posología Jufe)", width="stretch"):
        query_preset = "¿Qué laboratorios fabrican la Sertralina y cuáles son sus nombres comerciales y posología según Jufe?"
    if col2.button("⚠️ Paroxetina y Bupropión (CYP2D6)", width="stretch"):
        query_preset = "¿Qué interacciones y precauciones enzimáticas con CYP2D6 describe Stahl para Paroxetina y Bupropión?"
    if col3.button("📋 CANMAT Manía y Viraje", width="stretch"):
        query_preset = "¿Qué fármacos recomienda CANMAT en primera línea para manía aguda y por qué desaconseja los antidepresivos solos en depresión bipolar?"
    if col4.button("🛑 Haloperidol + Biperideno", width="stretch"):
        query_preset = "Un paciente en tratamiento con Haloperidol comienza con rigidez y temblor y le indican medicación antiparkinsoniana. ¿Qué dicen Jufe y Stahl sobre este riesgo?"

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

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_modo_respuesta(msg.get("modo"))
            if msg.get("voice_snippet"):
                st.markdown(
                    f'<div class="voice-card">🔊 <b>Voice Snippet Clínico (&lt; 40s):</b><br>{escape(msg["voice_snippet"])}</div>',
                    unsafe_allow_html=True
                )
            render_fuentes(msg.get("fuentes"))

    user_query = st.chat_input("Escribe tu consulta clínica aquí...") or query_preset

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Consultando literatura clínica y analizando interacciones..."):
                try:
                    resultado = rag_engine.consultar(user_query, top_k=3)
                except Exception as e:
                    traceback.print_exc(file=sys.stderr)
                    resultado = {
                        "respuesta": f"⚠️ Error al procesar la consulta ({type(e).__name__}). Revisa los logs del servidor.",
                        "voice_snippet": "",
                        "modo_respuesta": "error",
                        "documentos_recuperados": [],
                    }
                respuesta_texto = resultado.get("respuesta", "")
                voice_snip = resultado.get("voice_snippet", "")
                fuentes = resultado.get("documentos_recuperados", [])
                modo = resultado.get("modo_respuesta") or ("llm" if resultado.get("generado_con_llm") else "contingencia")

                st.markdown(respuesta_texto)
                render_modo_respuesta(modo)

                if voice_snip:
                    st.markdown(
                        f'<div class="voice-card">🔊 <b>Voice Snippet Clínico (&lt; 40s):</b><br>{escape(voice_snip)}</div>',
                        unsafe_allow_html=True
                    )

                render_fuentes(fuentes)

        st.session_state.messages.append({
            "role": "assistant",
            "content": respuesta_texto,
            "modo": modo,
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
        "Clozapina", "Risperidona", "Haloperidol", "Litio", "Lamotrigina", "Valproato"
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

        # Regla A: Paroxetina + Bupropión
        if "Paroxetina" in seleccion and "Bupropión" in seleccion:
            interacciones_detectadas.append({
                "nivel": "CRÍTICO",
                "fuente": "INTER-CYP2D6-001",
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

        # Regla B: Duplicidad Terapéutica de ISRS
        if len(isrs_seleccionados) >= 2:
            interacciones_detectadas.append({
                "nivel": "CRÍTICO",
                "fuente": None,
                "titulo": f"🔴 Duplicidad Terapéutica Inadecuada de ISRS ({' + '.join(sorted(isrs_seleccionados))})",
                "descripcion": (
                    "**Mecanismo:** La prescripción simultánea de dos o más fármacos de la misma familia (ISRS) carece de fundamento clínico "
                    "basado en evidencia y no aporta beneficio sinérgico en depresión mayor o trastornos de ansiedad.\n\n"
                    "**Riesgo Clínico:** Multiplica de forma inmediata el riesgo de **Síndrome Serotoninérgico** potencialmente letal "
                    "(hipertermia, clonus, mioclonías, inestabilidad autonómica, agitación psicomotriz) y riesgo hemorrágico por depleción plaquetaria de serotonina.\n\n"
                    "**Conducta Racional:** Contraindicado. Realizar cambio escalonado con período de retirada o lavado adecuado si se planea una rotación."
                )
            })

        # Regla C: Inhibición potente de CYP2D6 + Sustrato 2D6
        inhibidores_2d6 = {"Paroxetina", "Bupropión", "Fluoxetina"}
        sustratos_2d6 = {"Risperidona", "Haloperidol"}

        inh_presentes = set(seleccion).intersection(inhibidores_2d6)
        sus_presentes = set(seleccion).intersection(sustratos_2d6)

        if inh_presentes and sus_presentes:
            interacciones_detectadas.append({
                "nivel": "CRÍTICO",
                "fuente": "INTER-CYP2D6-001",
                "titulo": f"🔴 Bloqueo Enzimático Potente de CYP2D6 ({', '.join(sorted(inh_presentes))} + {', '.join(sorted(sus_presentes))})",
                "descripcion": (
                    f"**Mecanismo:** {', '.join(sorted(inh_presentes))} inhibe potentemente el CYP2D6, provocando una elevación "
                    f"marcada de las concentraciones plasmáticas de {', '.join(sorted(sus_presentes))}.\n\n"
                    f"**Riesgo Clínico:** Aparición de síntomas extrapiramidales agudos, sedación excesiva y prolongación del intervalo QTc.\n\n"
                    f"**Pauta:** Reducir la dosis del sustrato en un 50% o rotar hacia un antidepresivo con mínimo impacto 2D6 (ej. Sertralina o Escitalopram)."
                )
            })

        # Regla D: Litio e interacciones cruzadas
        if "Litio" in seleccion:
            if "Haloperidol" in seleccion:
                interacciones_detectadas.append({
                    "nivel": "CRÍTICO",
                    "fuente": None,
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
                    "fuente": None,
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
                    "fuente": None,
                "titulo": "🟡 Monitorización Neurológica Estricta (Litio + Clozapina)",
                    "descripcion": (
                        "**Riesgo:** Ambos agentes disminuyen el umbral convulsivo y aumentan el riesgo de mioclonías, confusión y síndrome neuroléptico maligno.\n\n"
                        "**Pauta:** Titulación lenta de clozapina y hemograma seriado habitual."
                    )
                })

            # Recordatorio renal permanente para Litio
            interacciones_detectadas.append({
                "nivel": "ADVERTENCIA",
                "fuente": "INTER-LIT-RENAL-004, MONO-LITIO-005",
                "titulo": "🟡 Vigilancia de Eliminación Renal de Litio",
                "descripcion": (
                    "**Riesgo Renal:** El litio tiene eliminación exclusivamente renal sin metabolismo CYP450. "
                    "La prescripción concomitante de AINEs (ibuprofeno, naproxeno), IECA o diuréticos tiazídicos "
                    "reduce su aclaramiento renal (los AINEs pueden elevar la litemia hasta un 40-60%), precipitando toxicidad lítica potencialmente fatal."
                )
            })

        # Regla E: Clozapina (CYP1A2)
        if "Clozapina" in seleccion:
            interacciones_detectadas.append({
                "nivel": "ADVERTENCIA",
                "fuente": "MONO-CLOZAPINA-006, INTER-CYP1A2-002",
                "titulo": "🟡 Monitorización Estricta de Clozapina (Sustrato Principal CYP1A2)",
                "descripcion": (
                    "**Vía Metabólica:** Clozapina depende en un 70% de CYP1A2. Fármacos inhibidores elevan su nivel hacia el rango convulsivo; "
                    "los hidrocarburos del humo del tabaco inducen CYP1A2 y reducen sus niveles hasta un 50%. Requiere hemograma seriado obligatorio."
                )
            })

        # Regla F: Lamotrigina + Valproato (Glucuronidación)
        if "Lamotrigina" in seleccion and "Valproato" in seleccion:
            interacciones_detectadas.append({
                "nivel": "CRÍTICO",
                "fuente": "INTER-LAM-VAL-003, ALERTA-LAM-SJS-002",
                "titulo": "🔴 Inhibición de Glucuronidación (Valproato + Lamotrigina)",
                "descripcion": (
                    "**Interacción:** El ácido valproico inhibe fuertemente la glucuronidación (UGT1A4/2B7) de la lamotrigina, "
                    "duplicando con creces su semivida plasmática y aumentando drásticamente la incidencia de erupciones cutáneas graves "
                    "y Síndrome de Stevens-Johnson.\n\n"
                    "**Pauta Obligatoria:** Reducir la pauta de titulación de lamotrigina a menos del 50% de la dosis estándar habitual."
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
                fuente = item.get("fuente")
                if fuente:
                    pie = f'<p><small>📎 Respaldo en el corpus: {", ".join(f"[{f.strip()}]" for f in fuente.split(","))}</small></p>'
                else:
                    pie = '<p><small>⚠️ Regla heurística sin respaldo directo verificado en el corpus de Sofia: contrastar con la ficha técnica o la literatura antes de actuar.</small></p>'
                st.markdown(
                    f'<div class="{css_class}"><h4>{item["titulo"]}</h4>{_md_a_html(item["descripcion"])}{pie}</div>',
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

    if st.button("🔍 Evaluar Cascada de Prescripción", width="stretch"):
        nombre_farmaco = farmaco_paciente.split(" (")[0]
        if nombre_farmaco in ("Haloperidol", "Risperidona") and "Rigidez" in sintoma_emergente:
            st.markdown(f"""
            <div class="alert-critical">
                <h3>🚨 ALERTA ROJA: CASCADA DE PRESCRIPCIÓN CLÁSICA DETECTADA</h3>
                <p><b>Fármaco Causal:</b> {farmaco_paciente}.</p>
                <p><b>Error Típico:</b> Prescribir Biperideno o Trihexifenidilo para tratar el "Parkinsonismo".</p>
                <p><b>Riesgo Severo Documentado por Jufe y Stahl:</b> La adición de anticolinérgicos aumenta el riesgo de <b>discinesia tardía irreversible</b>, deterioro cognitivo, boca seca y retención urinaria.</p>
                <hr>
                <h4>📋 Protocolo de Desprescripción Sofia:</h4>
                <ol>
                    <li><b>Reducir la dosis</b> de {nombre_farmaco} a la menor dosis antipsicótica efectiva.</li>
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
            st.markdown(f"""
            <div class="alert-warning" style="background-color: #f8fafc; border-left: 5px solid #64748b; color: #1e293b;">
                <h3>⚪ Sin regla específica de cascada para esta combinación</h3>
                <p>Este detector solo cubre unas pocas cascadas codificadas. <b>La ausencia de alerta no descarta una cascada de prescripción ni indica que el síntoma sea un efecto adverso benigno.</b></p>
                <p>Revisa la monografía de <b>{nombre_farmaco}</b>, valora si el síntoma puede ser un efecto adverso del fármaco antes de añadir otro tratamiento y plantea el caso a <b>Sofia</b> en el asistente RAG o a un especialista.</p>
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

    if "canmat" not in data or "patologia" not in data["canmat"].columns:
        st.warning("No se pudo cargar el dataset de CANMAT. Revisa la carpeta data/processed/.")
    else:
        df_canmat_filtrado = data["canmat"][data["canmat"]["patologia"] == filtro_patologia]
        columnas = [c for c in ["fase", "linea", "estrategia", "farmaco", "atc", "nivel_evidencia", "comentarios"]
                    if c in df_canmat_filtrado.columns]
        st.dataframe(
            df_canmat_filtrado[columnas],
            width="stretch",
            hide_index=True
        )

    st.markdown("---")
    st.markdown("#### 🔍 Catálogo Rápido de Marcas y Códigos ATC (OMS)")

    busqueda_farmaco = st.text_input("Buscar principio activo:", value="Sertralina")
    info_atc = ATCEnricher.get_info(busqueda_farmaco)

    if info_atc:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Principio Activo", info_atc.get("nombre_oficial", "N/D"))
        col_b.metric("Código ATC", info_atc.get("atc_codigo", "N/D"))
        col_c.metric("Grupo Terapéutico", info_atc.get("atc_nivel3", "N/D"))

        marcas = info_atc.get("marcas_comerciales") or ["N/D"]
        st.markdown(f"**Marcas comerciales registradas:** `{', '.join(marcas)}`")
        st.caption(
            f"Clasificación OMS: {info_atc.get('atc_nivel1', 'N/D')} → "
            f"{info_atc.get('atc_nivel2', 'N/D')} → {info_atc.get('atc_nivel4', 'N/D')}"
        )
    else:
        st.warning(f"No se encontró información en el catálogo base para '{busqueda_farmaco}'.")
