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
def resolver_credenciales_ollama() -> tuple[str, str, str]:
    """Extrae y valida credenciales de st.secrets de manera segura."""
    sec_key, sec_host, sec_model = "", "", ""
    try:
        if hasattr(st, "secrets"):
            sec_key = str(st.secrets.get("OLLAMA_API_KEY", "")).strip()
            sec_host = str(st.secrets.get("OLLAMA_HOST", "")).strip()
            sec_model = str(st.secrets.get("OLLAMA_MODEL", "")).strip()
    except Exception:
        pass

    if sec_key in ("", "tu_ollama_api_key_aqui"):
        sec_key = ""

    return sec_key, sec_host, sec_model


@st.cache_resource(ttl=600, show_spinner="Inicializando base de conocimiento clínica de Sofia...")
def get_rag_engine(api_key: Optional[str] = None, host: Optional[str] = None, model: Optional[str] = None):
    return ClinicalRAGEngine(
        ollama_host=host or None,
        llm_model=model or None,
        api_key=api_key or None
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

    sec_key, sec_host, sec_model = resolver_credenciales_ollama()
    active_key = st.session_state.get("ollama_api_key") or sec_key or None
    active_host = st.session_state.get("ollama_host") or sec_host or None
    active_model = st.session_state.get("ollama_model") or sec_model or None

    rag_engine = get_rag_engine(api_key=active_key, host=active_host, model=active_model)
    data = load_datasets()

    # Inspección segura de atributos del motor
    _api_key = getattr(rag_engine, "api_key", None)
    _host = getattr(rag_engine, "ollama_host", None) or ""
    _modelo = getattr(rag_engine, "llm_model", "N/D")
    _usa_ollama = getattr(getattr(rag_engine, "vector_store", None), "use_ollama", False)
    _modo_idx = getattr(getattr(rag_engine, "vector_store", None), "modo_indice", "N/D")

    # Detección de configuración de Ollama Cloud y validación de clave
    es_ollama_cloud = "ollama.com" in _host
    key_faltante = not _api_key or _api_key in ("", "tu_ollama_api_key_aqui")

    with st.expander("⚙️ Conexión Ollama Cloud / Remoto", expanded=key_faltante and es_ollama_cloud):
        if sec_key:
            st.success(f"🔑 Secrets detectados (`{sec_model or 'qwen2.5:7b'}`)")
        else:
            st.caption("Si despliegas fuera de tu entorno local (Streamlit Cloud), configura tu API Key:")

        if key_faltante and es_ollama_cloud:
            st.warning(
                "🔑 **Falta la API Key de Ollama Cloud:**\n\n"
                "Para habilitar el modelo generativo LLM en Streamlit Community Cloud:\n"
                "1. Genera tu clave en [ollama.com/settings/keys](https://ollama.com/settings/keys)\n"
                "2. En el panel de Streamlit Cloud ve a **App Settings → Secrets** y define:\n"
                "```toml\n"
                'OLLAMA_API_KEY = "tu_clave_real_aqui"\n'
                'OLLAMA_HOST = "https://ollama.com"\n'
                'OLLAMA_MODEL = "qwen2.5:7b"\n'
                "```\n"
                "O ingrésala a continuación para esta sesión:"
            )
        user_key = st.text_input(
            "Ollama Cloud API Key:",
            type="password",
            value=st.session_state.get("ollama_api_key", "") or sec_key,
            help="Obtén tu API key en ollama.com/settings/keys"
        )
        user_host = st.text_input(
            "Host URL:",
            value=st.session_state.get("ollama_host", "") or sec_host or "https://ollama.com",
            placeholder="https://ollama.com"
        )
        user_model = st.text_input(
            "Modelo LLM:",
            value=st.session_state.get("ollama_model", "") or sec_model or "qwen2.5:7b",
            placeholder="qwen2.5:7b o gpt-oss:20b"
        )
        if st.button("Aplicar Credenciales", width="stretch"):
            st.session_state["ollama_api_key"] = user_key
            st.session_state["ollama_host"] = user_host
            st.session_state["ollama_model"] = user_model
            st.cache_resource.clear()
            st.rerun()

    if es_ollama_cloud and key_faltante:
        st.warning("⚠️ **Ollama Cloud sin API Key:** Operando en modo contingencia.")
    elif not key_faltante and es_ollama_cloud:
        st.success(f"☁️ Ollama Cloud activo (`{_modelo}`)")
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
# PESTAÑA 3: Detector de Cascadas de Prescripción y Desprescripción
# =============================================================================
with tab_cascade:
    st.markdown("#### Detector Heurístico de Cascadas de Prescripción y Desprescripción")
    st.write(
        "Una **cascada de prescripción** ocurre cuando el efecto adverso de un primer fármaco es malinterpretado como "
        "un nuevo problema de salud, conduciendo a prescribir un segundo fármaco innecesario que introduce nueva toxicidad e iatrogenia."
    )

    CASCADAS_CLINICAS = {
        "Haloperidol (Antipsicótico Típico)": {
            "Rigidez muscular, lentitud motora o temblor de reposo (Parkinsonismo secundario)": {
                "nivel": "CRÍTICO",
                "titulo": "🚨 ALERTA ROJA: CASCADA DE PRESCRIPCIÓN CLÁSICA (PARKINSONISMO → BIPERIDENO)",
                "mecanismo": "Bloqueo masivo (>80%) de receptores dopaminérgicos D2 en la vía nigroestriatal, provocando una disfunción extrapiramidal idéntica a la enfermedad de Parkinson idiopática.",
                "error_tipico": "Prescribir Biperideno (Akineton) o Trihexifenidilo de forma crónica para enmascarar el temblor o la rigidez en lugar de optimizar la dosis del antipsicótico.",
                "riesgo_iatrogenico": "Elevada carga anticolinérgica agregada: deterioro neurocognitivo (memoria episódica y funciones ejecutivas), boca seca, estreñimiento grave / íleo paralítico, retención urinaria y riesgo elevado de inducir o desenmascarar discinesia tardía irreversible.",
                "protocolo": [
                    "<b>Reducir la dosis</b> de Haloperidol a la menor dosis antipsicótica efectiva (habitualmente ≤ 2-4 mg/día).",
                    "Si persisten los síntomas extrapiramidales o existe riesgo de recaída psicótica, <b>rotar</b> hacia un antipsicótico atípico con menor perfil extrapiramidal (ej. Quetiapina, Aripiprazol u Olanzapina).",
                    "Si ya se inició Biperideno, realizar <b>desescalada y retirada gradual</b> (reduciendo 1-2 mg/semana a lo largo de 2-4 semanas) para evitar rebote colinérgico agudo y taquifilaxia."
                ],
                "fuente": "Jufe - Psicofarmacología Práctica, Stahl - Prescriber's Guide, Criterios STOPP/START"
            },
            "Inquietud motora imperiosa e incapacidad para permanecer sentado (Acatisia)": {
                "nivel": "ADVERTENCIA",
                "titulo": "⚠️ SOSPECHA DE ACATISIA FARMACOLÓGICA VS. AGITACIÓN PSICÓTICA",
                "mecanismo": "Hipoactividad dopaminérgica en circuitos mesocorticales y estriatales inducida por el potente bloqueo D2.",
                "error_tipico": "Confundir la inquietud motora con empeoramiento de la psicosis o ansiedad y aumentar la dosis de Haloperidol, o instaurar benzodiacepinas a dosis altas a largo plazo.",
                "riesgo_iatrogenico": "Agravamiento paradójico de la angustia subjetiva e impulsividad motora, elevando el riesgo de conducta violenta o ideación autolítica.",
                "protocolo": [
                    "<b>Disminuir la dosis</b> de Haloperidol de forma inmediata.",
                    "Considerar un beta-bloqueante lipofílico a dosis bajas (<b>Propranolol 10-40 mg/día</b>) como coadyuvante sintomático transitorio si no hay contraindicación respiratoria o bradicardia.",
                    "Rotar a un antipsicótico con menor potencial acatisiante si no cede tras el ajuste de dosis."
                ],
                "fuente": "Jufe - Psicofarmacología Práctica, Stahl"
            },
            "Contracciones espasmódicas agudas en cuello, mandíbula u ojos (Distonía aguda)": {
                "nivel": "CRÍTICO",
                "titulo": "🛑 EMERGENCIA EXTRAPIRAMIDAL AGUDA (DISTONÍA AGUDA O CRISIS OCULÓGIRA)",
                "mecanismo": "Desbalance agudo hipercolinérgico estriatal por bloqueo D2 intenso en fases tempranas o tras aumentos bruscos de posología.",
                "error_tipico": "Manejar la urgencia con anticolinérgicos parenterales pero dejar luego el Biperideno prescrito de forma crónica e indefinida durante meses.",
                "riesgo_iatrogenico": "Cronicidad inadvertida de la terapia anticolinérgica con toxicidad cognitiva y autonómica.",
                "protocolo": [
                    "<b>Tratamiento agudo de rescate:</b> Biperideno 5 mg IM/IV lenta o benzodiacepina parenteral de acción rápida.",
                    "<b>Reevaluar la posología:</b> Reducir dosis o ralentizar la titulación del antipsicótico.",
                    "<b>Retirada planificada:</b> Suspender el anticolinérgico oral en pocas semanas y considerar la sustitución por un antipsicótico atípico."
                ],
                "fuente": "Jufe - Psicofarmacología Práctica, Stahl"
            }
        },
        "Risperidona (Antipsicótico Atípico)": {
            "Rigidez, lentitud motora o temblor dosis-dependiente": {
                "nivel": "CRÍTICO",
                "titulo": "🚨 CASCADA EXTRAPIRAMIDAL POR PÉRDIDA DE ATIPICIDAD (DOSIS-DEPENDIENTE)",
                "mecanismo": "A dosis superiores a 4-6 mg/día, la Risperidona sobrepasa el 80% de ocupación de receptores D2 en el estriado, perdiendo la ventaja protectora de su antagonismo 5-HT2A y comportándose clínicamente como un neuroléptico típico.",
                "error_tipico": "Añadir Biperideno u otros anticolinérgicos en lugar de retornar la Risperidona a su rango de atipicidad terapéutica.",
                "riesgo_iatrogenico": "Deterioro cognitivo, síndrome confusional en pacientes vulnerables y riesgo incrementado de discinesia tardía.",
                "protocolo": [
                    "<b>Reducir la dosis</b> a un intervalo óptimo de atipicidad (habitualmente 2-4 mg/día).",
                    "Si se requiere mayor potencia antipsicótica sin efectos motores, valorar la rotación a otro antipsicótico atípico (ej. Aripiprazol o Quetiapina).",
                    "Evitar la prescripción crónica profiláctica de anticolinérgicos en combinación con antipsicóticos de segunda generación."
                ],
                "fuente": "Jufe - Psicofarmacología Práctica, Stahl"
            },
            "Galactorrea, amenorrea o ginecomastia (Hiperprolactinemia marcada)": {
                "nivel": "ADVERTENCIA",
                "titulo": "⚠️ CASCADA ENDOCRINA: HIPERPROLACTINEMIA Y RIESGO DE AGONISTAS DOPAMINÉRGICOS",
                "mecanismo": "Bloqueo D2 en las células lactotrópicas de la adenohipófisis (vía tuberoinfundibular), zona carente de barrera hematoencefálica estricta.",
                "error_tipico": "Derivar a endocrinología y prescribir agonistas dopaminérgicos (Cabergolina, Bromocriptina), lo que compite frontalmente con la acción antipsicótica y reactiva los síntomas psicóticos.",
                "riesgo_iatrogenico": "Reagudización psicótica por competición D2 mesolímbica; desmineralización ósea acelerada por hipogonadismo crónico.",
                "protocolo": [
                    "Medir prolactina sérica basal y descartar prolactinoma orgánico si los niveles son extremos (>150-200 ng/mL).",
                    "<b>Estrategia de elección:</b> Coadyuvar con una dosis baja de <b>Aripiprazol (5-10 mg/día)</b>; su agonismo parcial D2 hipofisario rescata el tono inhibitorio y normaliza la prolactina sin descompensar la psicosis.",
                    "Si persiste, rotar hacia antipsicóticos ahorradores de prolactina (Olanzapina, Quetiapina o Aripiprazol en monoterapia)."
                ],
                "fuente": "Consensos ADA/APA y Guías de Endocrinología, Stahl"
            },
            "Aumento de peso progresivo y resistencia a la insulina": {
                "nivel": "ADVERTENCIA",
                "titulo": "⚠️ ALTERACIÓN METABÓLICA Y GANANCIA PONDERAL",
                "mecanismo": "Antagonismo moderado de receptores 5-HT2C e histaminérgicos H1 que estimula el centro hipotalámico del apetito.",
                "error_tipico": "Iniciar fármacos hipolipemiantes o antidiabéticos sin intervenir precozmente en la dieta o valorar alternativas metabólicamente neutras.",
                "riesgo_iatrogenico": "Desarrollo progresivo de síndrome metabólico y aumento del riesgo cardiovascular global.",
                "protocolo": [
                    "Registrar peso, perímetro abdominal, glucemia y perfil lipídico basal y a las 4-8 semanas.",
                    "Si la ganancia de peso supera el 5% respecto al basal en el primer mes, plantear rotación a antipsicóticos de bajo impacto metabólico (Aripiprazol, Lurasidona o Ziprasidona).",
                    "Acompañar siempre de asesoramiento nutricional y actividad física."
                ],
                "fuente": "Consenso ADA/APA, Jufe"
            }
        },
        "Olanzapina (Antipsicótico Atípico)": {
            "Aumento de peso masivo (>5% basal), hiperfagia y dislipidemia": {
                "nivel": "CRÍTICO",
                "titulo": "🚨 CASCADA METABÓLICA: DE OLANZAPINA A POLIFARMACIA CARDIOMETABÓLICA",
                "mecanismo": "Potente bloqueo histaminérgico H1 que activa la AMPK hipotalámica y antagonismo 5-HT2C que anula las señales de saciedad mediadas por leptina y POMC, induciendo apetencia compulsiva por hidratos de carbono.",
                "error_tipico": "Prescribir estatinas y fibratos de forma automática para la dislipidemia resultante sin replantear el tratamiento antipsicótico causante.",
                "riesgo_iatrogenico": "Obesidad grado II-III, esteatosis hepática metabólica, cardiopatía isquémica prematura y polifarmacia cardiovascular.",
                "protocolo": [
                    "<b>Rotación estratégica:</b> Si la estabilidad psicopatológica lo consiente, rotar a antipsicóticos metabólicamente favorables (Aripiprazol, Lurasidona, Cariprazina o Ziprasidona).",
                    "<b>Coadyuvancia con Metformina:</b> Si Olanzapina es insustituible por eficacia clínica exclusiva, iniciar precozmente Metformina (500 a 850 mg c/12h) para mitigar la lipogénesis y mejorar la sensibilidad a la insulina.",
                    "Controles analíticos estrictos: glucemia y lípidos basales, al mes, a los 3 meses y posteriormente de forma anual."
                ],
                "fuente": "Consenso ADA/APA sobre Antipsicóticos y Diabetes, Jufe, Stahl"
            },
            "Hiperglucemia en ayunas o elevación de HbA1c (Resistencia a la insulina)": {
                "nivel": "CRÍTICO",
                "titulo": "🚨 DISMETABOLISMO GRAVE: RESISTENCIA A LA INSULINA POR BLOQUEO M3 PANCREÁTICO",
                "mecanismo": "El antagonismo de receptores muscarínicos M3 en las células beta de los islotes de Langerhans reduce directamente la secreción aguda de insulina inducida por glucosa, independientemente de la masa adiposa.",
                "error_tipico": "Etiquetar al paciente con diabetes mellitus tipo 2 idiopática e iniciar antidiabéticos orales o insulina sin cuestionar la Olanzapina.",
                "riesgo_iatrogenico": "Diabetes inducida por fármacos de difícil control, episodios de cetoacidosis atípica y daño micro/macrovascular.",
                "protocolo": [
                    "Considerar de forma prioritaria la desescalada o rotación a un antipsicótico con nula afinidad M3 (ej. Aripiprazol).",
                    "Si la sustitución no es viable, coordinar con endocrinología el inicio de agentes sensibilizantes a la insulina (Metformina) o análogos GLP-1.",
                    "Evitar fármacos que promuevan mayor ganancia ponderal (como sulfonilureas) siempre que sea posible."
                ],
                "fuente": "Jufe - Psicofarmacología Práctica, Consenso ADA/APA"
            },
            "Sedación diurna profunda y somnolencia matutina": {
                "nivel": "ADVERTENCIA",
                "titulo": "⚠️ SEDACIÓN CENTRAL POR BLOQUEO HISTAMINÉRGICO H1",
                "mecanismo": "Afinidad nanomolar por receptores histaminérgicos H1 y muscarínicos M1 en el sistema nervioso central.",
                "error_tipico": "Prescribir psicoestimulantes (Metilfenidato, Modafinilo) o consumo abusivo de cafeína/bebidas energéticas.",
                "riesgo_iatrogenico": "Riesgo de descompensación psicótica o arritmias por interacción entre estimulantes y antipsicóticos.",
                "protocolo": [
                    "Administrar la dosis total diaria en una toma única nocturna, 1 a 2 horas antes de dormir.",
                    "Reducir ligeramente la dosis si el estado clínico lo permite.",
                    "Si la interferencia sociolaboral es severa, rotar hacia un atípico activador."
                ],
                "fuente": "Stahl - Prescriber's Guide, Jufe"
            }
        },
        "Clozapina (Antipsicótico Atípico)": {
            "Sialorrea nocturna profusa (hipersalivación y babeo en almohada)": {
                "nivel": "ADVERTENCIA",
                "titulo": "⚠️ SIALORREA PARADÓJICA POR AGONISMO MUSCARÍNICO M4",
                "mecanismo": "Agonismo selectivo sobre receptores muscarínicos colinérgicos M4 periféricos y centrales, asociado a inhibición del reflejo deglutorio durante las fases profundas del sueño.",
                "error_tipico": "Prescribir anticolinérgicos sistémicos de alta potencia por vía oral (Biperideno o Trihexifenidilo) para frenar la salivación.",
                "riesgo_iatrogenico": "¡ALTO PELIGRO DE ÍLEO PARALÍTICO! La adición de anticolinérgicos sistémicos a la Clozapina paraliza la motilidad cólica provocando constipación severa, obstrucción intestinal potencialmente letal y agravamiento cognitivo.",
                "protocolo": [
                    "<b>NUNCA prescribir anticolinérgicos orales sistémicos de forma concomitante con Clozapina.</b>",
                    "<b>Tratamiento local de primera línea:</b> Gotas de Atropina oftálmica al 1% (1 a 2 gotas sublinguales al acostarse) o solución tópica de Bromuro de Ipratropio.",
                    "Ajustar la dosis nocturna de Clozapina, recomendar toalla absorbente sobre la almohada y dormir con la cabecera ligeramente incorporada."
                ],
                "fuente": "Jufe - Psicofarmacología Práctica, Stahl - Prescriber's Guide"
            },
            "Estreñimiento pertinaz o distensión abdominal (Hipomotilidad gastrointestinal)": {
                "nivel": "CRÍTICO",
                "titulo": "🛑 ALERTA CRÍTICA: RIESGO DE ÍLEO PARALÍTICO POR CLOZAPINA",
                "mecanismo": "Potente acción antimuscarínica (M1, M2, M3) periférica y bloqueo 5-HT3/5-HT2C que reduce intensamente el peristaltismo intestinal.",
                "error_tipico": "Tratar con indiferencia el estreñimiento o prescribir suplementos formadores de masa sin pauta laxante activa.",
                "riesgo_iatrogenico": "El íleo paralítico por clozapina tiene una letalidad superior a la agranulocitosis debido a megacolon tóxico, necrosis y perforación intestinal con sepsis.",
                "protocolo": [
                    "<b>Monitorización obligatoria:</b> Interrogar el hábito defecatorio en cada consulta médica (si transcurren >48 horas sin deposición, intervenir activamente).",
                    "Instaurar de inmediato laxantes osmóticos (Macrogol / Polietilenglicol) y pauta de hidratación oral abundante.",
                    "Evitar terminantemente la adición de cualquier otro fármaco con propiedades anticolinérgicas."
                ],
                "fuente": "Alertas Sanitarias de Farmacovigilancia, Stahl, Jufe"
            },
            "Aumento ponderal severo y resistencia insulínica en esquizofrenia refractaria": {
                "nivel": "CRÍTICO",
                "titulo": "🚨 RIESGO CARDIOMETABÓLICO EN PACIENTE REFRACTARIO",
                "mecanismo": "Estimulación central intensa del apetito por bloqueo H1 y alteración directa de la secreción pancreática de insulina.",
                "error_tipico": "Suspender bruscamente la Clozapina (provocando tormenta colinérgica y recaída psicótica masiva) o descuidar la monitorización analítica.",
                "riesgo_iatrogenico": "Recaída psicótica intratable por retirada vs. morbilidad cardiovascular grave por abandono metabólico.",
                "protocolo": [
                    "Como Clozapina es el único fármaco con indicación aprobada en resistencia, la rotación a menudo es inviable.",
                    "Estrategia de referencia: Coadyuvancia precoz con <b>Metformina (850-1000 mg c/12h)</b> o añadir <b>Aripiprazol a dosis bajas (5-10 mg/día)</b> para frenar la ganancia de peso.",
                    "Revisión trimestral de lípidos, glucosa basal y HbA1c."
                ],
                "fuente": "Consenso ADA/APA, Jufe, Stahl"
            }
        },
        "Litio (Carbonato de Litio)": {
            "Poliuria marcada (>3 L/día) y polidipsia intensa (Diabetes insípida nefrogénica)": {
                "nivel": "CRÍTICO",
                "titulo": "🛑 ALERTA ROJA: DIABETES INSÍPIDA NEFROGÉNICA Y RIESGO LETAL DE TIAZIDAS",
                "mecanismo": "El litio penetra a las células principales del túbulo colector renal a través de los canales epiteliales de sodio (ENaC) e inhibe la adenilil ciclasa y la GSK-3beta, bloqueando la translocación de acuaporina-2 mediada por vasopresina (ADH).",
                "error_tipico": "¡Prescribir diuréticos tiazídicos (Hidroclorotiazida, Clortalidona) para la hipertensión o retención hidrosalina sin ajustar el Litio!",
                "riesgo_iatrogenico": "Las tiazidas inducen natriuresis distal compensada con hiperreabsorción de sodio y LITIO en el túbulo proximal. Esto eleva la litemia un 40-70% provocando INTOXICACIÓN GRAVE POR LITIO POTENCIALMENTE MORTAL (ataxia, convulsiones, arritmias y fracaso renal agudo).",
                "protocolo": [
                    "Medir de inmediato litemia en ayunas, creatinina, ionograma plasmático y volumen urinario de 24 horas.",
                    "Si la poliuria es severa y el litio es imprescindible, el fármaco de elección es <b>Amilorida (5-10 mg/día)</b>: bloquea los canales ENaC impidiendo la acumulación de litio en el túbulo renal sin elevar la litemia.",
                    "Si un diurético tiazídico fuera imprescindible por causa cardiovascular mayor, la dosis de litio DEBE REDUCIRSE al 50% con determinaciones de litemia cada 48-72 horas hasta estabilizar niveles."
                ],
                "fuente": "Jufe - Psicofarmacología Práctica, Stahl - Prescriber's Guide"
            },
            "Temblor fino postural de manos (persistente o agravado)": {
                "nivel": "ADVERTENCIA",
                "titulo": "⚠️ CASCADA: TEMBLOR POR LITIO → PROPRANOLOL SIN MEDIR LITEMIA",
                "mecanismo": "Efecto electrofisiológico directo del catión litio sobre la excitabilidad y sincronización neuromuscular periférica.",
                "error_tipico": "Prescribir Propranolol u otros betabloqueantes de forma automática sin verificar si el temblor es el primer síntoma de sobredosis o litemia tóxica en ascenso.",
                "riesgo_iatrogenico": "Enmascaramiento de toxicidad renal/sistémica por litio; agravamiento de broncoespasmo en pacientes con hiperreactividad bronquial o bradicardia sintomática.",
                "protocolo": [
                    "<b>Paso 1 obligado:</b> Determinar litemia sérica a las 12 horas de la última dosis nocturna.",
                    "Si la litemia supera 0.8-1.0 mEq/L, ajustar la dosis a la baja hasta el rango terapéutico óptimo (0.6-0.8 mEq/L).",
                    "Moderar el consumo de cafeína (las fluctuaciones agudas alteran la excreción renal de litio).",
                    "Solo si la litemia es normal y el temblor causa interferencia funcional grave, añadir Propranolol a dosis bajas (10-20 mg c/8-12h) tras verificar ausencia de contraindicaciones respiratorias o cardíacas."
                ],
                "fuente": "Jufe - Psicofarmacología Práctica, Stahl"
            },
            "Astenia, aumento de peso y friolera (Hipotiroidismo farmacológico)": {
                "nivel": "SEGURO",
                "titulo": "ℹ️ MANEJO RACIONAL: HIPOTIROIDISMO INDUCIDO POR LITIO (NO SUSPENDER EL LITIO)",
                "mecanismo": "El litio interfiere en la captación tiroidea de yoduro, bloquea el acoplamiento de tirosinas y frena la liberación periférica de T3 y T4.",
                "error_tipico": "Suspender apresuradamente el carbonato de litio ante el hallazgo de TSH elevada, perdiendo la estabilidad afectiva en un paciente bipolar previamente compensado.",
                "riesgo_iatrogenico": "Recaída grave en manía o depresión bipolar con riesgo elevado de conducta suicida.",
                "protocolo": [
                    "Solicitar analítica completa con TSH y T4 libre.",
                    "<b>NO suspender el carbonato de litio:</b> el hipotiroidismo secundario se maneja de forma segura iniciando terapia sustitutiva con Levotiroxina oral a dosis estándar (25-50 mcg/día inicialmente).",
                    "Titular la dosis de levotiroxina según niveles de TSH cada 6-8 semanas manteniendo el litio en su rango terapéutico."
                ],
                "fuente": "Guías CANMAT para Trastorno Bipolar, Jufe"
            }
        },
        "Paroxetina (ISRS)": {
            "Disfunción sexual (anorgasmia, retardo eyaculatorio o caída de la libido)": {
                "nivel": "ADVERTENCIA",
                "titulo": "⚠️ CASCADA: DISFUNCIÓN SEXUAL POR ISRS → SILDENAFILO / TADALAFILO",
                "mecanismo": "La hiperactivación de receptores 5-HT2A y 5-HT2C inducida por el bloqueo de recaptación de serotonina inhibe la liberación de dopamina en el área mesolímbica y suprime la vía del óxido nítrico genital y espinal.",
                "error_tipico": "Recetar de forma rutinaria inhibidores de la fosfodiesterasa-5 (Sildenafilo/Tadalafilo) sin investigar alternativas de optimización del antidepresivo.",
                "riesgo_iatrogenico": "Gasto económico elevado y falta de eficacia sobre la anorgasmia o la pérdida de deseo (el sildenafilo únicamente actúa sobre la erección hemodinámica, no sobre los receptores centrales de la libido).",
                "protocolo": [
                    "Evaluar la reducción de la dosis de Paroxetina a la dosis mínima eficaz.",
                    "<b>Rotación estratégica:</b> Rotar hacia antidepresivos con mecanismo ahorrador de función sexual: <b>Bupropión</b> (inhibidor de recaptación de dopamina/noradrenalina), <b>Vortioxetina</b> (modulador multimodal) o <b>Mirtazapina</b>.",
                    "<b>Coadyuvancia farmacológica:</b> Si el paciente no puede cambiar de antidepresivo, añadir <b>Bupropión (150 mg/día)</b> como modulador dopaminérgico para revertir la disfunción sexual inducida por el ISRS."
                ],
                "fuente": "Jufe - Psicofarmacología Práctica, Stahl - Prescriber's Guide"
            },
            "Mareos, descargas eléctricas cefálicas ('brain zaps') y náuseas por olvido de dosis": {
                "nivel": "ADVERTENCIA",
                "titulo": "⚠️ SÍNDROME DE DISCONTINUACIÓN INTENSO POR VIDA MEDIA CORTA",
                "mecanismo": "La Paroxetina carece de metabolitos activos y posee una vida media corta (aprox. 21-24h) con alta afinidad por SERT, originando una brusca desensibilización serotoninérgica y un rebote colinérgico al retrasar una sola dosis.",
                "error_tipico": "Interpretar erróneamente los síntomas de abstinencia/discontinuación como una 'recaída fulminante del cuadro depresivo o ansioso' y subir la dosis o añadir ansiolíticos.",
                "riesgo_iatrogenico": "Polifarmacia con benzodiacepinas, retraso diagnóstico y dependencia psicológica severa al fármaco.",
                "protocolo": [
                    "Reinstaurar la dosis habitual de Paroxetina para confirmar el cese inmediato del malestar (prueba diagnóstica).",
                    "Si el objetivo clínico es retirar el antidepresivo, realizar una desescalada extraordinariamente lenta (ej. reducciones de 2.5-5 mg cada 3-4 semanas).",
                    "Alternativa validada: Cambiar previamente a <b>Fluoxetina</b> (cuya vida media combinada con norfluoxetina es de hasta 2 semanas), permitiendo una retirada suave sin fenómeno de rebote."
                ],
                "fuente": "Stahl - Prescriber's Guide, Jufe"
            },
            "Sedación matutina, ganancia de peso y boca seca": {
                "nivel": "ADVERTENCIA",
                "titulo": "⚠️ AFINIDAD ANTIMUSCARÍNICA PROPIA DE LA PAROXETINA",
                "mecanismo": "A diferencia de otros ISRS, la Paroxetina ejerce un antagonismo significativo sobre receptores muscarínicos colinérgicos M1 e inhibe la síntesis de óxido nítrico sintasa.",
                "error_tipico": "Añadir estimulantes diurnos, colagogos o fármacos para la xerostomía sin abordar la causa molecular.",
                "riesgo_iatrogenico": "Mayor riesgo de caídas, confusión y retención urinaria en pacientes mayores de 65 años.",
                "protocolo": [
                    "Rotar a un ISRS más limpio y selectivo sin actividad anticolinérgica (Escitalopram o Sertralina).",
                    "Si se conserva el tratamiento, concentrar la toma en el horario nocturno."
                ],
                "fuente": "Jufe - Psicofarmacología Práctica, Stahl"
            }
        },
        "Sertralina / Fluoxetina (ISRS)": {
            "Inquietud motora, agitación o ansiedad paradójica inicial": {
                "nivel": "ADVERTENCIA",
                "titulo": "⚠️ CASCADA DE ANSIEDAD INICIAL → BENZODIACEPINAS CRÓNICAS",
                "mecanismo": "Activación aguda de receptores 5-HT2C somatodendríticos antes de que se produzca la regulación a la baja (down-regulation) adaptativa tras 10-14 días de tratamiento continuo.",
                "error_tipico": "Prescribir benzodiacepinas de alta potencia (Lorazepam, Alprazolam) para la agitación inicial y mantenerlas de forma continuada durante meses o años por inercia terapéutica.",
                "riesgo_iatrogenico": "Dependencia farmacológica, tolerancia, sedación diurna, riesgo de caídas y deterioro cognitivo a largo plazo.",
                "protocolo": [
                    "Comenzar siempre con una titulación escalonada (ej. Sertralina 25 mg/día o Fluoxetina 10 mg/día durante los primeros 4-7 días).",
                    "<b>Psicoeducación activa:</b> Advertir al paciente que la hiperactivación o inquietud leve inicial es un fenómeno farmacológico transitorio que remite espontáneamente en 1 a 2 semanas.",
                    "Si es imprescindible asociar una benzodiacepina para mitigar la angustia aguda, pautarla estrictamente por un tiempo máximo de 10 a 14 días con pauta de retirada fijada desde el primer día."
                ],
                "fuente": "Guías CANMAT para Depresión, Jufe, Stahl"
            },
            "Disfunción sexual persistente (anorgasmia o retraso eyaculatorio)": {
                "nivel": "ADVERTENCIA",
                "titulo": "⚠️ DISFUNCIÓN SEXUAL SEROTONINÉRGICA POR ESTIMULACIÓN 5-HT2A",
                "mecanismo": "El aumento de serotonina intrasináptica estimula receptores 5-HT2A medulares y mesolímbicos, inhibiendo la neurotransmisión dopaminérgica del circuito de recompensa y el tono autonómico genital.",
                "error_tipico": "Normalizar el efecto como 'inevitable', lo que provoca que el paciente suspenda el tratamiento unilateralmente sin avisar al clínico.",
                "riesgo_iatrogenico": "Abandono terapéutico abrupto y recaída depresiva grave.",
                "protocolo": [
                    "Esperar 4 a 6 semanas tras el inicio (un 10-15% de pacientes experimenta tolerancia parcial al efecto sexual).",
                    "Reducir la dosis si la remisión depresiva está consolidada.",
                    "Rotar o coadyuvar con <b>Bupropión (150-300 mg/día)</b> o considerar cambio a <b>Vortioxetina</b>."
                ],
                "fuente": "Jufe, Stahl"
            }
        },
        "Donepezilo (Inhibidor de la Acetilcolinesterasa)": {
            "Incontinencia urinaria de urgencia o polaquiuria sobrevenida": {
                "nivel": "CRÍTICO",
                "titulo": "🚨 CASCADA FARMACOLÓGICA NEFRO-UROLÓGICA: DONEPEZILO → OXIBUTININA",
                "mecanismo": "El Donepezilo inhibe la acetilcolinesterasa incrementando la acetilcolina en todo el organismo. En la vejiga, estimula los receptores muscarínicos M2 y M3 del músculo detrusor, provocando contracciones involuntarias e incontinencia de urgencia.",
                "error_tipico": "Derivar a urología y prescribir anticolinérgicos urológicos como Oxibutinina, Tolterodina o Solifenacina para tratar la incontinencia.",
                "riesgo_iatrogenico": "¡ANTAGONISMO FARMACOLÓGICO FRONTAL! Los anticolinérgicos urológicos atraviesan la barrera hematoencefálica y BLOQUEAN los receptores colinérgicos cerebrales, aniquilando el beneficio cognitivo del Donepezilo y acelerando el deterioro de la demencia y el delirio confusional.",
                "protocolo": [
                    "<b>Desprescribir de inmediato</b> cualquier fármaco anticolinérgico urológico (Oxibutinina, Tolterodina).",
                    "Reconocer la incontinencia como un efecto adverso colinérgico del inhibidor de acetilcolinesterasa.",
                    "Reducir la dosis de Donepezilo (de 10 mg a 5 mg/día) o evaluar cambio a parches transdérmicos de Rivastigmina (menor pico plasmático periférico).",
                    "Si el tratamiento urológico es estrictamente ineludible, utilizar un agonista de receptores beta-3 adrenérgicos (<b>Mirabegrón</b>), el cual relaja el detrusor vesical sin ninguna actividad anticolinérgica en el sistema nervioso central."
                ],
                "fuente": "Criterios STOPP/START para Pacientes Ancianos, Stahl, Jufe"
            },
            "Bradicardia sintomática, mareos ortostáticos o síncope": {
                "nivel": "CRÍTICO",
                "titulo": "🛑 RIESGO CARDIOLÓGICO: BRADICARDIA COLINÉRGICA Y SÍNCOPE VAGAL",
                "mecanismo": "El exceso colinérgico estimula el tono vagal sobre los receptores muscarínicos M2 del nódulo sinoauricular y auriculoventricular cardíaco.",
                "error_tipico": "Interpretar el síncope como un 'accidente cerebrovascular isquémico transitorio' o una 'caída accidental del anciano' e iniciar antiagregantes o implante de marcapasos sin retirar el fármaco.",
                "riesgo_iatrogenico": "Caídas graves, fracturas de cadera, traumatismo craneoencefálico e intervenciones invasivas innecesarias.",
                "protocolo": [
                    "Realizar electrocardiograma basal para evaluar frecuencia cardíaca y descartar bloqueo auriculoventricular o intervalo PR prolongado.",
                    "Suspender o reducir la dosis de Donepezilo si la frecuencia cardíaca desciende de 50 lpm o existen síntomas presincopales.",
                    "Evitar la combinación con otros bradicardizantes como betabloqueantes o calcioantagonistas no dihidropiridínicos (Verapamilo, Diltiazem)."
                ],
                "fuente": "Jufe, Stahl, Criterios STOPP/START"
            }
        }
    }

    col_farmaco, col_sintoma = st.columns(2)

    farmaco_paciente = col_farmaco.selectbox(
        "Fármaco Primario del Paciente:",
        options=list(CASCADAS_CLINICAS.keys()),
        key="cascade_farmaco_select"
    )

    sintomas_posibles = list(CASCADAS_CLINICAS[farmaco_paciente].keys())

    sintoma_emergente = col_sintoma.selectbox(
        "Nuevo Síntoma o Signo Clínico Emergente:",
        options=sintomas_posibles,
        key="cascade_sintoma_select"
    )

    # Renderizado reactivo inmediato de la evaluación clínica
    regla = CASCADAS_CLINICAS.get(farmaco_paciente, {}).get(sintoma_emergente)

    if regla:
        css_class = "alert-critical" if regla["nivel"] == "CRÍTICO" else ("alert-warning" if regla["nivel"] == "ADVERTENCIA" else "alert-safe")
        badge_bg = "#dc2626" if regla["nivel"] == "CRÍTICO" else ("#f59e0b" if regla["nivel"] == "ADVERTENCIA" else "#16a34a")
        badge_label = "ALERTA CRÍTICA" if regla["nivel"] == "CRÍTICO" else ("PRECAUCIÓN CLÍNICA" if regla["nivel"] == "ADVERTENCIA" else "MANEJO RACIONAL")

        protocolo_html = "".join(f"<li style='margin-bottom: 6px;'>{item}</li>" for item in regla["protocolo"])

        st.markdown(f"""
        <div class="{css_class}" style="padding: 16px 20px; border-radius: 8px; margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 6px;">
                <span style="background-color: {badge_bg}; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.5px;">{badge_label}</span>
                <span style="color: #64748b; font-size: 0.82rem; font-weight: 600;">📚 {regla.get('fuente', '')}</span>
            </div>
            <h3 style="margin-top: 4px; margin-bottom: 12px; font-size: 1.18rem; font-weight: 700;">{regla["titulo"]}</h3>
            <p style="margin-bottom: 8px;"><b>🩺 Mecanismo Fisiopatológico:</b> {regla["mecanismo"]}</p>
            <p style="margin-bottom: 8px;"><b>❌ Error Típico de Prescripción (Cascada):</b> {regla["error_tipico"]}</p>
            <p style="margin-bottom: 8px;"><b>⚠️ Riesgo Iatrogénico Documentado:</b> {regla["riesgo_iatrogenico"]}</p>
            <hr style="margin: 12px 0; border: none; border-top: 1px solid rgba(0,0,0,0.12);">
            <h4 style="margin-bottom: 8px; font-size: 1.0rem; font-weight: 700;">📋 Protocolo Sofia de Desprescripción y Optimización Racional:</h4>
            <ol style="margin-bottom: 4px; padding-left: 22px; line-height: 1.55;">
                {protocolo_html}
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📌 Las 3 Preguntas Clave del Algoritmo de Detección de Cascadas (Criterios STOPP/START y Sofia)"):
        st.markdown("""
        Ante cualquier paciente polimedicado que refiera un nuevo motivo de consulta o síntoma emergente:
        1. **¿Cuándo comenzó exactamente el síntoma actual?** Verificar si coincide temporalmente con la introducción o escalada de dosis de un fármaco previo.
        2. **¿Qué modificaciones se hicieron en la medicación en las 8 semanas previas?** Considerar siempre el tiempo de latencia y la acumulación plasmática.
        3. **¿Puede la farmacología del fármaco causal explicar este nuevo cuadro clínico?** Analizar perfiles de afinidad por receptores (D2, H1, 5-HT2C, M1-M4) antes de diagnosticar una nueva enfermedad o prescribir un segundo fármaco.
        """)



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
