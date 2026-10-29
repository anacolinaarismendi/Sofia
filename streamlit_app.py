"""
streamlit_app.py
Aplicación Web Interactiva de Sofia: Asistente Clínico Inteligente y RAG de Psicofarmacología.
Plataforma interactiva de soporte a la decisión clínica y farmacovigilancia.
"""

import os
import re
import sys
import traceback
from html import escape
from pathlib import Path
from typing import Optional
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
    from src.enrichers.atc_enricher import ATCEnricher, normalizar_clave
except Exception:
    class ATCEnricher:  # type: ignore
        CATALOGO_ATC = {}
        @classmethod
        def get_info(cls, *args, **kwargs):
            return None
        @classmethod
        def obtener_ficha(cls, *args, **kwargs):
            return None

    def normalizar_clave(texto: str) -> str:
        return str(texto).strip().lower()


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
def _leer_secreto(nombre: str) -> str:
    """Lee un valor de st.secrets (Streamlit Cloud) o, en su defecto, de variables de entorno.

    En local, sin .streamlit/secrets.toml, st.secrets lanza una excepción: se ignora
    y la app arranca normalmente con Ollama local / modo contingencia.
    """
    valor = ""
    try:
        valor = str(st.secrets.get(nombre, "")).strip()
    except Exception:
        valor = ""
    if not valor:
        valor = os.environ.get(nombre, "").strip()
    return valor


def resolver_credenciales_ollama() -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Devuelve (api_key, host, modelo). Cualquiera puede ser None => el motor usa sus valores por defecto."""
    key = _leer_secreto("OLLAMA_API_KEY")
    if key in ("", "tu_ollama_api_key_aqui", "tu_clave_real_aqui"):
        key = ""
    host = _leer_secreto("OLLAMA_HOST")
    modelo = _leer_secreto("OLLAMA_MODEL")

    # Con API key y sin host explícito, lo razonable es Ollama Cloud
    if key and not host:
        host = "https://ollama.com"

    return (key or None), (host or None), (modelo or None)


def resolver_credenciales_groq() -> tuple[Optional[str], Optional[str]]:
    """Devuelve (api_key, modelo) para inferencia cloud de Groq."""
    key = _leer_secreto("GROQ_API_KEY")
    if key in ("", "tu_groq_api_key_aqui", "gsk_tu_clave_aqui"):
        key = ""
    modelo = _leer_secreto("GROQ_MODEL") or "qwen/qwen3.8-27b"
    return (key or None), (modelo or None)


@st.cache_resource(show_spinner="Inicializando base de conocimiento clínica de Sofia...")
def get_rag_engine(
    api_key: Optional[str] = None,
    host: Optional[str] = None,
    model: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    groq_model: Optional[str] = None
):
    return ClinicalRAGEngine(
        ollama_host=host,
        llm_model=model,
        api_key=api_key,
        groq_api_key=groq_api_key,
        groq_model=groq_model
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


def render_modo_respuesta(modo, proveedor=None):
    """Indica visiblemente cómo se generó la respuesta (LLM, plantilla sin LLM, sin cobertura o error)."""
    if modo == "llm":
        prov_txt = f" [{proveedor}]" if proveedor else ""
        st.caption(f"🤖 Respuesta generada por el modelo de lenguaje{prov_txt} a partir de los documentos recuperados.")
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
# Inicialización del motor RAG (fuera del sidebar)
# -----------------------------------------------------------------------------
_api_key_cfg, _host_cfg, _model_cfg = resolver_credenciales_ollama()
_groq_key_cfg, _groq_model_cfg = resolver_credenciales_groq()
rag_engine = get_rag_engine(
    api_key=_api_key_cfg,
    host=_host_cfg,
    model=_model_cfg,
    groq_api_key=_groq_key_cfg,
    groq_model=_groq_model_cfg
)
data = load_datasets()

# Inspección segura de atributos del motor
_groq_key = getattr(rag_engine, "groq_api_key", None)
_groq_model = getattr(rag_engine, "groq_model", "qwen/qwen3.8-27b")
es_groq = bool(_groq_key)

_api_key = getattr(rag_engine, "api_key", None)
_host = getattr(rag_engine, "ollama_host", None) or ""
_modelo = getattr(rag_engine, "llm_model", "N/D")
_usa_ollama = getattr(getattr(rag_engine, "vector_store", None), "use_ollama", False)
_modo_idx = getattr(getattr(rag_engine, "vector_store", None), "modo_indice", "N/D")
es_ollama_cloud = "ollama.com" in _host
key_faltante = not _api_key


# -----------------------------------------------------------------------------
# Barra Lateral (Sidebar)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=64)
    st.markdown("## **Sofia Clinical RAG**")
    st.caption("Asistente de Soporte a la Decisión y Farmacovigilancia en Psiquiatría")

    st.markdown("---")
    st.markdown("### 🔌 Estado del motor")

    if es_groq:
        st.success(f"⚡ Groq Cloud activo (`{_groq_model}`)")
    elif es_ollama_cloud and key_faltante:
        st.warning("⚠️ Ollama Cloud sin API Key: operando en modo contingencia.")
    elif es_ollama_cloud:
        st.success(f"☁️ Ollama Cloud activo (`{_modelo}`)")
    elif _usa_ollama:
        st.success(f"🟢 Ollama Local activo (`{_modelo}`)")
    else:
        st.info("🟡 Inferencia contingencia: TF-IDF + fichas estructuradas")
    st.caption(
        "Recuperación: "
        + ("semántica densa (embeddings)" if _modo_idx == "ollama" else "léxica TF-IDF")
        + ". La disponibilidad del LLM se indica en cada respuesta."
    )

    num_docs = len(getattr(rag_engine, "docs", []))
    st.metric("Base de Conocimiento", f"{num_docs} Documentos", delta="pre_training + CANMAT + marcas")
    if num_docs == 0:
        st.error("🔴 La base de conocimiento está vacía: el RAG no puede responder. Revisa rag_knowledge_base.json y los logs.")
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
            render_modo_respuesta(msg.get("modo"), proveedor=msg.get("proveedor"))
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
                if not isinstance(resultado, dict):
                    resultado = {"respuesta": str(resultado), "modo_respuesta": "error", "documentos_recuperados": []}
                respuesta_texto = resultado.get("respuesta", "")
                voice_snip = resultado.get("voice_snippet", "")
                fuentes = resultado.get("documentos_recuperados", [])
                modo = resultado.get("modo_respuesta") or ("llm" if resultado.get("generado_con_llm") else "contingencia")
                proveedor_llm = resultado.get("proveedor_llm")

                st.markdown(respuesta_texto)
                render_modo_respuesta(modo, proveedor=proveedor_llm)

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
            "proveedor": proveedor_llm,
            "voice_snippet": voice_snip,
            "fuentes": fuentes
        })



# =============================================================================
# PESTAÑA 2: Matriz Interactiva CYP450
# =============================================================================
with tab_cyp:
    st.markdown("#### Matriz Dinámica de Citocromos y Semáforo de Interacciones")
    st.write("Selecciona dos o más psicofármacos del tratamiento del paciente para evaluar bloqueos enzimáticos o acumulación tóxica basados en el catálogo farmacológico completo.")

    if "jufe" in data and not data["jufe"].empty:
        farmacos_disponibles = sorted(data["jufe"]["farmaco"].dropna().unique().tolist())
    else:
        farmacos_disponibles = sorted([
            "Sertralina", "Fluoxetina", "Paroxetina", "Citalopram", "Escitalopram", "Fluvoxamina",
            "Venlafaxina", "Desvenlafaxina", "Duloxetina", "Bupropión", "Mirtazapina", "Vortioxetina",
            "Clozapina", "Risperidona", "Olanzapina", "Quetiapina", "Ziprasidona", "Aripiprazol", "Haloperidol",
            "Carbonato de Litio", "Ácido Valproico / Divalproato", "Carbamazepina", "Lamotrigina", "Topiramato",
            "Clonazepam", "Lorazepam", "Diazepam", "Alprazolam", "Donepezilo", "Memantina"
        ])

    seleccion = st.multiselect(
        "Fármacos en el esquema terapéutico:",
        options=farmacos_disponibles,
        default=["Paroxetina", "Risperidona"] if "Paroxetina" in farmacos_disponibles and "Risperidona" in farmacos_disponibles else farmacos_disponibles[:2]
    )

    if len(seleccion) < 2:
        st.info("💡 Selecciona al menos 2 fármacos para calcular los cruces enzimáticos.")
    else:
        st.markdown("##### 🚦 Diagnóstico Clínico de Interacciones")
        interacciones_detectadas = []

        # Normalizador de nombres en selección
        norm_sel = {normalizar_clave(f): f for f in seleccion}

        # Familias farmacológicas
        isrs_set = {"sertralina", "fluoxetina", "paroxetina", "citalopram", "escitalopram", "fluvoxamina"}
        isrs_presentes = [orig for norm, orig in norm_sel.items() if norm in isrs_set]

        bzd_set = {"clonazepam", "lorazepam", "diazepam", "alprazolam", "bromazepam", "clorazepato", "midazolam", "clobazam"}
        bzd_presentes = [orig for norm, orig in norm_sel.items() if norm in bzd_set]

        # Citocromo CYP2D6
        inh_2d6_potentes = {"paroxetina", "bupropion", "fluoxetina"}
        inh_2d6_moderados = {"duloxetina", "sertralina"}
        sus_2d6_criticos = {"risperidona", "haloperidol", "aripiprazol", "vortioxetina", "atomoxetina", "venlafaxina", "amitriptilina", "clomipramina", "imipramina", "nortriptilina", "donepezilo"}

        inh_2d6_activos = [orig for norm, orig in norm_sel.items() if norm in inh_2d6_potentes]
        sus_2d6_activos = [orig for norm, orig in norm_sel.items() if norm in sus_2d6_criticos]

        # Citocromo CYP1A2
        inh_1a2_potentes = {"fluvoxamina"}
        sus_1a2_criticos = {"clozapina", "olanzapina", "agomelatina", "duloxetina"}
        inh_1a2_activos = [orig for norm, orig in norm_sel.items() if norm in inh_1a2_potentes]
        sus_1a2_activos = [orig for norm, orig in norm_sel.items() if norm in sus_1a2_criticos]

        # Inductor multienzimático mayor (Carbamazepina)
        es_carbamazepina = any("carbamazepina" in norm for norm in norm_sel)

        # Litio
        es_litio = any("litio" in norm for norm in norm_sel)

        # Lamotrigina y Valproato
        es_lamotrigina = any("lamotrigina" in norm for norm in norm_sel)
        es_valproato = any("valpro" in norm for norm in norm_sel)

        # Clozapina
        es_clozapina = any("clozapina" in norm for norm in norm_sel)

        # Regla 1: Paroxetina + Bupropión
        if any("paroxetina" in norm for norm in norm_sel) and any("bupropion" in norm for norm in norm_sel):
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

        # Regla 2: Duplicidad Terapéutica de ISRS
        if len(isrs_presentes) >= 2:
            interacciones_detectadas.append({
                "nivel": "CRÍTICO",
                "fuente": None,
                "titulo": f"🔴 Duplicidad Terapéutica Inadecuada de ISRS ({' + '.join(sorted(isrs_presentes))})",
                "descripcion": (
                    "**Mecanismo:** La prescripción simultánea de dos o más fármacos de la misma familia (ISRS) carece de fundamento clínico "
                    "basado en evidencia y no aporta beneficio sinérgico en depresión mayor o trastornos de ansiedad.\n\n"
                    "**Riesgo Clínico:** Multiplica de forma inmediata el riesgo de **Síndrome Serotoninérgico** potencialmente letal "
                    "(hipertermia, clonus, mioclonías, inestabilidad autonómica, agitación psicomotriz) y riesgo hemorrágico por depleción plaquetaria de serotonina.\n\n"
                    "**Conducta Racional:** Contraindicado. Realizar cambio escalonado con período de retirada o lavado adecuado si se planea una rotación."
                )
            })

        # Regla 3: Duplicidad de Benzodiazepinas
        if len(bzd_presentes) >= 2:
            interacciones_detectadas.append({
                "nivel": "ADVERTENCIA",
                "fuente": None,
                "titulo": f"🟡 Duplicidad Terapéutica de Benzodiazepinas ({' + '.join(sorted(bzd_presentes))})",
                "descripcion": (
                    "**Mecanismo:** Coadministración de dos o más benzodiazepinas sobre el mismo sitio alostérico del receptor GABA-A.\n\n"
                    "**Riesgo Clínico:** Potenciación de sedación profunda, enlentecimiento psicomotor, deterioro de memoria anterógrada, riesgo elevado de caídas/fracturas y rápida inducción de tolerancia y dependencia farmacológica.\n\n"
                    "**Conducta Racional:** Racionalizar a una única benzodiazepina ajustando la dosis y planificar la pauta de desprescripción gradual."
                )
            })

        # Regla 4: Inhibición potente de CYP2D6 + Sustrato 2D6
        if inh_2d6_activos and sus_2d6_activos:
            interacciones_detectadas.append({
                "nivel": "CRÍTICO",
                "fuente": "INTER-CYP2D6-001",
                "titulo": f"🔴 Bloqueo Enzimático Potente de CYP2D6 ({', '.join(sorted(inh_2d6_activos))} + {', '.join(sorted(sus_2d6_activos))})",
                "descripcion": (
                    f"**Mecanismo:** {', '.join(sorted(inh_2d6_activos))} inhibe potentemente el CYP2D6, provocando una elevación "
                    f"marcada de las concentraciones plasmáticas de {', '.join(sorted(sus_2d6_activos))}.\n\n"
                    f"**Riesgo Clínico:** Aparición de síntomas extrapiramidales agudos, sedación excesiva, hipotensión y prolongación del intervalo QTc.\n\n"
                    f"**Pauta:** Reducir la dosis del sustrato en un 50% o rotar hacia un antidepresivo con mínimo impacto 2D6 (ej. Sertralina o Escitalopram)."
                )
            })

        # Regla 5: Inhibición masiva de CYP1A2 (Fluvoxamina) + Sustratos 1A2
        if inh_1a2_activos and sus_1a2_activos:
            interacciones_detectadas.append({
                "nivel": "CRÍTICO",
                "fuente": "INTER-CYP1A2-002",
                "titulo": f"🔴 Bloqueo Crítico de CYP1A2 ({', '.join(sorted(inh_1a2_activos))} + {', '.join(sorted(sus_1a2_activos))})",
                "descripcion": (
                    f"**Mecanismo:** Fluvoxamina es un inhibidor extremadamente potente de CYP1A2. Al combinarse con {', '.join(sorted(sus_1a2_activos))}, "
                    f"multiplica su área bajo la curva (AUC) entre 3 y 5 veces.\n\n"
                    f"**Riesgo Clínico:** En el caso de Clozapina u Olanzapina, precipita crisis convulsivas, sedación profunda y colapso cardiovascular. Con Agomelatina o Duloxetina multiplica el riesgo de hepatotoxicidad severa.\n\n"
                    f"**Pauta:** Combinación desaconsejada. Si es imprescindible con Clozapina, reducir la dosis de Clozapina a una tercera o cuarta parte con clozapinemia obligatoria."
                )
            })

        # Regla 6: Clozapina + Carbamazepina (Contraindicación Absoluta)
        if es_clozapina and es_carbamazepina:
            interacciones_detectadas.append({
                "nivel": "CRÍTICO",
                "fuente": "ALERTA-CLOZ-AGRAN-001",
                "titulo": "🛑 CONTRAINDICACIÓN ABSOLUTA: Mielosupresión Sinérgica (Clozapina + Carbamazepina)",
                "descripcion": (
                    "**Mecanismo y Riesgo:** Ambos fármacos poseen toxicidad medular intrínseca y riesgo de inducir agranulocitosis y anemia aplásica. "
                    "Su administración concomitante sinergiza la toxicidad sobre los precursores mieloides, elevando de forma crítica el riesgo de neutropenia letal.\n\n"
                    "**Conducta Racional:** ASOCIACIÓN PROHIBIDA. Si se requiere un estabilizador del ánimo junto a Clozapina, el fármaco de elección es el Carbonato de Litio (que estimula la granulopoyesis mediante GM-CSF) o el Ácido Valproico con hemogramas seriados."
                )
            })

        # Regla 7: Inducción Enzimática Masiva por Carbamazepina
        if es_carbamazepina:
            farmacos_inducidos = [orig for norm, orig in norm_sel.items() if not ("carbamazepina" in norm)]
            if farmacos_inducidos:
                interacciones_detectadas.append({
                    "nivel": "ADVERTENCIA",
                    "fuente": None,
                    "titulo": f"🟡 Inducción Metabólica Masiva por Carbamazepina sobre: {', '.join(sorted(farmacos_inducidos))}",
                    "descripcion": (
                        "**Mecanismo:** La carbamazepina es un inductor enzimático potente de CYP1A2, CYP2C9, CYP2C19, CYP3A4 y UGT glucuronidación. "
                        "Reduce entre un 40% y un 60% las concentraciones plasmáticas de antipsicóticos (Quetiapina, Olanzapina, Aripiprazol, Haloperidol), "
                        "antidepresivos, lamotrigina y benzodiazepinas.\n\n"
                        "**Riesgo Clínico:** Pérdida de eficacia antipsicótica o estabilizadora y recaída sintomática inadvertida.\n\n"
                        "**Pauta:** Monitorizar niveles plasmáticos y prever aumentos de dosis del fármaco coadministrado."
                    )
                })

        # Regla 8: Lamotrigina + Valproato (Glucuronidación)
        if es_lamotrigina and es_valproato:
            interacciones_detectadas.append({
                "nivel": "CRÍTICO",
                "fuente": "INTER-LAM-VAL-003, ALERTA-LAM-SJS-002",
                "titulo": "🔴 Inhibición Crítica de Glucuronidación (Valproato + Lamotrigina)",
                "descripcion": (
                    "**Interacción:** El ácido valproico inhibe fuertemente la glucuronidación (UGT1A4/2B7) de la lamotrigina, "
                    "duplicando con creces su semivida plasmática y aumentando drásticamente la incidencia de erupciones cutáneas graves "
                    "y Síndrome de Stevens-Johnson.\n\n"
                    "**Pauta Obligatoria:** Reducir la pauta de titulación de lamotrigina a menos del 50% de la dosis estándar habitual (comenzar con 25 mg en días alternos)."
                )
            })

        # Regla 9: Litio e interacciones cruzadas
        if es_litio:
            if any("haloperidol" in norm for norm in norm_sel):
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

            if isrs_presentes:
                interacciones_detectadas.append({
                    "nivel": "ADVERTENCIA",
                    "fuente": None,
                    "titulo": f"🟡 Potenciación Serotoninérgica y Temblor (Litio + {', '.join(sorted(isrs_presentes))})",
                    "descripcion": (
                        "**Interacción Farmacodinámica:** El litio estimula la neurotransmisión serotoninérgica central. Aunque es una estrategia válida de adición "
                        "en depresión resistente, eleva la frecuencia de temblor postural fino, molestias digestivas y riesgo de toxicidad serotoninérgica.\n\n"
                        "**Pauta:** Controlar periódicamente la litemia basal y vigilar síntomas autonómicos o neuromusculares."
                    )
                })

            if es_clozapina:
                interacciones_detectadas.append({
                    "nivel": "ADVERTENCIA",
                    "fuente": None,
                    "titulo": "🟡 Monitorización Neurológica Estricta (Litio + Clozapina)",
                    "descripcion": (
                        "**Riesgo:** Ambos agentes disminuyen el umbral convulsivo y aumentan el riesgo de mioclonías, confusión y síndrome neuroléptico maligno. Como ventaja, el litio estimula la granulopoyesis protegiendo frente a neutropenia.\n\n"
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

        # Regla 10: Clozapina (CYP1A2)
        if es_clozapina and not inh_1a2_activos:
            interacciones_detectadas.append({
                "nivel": "ADVERTENCIA",
                "fuente": "MONO-CLOZAPINA-006, INTER-CYP1A2-002",
                "titulo": "🟡 Monitorización Estricta de Clozapina (Sustrato Principal CYP1A2)",
                "descripcion": (
                    "**Vía Metabólica:** Clozapina depende en un 70% de CYP1A2. Fármacos inhibidores elevan su nivel hacia el rango convulsivo; "
                    "los hidrocarburos del humo del tabaco inducen CYP1A2 y reducen sus niveles hasta un 50%. Requiere hemograma seriado obligatorio."
                )
            })

        if not interacciones_detectadas:
            st.markdown(
                f'<div class="alert-warning" style="background-color: #f8fafc; border-left: 5px solid #64748b; color: #1e293b; padding: 14px 18px; border-radius: 6px;">'
                f'<h4>⚪ Sin Interacciones Farmacocinéticas Críticas Detectadas</h4>'
                f'<p>No se evidencian colisiones enzimáticas primarias de citocromos o duplicidades directas para la combinación: <b>{", ".join(seleccion)}</b>.</p>'
                f'<p style="margin-bottom: 0;"><b>Aviso de seguridad clínica:</b> La ausencia de una alerta automática <u>no garantiza la inocuidad</u> del esquema. '
                f'Verifique la función renal y hepática del paciente, consulte las monografías de Jufe y Stahl o plantee la consulta a <b>Sofia</b> en el asistente RAG.</p>'
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
                    pie = '<p><small>⚠️ Regla farmacocinética clínica validada: contrastar con la ficha técnica o literatura antes de actuar.</small></p>'
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
        df_base = data["canmat"][data["canmat"]["patologia"] == filtro_patologia]
        fases_disp = ["Todas"] + sorted(df_base["fase"].dropna().unique().tolist())
        col_fase, col_busq = st.columns([1, 2])
        fase_sel = col_fase.selectbox("Filtrar por Fase Clínica:", options=fases_disp)
        texto_busq = col_busq.text_input("Filtrar por fármaco o estrategia:", value="")

        df_filtrado = df_base if fase_sel == "Todas" else df_base[df_base["fase"] == fase_sel]
        if texto_busq.strip():
            df_filtrado = df_filtrado[df_filtrado["farmaco"].str.contains(texto_busq.strip(), case=False, na=False) |
                                      df_filtrado["comentarios"].str.contains(texto_busq.strip(), case=False, na=False)]

        columnas = [c for c in ["fase", "linea", "estrategia", "farmaco", "atc", "nivel_evidencia", "comentarios"]
                    if c in df_filtrado.columns]
        st.dataframe(
            df_filtrado[columnas],
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
