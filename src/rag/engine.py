"""
engine.py
Motor RAG Clínico y Base Vectorial de Sofia:
- Ingesta estructurada del corpus pre_training/ (Markdown con YAML Frontmatter y Voice Snippets).
- Almacén vectorial dual (Ollama embeddinggemma:latest con fallback automático TF-IDF).
- Directrices clínicas formalizadas (Clinical System Prompt con Grounding estricto y trazabilidad [DOC-ID]).
- Generación de respuestas médicas estructuradas en 4 apartados con soporte dual verbal/escrito.
"""

import os
import sys
import time
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import requests

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None
    SKLEARN_AVAILABLE = False


# Directorio base portable
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)

# Umbral mínimo de similitud coseno TF-IDF por debajo del cual el corpus se considera
# sin cobertura para la consulta (calibrado empíricamente sobre consultas dentro/fuera de alcance).
# Es una heurística léxica: no sustituye el control de grounding del LLM.
MIN_SCORE_TFIDF = 0.12

# Una respuesta cortada por num_predict se detecta con done_reason == "length".
NUM_PREDICT = 1200

MENSAJE_SIN_COBERTURA = (
    "**Sin cobertura en el corpus clínico de Sofia.**\n\n"
    "No se han encontrado documentos suficientemente relevantes para esta consulta en el corpus curado "
    "(psicofármacos y sus interacciones/guías indexadas). Por seguridad clínica Sofia no emite una "
    "recomendación sin respaldo documental.\n\n"
    "Reformula la consulta indicando el principio activo o la interacción concreta, o consulta la "
    "ficha técnica oficial / un especialista."
)

SOFIA_CLINICAL_PROMPT = """Eres Sofia, asistente clínica especializada en Psicofarmacología Clínica, Psiquiatría Basada en la Evidencia y Farmacovigilancia.
Tu función es asesorar a médicos adjuntos, residentes y profesionales de la salud con el máximo rigor científico, basándote EXCLUSIVAMENTE en el CONTEXTO CLÍNICO RECUPERADO.

### DIRECTRICES DE RAZONAMIENTO Y SEGURIDAD CLÍNICA:
1. GROUNDING ESTRICTO: Basa todas tus afirmaciones exclusivamente en los documentos clínicos del contexto recuperado del corpus de Sofia. Si el contexto no contiene datos suficientes para responder con seguridad, declara explícitamente la limitación sin inventar datos.
2. TRAZABILIDAD Y CITACIÓN POR IDENTIFICADOR ([DOC-ID]): No atribuyas citas textuales directas a autores personales (NO uses «Según la Dra. Jufe...» ni «El Dr. Stahl describe...»). En su lugar, debes respaldar cada afirmación farmacológica, posológica o de seguridad citando entre corchetes el código identificador único de un documento que aparezca en el CONTEXTO CLÍNICO RECUPERADO (formato de ejemplo: [MONO-SERTRALINA-001], [INTER-CYP2D6-001], [CASC-HAL-BIP-001], [GUIA-CANMAT-MANIA-001]). NUNCA cites un identificador que no figure en el contexto recuperado.
3. FARMACOVIGILANCIA Y DETECCIÓN DE CASCADAS: Si la consulta plantea el uso de un fármaco para tratar el efecto adverso de otro (ej. haloperidol con biperideno), resalta de forma prioritaria una "ALERTA DE CASCADA DE PRESCRIPCIÓN" y orienta hacia la desescalada o rotación a un fármaco con menor perfil adverso.
4. METABOLISMO ENZIMÁTICO Y CITOCROMOS: Especifica siempre las vías metabólicas y los citocromos involucrados (CYP1A2, CYP2D6, CYP3A4, CYP2B6, glucuronidación UGT1A4 o eliminación renal de litio).

### FORMATO DE RESPUESTA CLÍNICA OBLIGATORIO:
Estructura tu respuesta en los siguientes 4 apartados numerados:
- **1. Juicio Clínico Inmediato:** Conclusión directa y aplicable a la consulta en 2-3 líneas.
- **2. Mecanismo Farmacológico y Cinética:** Explicación fisiopatológica, receptores involucrados (D2, 5-HT, etc.) y/o perfil enzimático CYP450.
- **3. Pauta Operativa y Recomendaciones:** Dosis sugeridas, velocidad de titulación/desescalada y parámetros obligatorios de monitorización analítica o clínica.
- **4. Alertas de Seguridad y Trazabilidad Comercial:** Advertencias críticas (caja negra, litemia, agranulocitosis) y nombres comerciales / laboratorios aplicables.
"""

SOFIA_SOUL_PROMPT = SOFIA_CLINICAL_PROMPT  # Alias retrocompatible



def cargar_corpus_pretraining(directorio: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Carga e indexa los documentos clínicos en Markdown desde pre_training/."""
    dir_path = directorio or (WORKSPACE_ROOT / "pre_training")
    documentos: List[Dict[str, Any]] = []

    if not dir_path.exists():
        return documentos

    for f in sorted(dir_path.glob("**/*.md")):
        if f.name.startswith("00_"):
            continue
        try:
            with open(f, "r", encoding="utf-8") as arch:
                texto = arch.read()
        except Exception:
            continue

        parts = texto.split("---", 2)
        meta: Dict[str, Any] = {}
        cuerpo = texto
        if len(parts) >= 3:
            raw_yaml = parts[1]
            cuerpo = parts[2].strip()
            for line in raw_yaml.strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"').strip("'")

        # Extraer Voice Snippet (< 40s) si existe en la sección 1
        voice_snippet = ""
        match_voice = re.search(r"## 1\. Resumen Ejecutivo Verbalizable.*?\n>(.*?)(?=\n##|\Z)", cuerpo, re.DOTALL)
        if match_voice:
            voice_snippet = match_voice.group(1).replace("\n>", " ").replace('"', '').strip()

        doc_id = meta.get("id", f.stem)
        tipo = meta.get("tipo", f.parent.name)
        titulo = meta.get("titulo", meta.get("farmaco", f.stem.replace("_", " ").title()))
        atc = meta.get("atc_codigo", "N/D")

        # Ruta relativa limpia al proyecto
        try:
            rel_path = str(f.relative_to(WORKSPACE_ROOT))
        except Exception:
            rel_path = f.name

        documentos.append({
            "id": doc_id,
            "tipo": tipo,
            "titulo": titulo,
            "atc": atc,
            "voice_snippet": voice_snippet,
            "contenido": cuerpo,
            "archivo_origen": rel_path
        })

    return documentos


class ClinicalVectorStore:
    """Almacén vectorial clínico híbrido con Ollama (local o cloud) y fallback local TF-IDF."""

    def __init__(
        self,
        ollama_host: Optional[str] = None,
        embedding_model: str = "embeddinggemma:latest",
        api_key: Optional[str] = None
    ):
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY")
        env_host = os.environ.get("OLLAMA_HOST")
        if ollama_host:
            self.ollama_host = ollama_host
        elif env_host:
            self.ollama_host = env_host
        elif self.api_key:
            self.ollama_host = "https://ollama.com"
        else:
            self.ollama_host = "http://localhost:11434"

        self.embedding_model = embedding_model
        self.documentos: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.use_ollama = False
        self._verificar_ollama()
        self.modo_indice = "ollama" if self.use_ollama else ("tfidf" if SKLEARN_AVAILABLE else "tokens")

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _verificar_ollama(self) -> bool:
        """Comprueba disponibilidad del host de Ollama (local o cloud)."""
        try:
            r = requests.get(f"{self.ollama_host}/api/tags", headers=self._get_headers(), timeout=2.5)
            if r.status_code == 200:
                tags = [m.get("name", "") for m in r.json().get("models", [])]
                self.use_ollama = any(self.embedding_model in t for t in tags)
                return self.use_ollama
        except Exception:
            pass
        self.use_ollama = False
        return False

    def indexar_documentos(self, docs: List[Dict[str, Any]]):
        """Indexa los documentos clínicos extrayendo representaciones vectoriales."""
        self.documentos = docs
        textos = [f"{d['titulo']} {d.get('atc', '')}\n{d['contenido']}" for d in docs]

        # Inicializar TF-IDF con normalización unicode de acentos
        if SKLEARN_AVAILABLE and TfidfVectorizer is not None:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=10000,
                strip_accents="unicode"
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(textos)
        else:
            self.vectorizer = None
            self.tfidf_matrix = None
            self.doc_tokens = [set(re.findall(r"\w+", t.lower())) for t in textos]

        if self.use_ollama:
            try:
                self.embeddings = np.array(self._embeber_textos(textos))
                self.modo_indice = "ollama"
            except Exception as exc:
                logger.warning("Fallo al generar embeddings con Ollama; se usa TF-IDF: %s", exc)
                self.use_ollama = False
                self.embeddings = None
                self.modo_indice = "tfidf" if self.vectorizer is not None else "tokens"
        else:
            self.modo_indice = "tfidf" if self.vectorizer is not None else "tokens"

    def _embeber_textos(self, textos: List[str], max_chars: int = 6000) -> List[List[float]]:
        """Obtiene embeddings por lote (/api/embed) con fallback al endpoint heredado (/api/embeddings)."""
        recortados = [t[:max_chars] for t in textos]
        try:
            res = requests.post(
                f"{self.ollama_host}/api/embed",
                headers=self._get_headers(),
                json={"model": self.embedding_model, "input": recortados},
                timeout=120
            )
            if res.status_code == 200:
                embs = res.json().get("embeddings")
                if embs and len(embs) == len(recortados):
                    return embs
            logger.info("/api/embed no disponible (status %s); se usa /api/embeddings", res.status_code)
        except Exception as exc:
            logger.info("/api/embed falló (%s); se usa /api/embeddings", exc)

        emb_list = []
        for t in recortados:
            res = requests.post(
                f"{self.ollama_host}/api/embeddings",
                headers=self._get_headers(),
                json={"model": self.embedding_model, "prompt": t},
                timeout=30
            )
            if res.status_code != 200:
                raise RuntimeError(f"Error status {res.status_code}")
            emb_list.append(res.json()["embedding"])
        return emb_list

    def buscar(self, query: str, top_k: int = 4, filtro_tipo: Optional[str] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Recupera los top_k documentos clínicos más relevantes para la consulta."""
        if not self.documentos:
            return []

        docs_candidatos = self.documentos
        indices_candidatos = list(range(len(docs_candidatos)))

        if filtro_tipo:
            indices_candidatos = [i for i, d in enumerate(docs_candidatos) if d.get("tipo") == filtro_tipo]
            if not indices_candidatos:
                indices_candidatos = list(range(len(docs_candidatos)))

        # Búsqueda densa con Ollama si está disponible y activa
        if self.modo_indice == "ollama" and self.use_ollama and self.embeddings is not None and cosine_similarity is not None:
            try:
                res = requests.post(
                    f"{self.ollama_host}/api/embeddings",
                    headers=self._get_headers(),
                    json={"model": self.embedding_model, "prompt": query},
                    timeout=5
                )
                if res.status_code == 200:
                    q_emb = np.array(res.json()["embedding"]).reshape(1, -1)
                    sims = cosine_similarity(q_emb, self.embeddings[indices_candidatos])[0]
                    top_idx_rel = np.argsort(sims)[::-1][:top_k]
                    return [(self.documentos[indices_candidatos[i]], float(sims[i])) for i in top_idx_rel]
            except Exception:
                pass

        # Fallback determinista local con TF-IDF si está disponible
        if self.vectorizer is not None and self.tfidf_matrix is not None and cosine_similarity is not None:
            q_vec = self.vectorizer.transform([query])
            sub_matrix = self.tfidf_matrix[indices_candidatos]
            sims = cosine_similarity(q_vec, sub_matrix)[0]
            top_idx_rel = np.argsort(sims)[::-1][:top_k]
            return [(self.documentos[indices_candidatos[i]], float(sims[i])) for i in top_idx_rel]
        else:
            # Fallback robusto sin scikit-learn (solapamiento de términos)
            q_tokens = set(re.findall(r"\w+", query.lower()))
            sims = []
            for i in indices_candidatos:
                doc_t = getattr(self, "doc_tokens", [])[i] if hasattr(self, "doc_tokens") and i < len(self.doc_tokens) else set()
                inter = len(q_tokens & doc_t)
                union = len(q_tokens | doc_t)
                score = (inter / union) if union > 0 else 0.0
                sims.append(score)
            sims_arr = np.array(sims)
            top_idx_rel = np.argsort(sims_arr)[::-1][:top_k]
            return [(self.documentos[indices_candidatos[i]], float(sims_arr[i])) for i in top_idx_rel]


class ClinicalRAGEngine:
    """Motor integral de RAG clínico de Sofia con soporte local, Ollama Cloud y groundedness estricto."""

    def __init__(
        self,
        ollama_host: Optional[str] = None,
        embedding_model: str = "embeddinggemma:latest",
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY")
        env_host = os.environ.get("OLLAMA_HOST")
        if ollama_host:
            self.ollama_host = ollama_host
        elif env_host:
            self.ollama_host = env_host
        elif self.api_key:
            self.ollama_host = "https://ollama.com"
        else:
            self.ollama_host = "http://localhost:11434"

        self.llm_model = llm_model or os.environ.get("OLLAMA_MODEL") or "qwen3.5:2b-mlx"
        self.vector_store = ClinicalVectorStore(
            ollama_host=self.ollama_host,
            embedding_model=embedding_model,
            api_key=self.api_key
        )

        kb_path = WORKSPACE_ROOT / "data" / "processed" / "rag_knowledge_base.json"
        self.kb_obsoleta = False
        if kb_path.exists():
            try:
                with open(kb_path, "r", encoding="utf-8") as f:
                    self.docs = json.load(f)
                pre_dir = WORKSPACE_ROOT / "pre_training"
                if pre_dir.exists():
                    kb_mtime = kb_path.stat().st_mtime
                    self.kb_obsoleta = any(
                        p.stat().st_mtime > kb_mtime for p in pre_dir.glob("**/*.md")
                    )
                    if self.kb_obsoleta:
                        logger.warning(
                            "rag_knowledge_base.json es anterior a algún documento de pre_training/: "
                            "regenerar la base de conocimiento."
                        )
            except Exception as exc:
                logger.warning("No se pudo leer rag_knowledge_base.json (%s); se carga pre_training/", exc)
                self.docs = cargar_corpus_pretraining()
        else:
            self.docs = cargar_corpus_pretraining()

        if self.docs:
            self.vector_store.indexar_documentos(self.docs)

    def construir_contexto(self, docs_recuperados: List[Tuple[Dict[str, Any], float]]) -> str:
        """Ensambla el contexto textual clínico con metadatos estructurados para el LLM."""
        bloques = []
        for i, (doc, score) in enumerate(docs_recuperados, 1):
            doc_id = doc.get("id", f"DOC-{i}")
            bloques.append(
                f"--- DOCUMENTO CLÍNICO [{doc_id}] ---\n"
                f"ID de Citación: [{doc_id}]\n"
                f"Título: {doc['titulo']}\n"
                f"Código ATC: {doc.get('atc', 'N/D')}\n"
                f"Tipo / Categoría: {doc.get('tipo', 'General')}\n"
                f"Score de Relevancia: {score:.4f}\n"
                f"Contenido:\n{doc['contenido']}\n"
            )
        return "\n\n".join(bloques)

    def consultar(self, pregunta: str, top_k: int = 3) -> Dict[str, Any]:
        """Ejecuta el pipeline completo de recuperación, ensamblado de contexto y generación."""
        t0_ret = time.time()
        docs_recuperados = self.vector_store.buscar(pregunta, top_k=top_k)
        lat_recuperacion = time.time() - t0_ret

        # Guardia de cobertura: con recuperación léxica (TF-IDF) no hay forma fiable de saber
        # si el corpus contiene la respuesta salvo por la similitud. Por debajo del umbral se
        # declara explícitamente la falta de cobertura en lugar de forzar un documento irrelevante.
        if self.vector_store.modo_indice != "ollama":
            top_score = docs_recuperados[0][1] if docs_recuperados else 0.0
            if not docs_recuperados or top_score < MIN_SCORE_TFIDF:
                return {
                    "pregunta": pregunta,
                    "respuesta": MENSAJE_SIN_COBERTURA,
                    "voice_snippet": "",
                    "generado_con_llm": False,
                    "modo_respuesta": "sin_cobertura",
                    "fuera_de_alcance": True,
                    "citas_no_recuperadas": [],
                    "documentos_recuperados": [],
                    "latencia_recuperacion_seg": round(lat_recuperacion, 4),
                    "latencia_generacion_seg": 0.0,
                    "latencia_total_seg": round(lat_recuperacion, 4)
                }

        t0_gen = time.time()
        contexto = self.construir_contexto(docs_recuperados)

        # Voice Snippet: solo de documentos con relevancia comparable a la del mejor resultado
        voice_snippet = ""
        mejor_score = docs_recuperados[0][1] if docs_recuperados else 0.0
        for doc, score in docs_recuperados:
            if doc.get("voice_snippet") and score >= 0.5 * mejor_score:
                voice_snippet = doc["voice_snippet"]
                break

        prompt_usuario = (
            f"### CONTEXTO CLÍNICO RECUPERADO:\n{contexto}\n\n"
            f"### CONSULTA CLÍNICA:\n{pregunta}\n\n"
            f"Genera tu respuesta clínica estructurada respetando estrictamente las 4 secciones numeradas "
            f"y citando los códigos identificadores de documento [DOC-ID] del contexto:"
        )

        respuesta_texto = ""
        generado_con_llm = False
        citas_no_recuperadas: List[str] = []

        # Intentar llamada a Ollama (local o Cloud)
        try:
            payload = {
                "model": self.llm_model,
                "system": SOFIA_CLINICAL_PROMPT,
                "prompt": prompt_usuario,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_predict": NUM_PREDICT
                }
            }
            res = requests.post(
                f"{self.ollama_host}/api/generate",
                headers=self.vector_store._get_headers(),
                json=payload,
                timeout=75
            )
            if res.status_code == 200:
                datos = res.json()
                raw_resp = datos.get("response", "").strip()
                if raw_resp:
                    # Limpiar bloques internos de pensamiento si están presentes
                    respuesta_texto = re.sub(r"<think>.*?</think>", "", raw_resp, flags=re.DOTALL).strip()
                    if not respuesta_texto:
                        respuesta_texto = raw_resp
                    generado_con_llm = True

                    # Validar que todo [DOC-ID] citado pertenezca a los documentos recuperados
                    ids_recuperados = {d.get("id", "").upper() for d, _ in docs_recuperados}
                    citados = re.findall(
                        r"\[((?:MONO|INTER|CASC|GUIA|ALERTA|MARCA|CYP|CANMAT)-[^\]\s]+)\]", respuesta_texto
                    )
                    citas_no_recuperadas = sorted({c for c in citados if c.upper() not in ids_recuperados})
                    if citas_no_recuperadas:
                        logger.warning("Citas fuera del contexto recuperado: %s", citas_no_recuperadas)
                        respuesta_texto += (
                            "\n\n> ⚠️ **Aviso de trazabilidad:** la respuesta cita identificadores que no figuran en los "
                            f"documentos recuperados ({', '.join(f'[{c}]' for c in citas_no_recuperadas)}). "
                            "Verifica esas referencias antes de usarlas."
                        )
                    if datos.get("done_reason") == "length":
                        respuesta_texto += (
                            "\n\n> ⚠️ **Respuesta truncada** por límite de longitud: revisa las alertas de seguridad "
                            "en los documentos recuperados."
                        )
            else:
                logger.warning("Ollama /api/generate devolvió status %s; se usa contingencia", res.status_code)
        except Exception as exc:
            logger.warning("Fallo al generar con Ollama; se usa contingencia: %s", exc)

        # Si Ollama no está disponible, estructurar respuesta determinista a partir de los documentos recuperados
        if not respuesta_texto and docs_recuperados:
            top_doc = docs_recuperados[0][0]
            top_id = top_doc.get("id", "DOC-001")
            otros_ids = [f"[{d.get('id')}]" for d, _ in docs_recuperados[1:] if d.get('id')]
            citas_adicionales = f" y referencias complementarias en {', '.join(otros_ids)}" if otros_ids else ""
            respuesta_texto = (
                f"**1. Juicio Clínico Inmediato:**\n"
                f"{voice_snippet or top_doc['titulo']} [{top_id}].\n\n"
                f"**2. Mecanismo Farmacológico y Cinética:**\n"
                f"Perfil farmacológico y enzimático validado en [{top_id}] (Código ATC: {top_doc.get('atc', 'N/D')}).\n\n"
                f"**3. Pauta Operativa y Recomendaciones:**\n"
                f"Seguir la dosificación, rango terapéutico y pauta de monitorización descritos en [{top_id}].\n\n"
                f"**4. Alertas de Seguridad y Trazabilidad Comercial:**\n"
                f"Advertencias críticas y precauciones farmacovigiladas documentadas en [{top_id}]{citas_adicionales}."
            )
        lat_generacion = time.time() - t0_gen

        return {
            "pregunta": pregunta,
            "respuesta": respuesta_texto,
            "voice_snippet": voice_snippet,
            "generado_con_llm": generado_con_llm,
            "modo_respuesta": "llm" if generado_con_llm else "contingencia",
            "fuera_de_alcance": False,
            "citas_no_recuperadas": citas_no_recuperadas,
            "documentos_recuperados": docs_recuperados,
            "latencia_recuperacion_seg": round(lat_recuperacion, 4),
            "latencia_generacion_seg": round(lat_generacion, 4),
            "latencia_total_seg": round(lat_recuperacion + lat_generacion, 4)
        }
