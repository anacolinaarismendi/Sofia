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

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None
    BM25_AVAILABLE = False

from .chunker import (
    tokenizar_clinico,
    crear_chunks_enriquecidos,
    agregar_chunks_a_documentos
)


# Directorio base portable
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)

SPANISH_STOPWORDS = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para",
    "con", "no", "una", "su", "al", "lo", "como", "mas", "pero", "sus", "le", "ya", "o",
    "este", "si", "porque", "esta", "son", "entre", "cuando", "muy", "sin", "sobre",
    "ser", "tiene", "tambien", "me", "hasta", "hay", "donde", "quien", "desde", "todo",
    "nos", "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante",
    "ellos", "e", "esto", "mi", "antes", "algunos", "unos", "yo", "otro", "otras",
    "otra", "tanto", "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual",
    "sea", "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros",
    "hacer", "paso", "cuales", "como", "cual"
]

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
4. METABOLISMO ENZIMÁTICO Y CITOCROMOS: Especifica las vías metabólicas y los citocromos involucrados (CYP1A2, CYP2D6, CYP3A4, CYP2B6, glucuronidación UGT1A4 o eliminación renal) ÚNICAMENTE si el contexto clínico recuperado los menciona explícitamente. No deduzcas ni agregues citocromos que no figuren en los documentos.

### FORMATO DE RESPUESTA CLÍNICA OBLIGATORIO:
Estructura tu respuesta en los siguientes 4 apartados numerados. Si para cualquiera de ellos no dispones de datos en el contexto clínico recuperado, escribe únicamente "No consta" para esa sección, evitando estrictamente inventar información para rellenar:
- **1. Juicio Clínico Inmediato:** Conclusión directa y aplicable a la consulta en 2-3 líneas (si no hay datos suficientes, escribe "No consta").
- **2. Mecanismo Farmacológico y Cinética:** Explicación fisiopatológica, receptores involucrados (D2, 5-HT, etc.) y perfil enzimático CYP450 (solo si el contexto los menciona explícitamente; de lo contrario, escribe "No consta").
- **3. Pauta Operativa y Recomendaciones:** Dosis sugeridas, velocidad de titulación/desescalada y parámetros obligatorios de monitorización analítica o clínica (si no constan en el contexto, escribe "No consta").
- **4. Alertas de Seguridad y Trazabilidad Comercial:** Advertencias críticas (caja negra, litemia, agranulocitosis) y nombres comerciales / laboratorios aplicables (si no constan en el contexto, escribe "No consta").
"""

SOFIA_SYSTEM_PROMPT = SOFIA_CLINICAL_PROMPT
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
    """
    Almacén vectorial clínico híbrido optimizado:
    - Segmentación semántica enriquecida con metadatos clínicos (Chunking estructural).
    - Señal léxica BM25 (rank-bm25) normalizada para nombres propios, citocromos y códigos ATC.
    - Señal densa neural (Ollama) con prefijos asimétricos o TF-IDF local de contingencia.
    - Fusión ponderada calibrada (alfa · semántica_recortada + (1 - alfa) · bm25_normalizado).
    - Caché persistente en disco (data/cache/embeddings_cache_*.npz) para arranque inmediato.
    """

    def __init__(
        self,
        ollama_host: Optional[str] = None,
        embedding_model: str = "embeddinggemma:latest",
        api_key: Optional[str] = None,
        alfa_hibrido: float = 0.65
    ):
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY")
        env_host = os.environ.get("OLLAMA_HOST")
        if ollama_host:
            self.ollama_host = ollama_host.rstrip("/")
        elif env_host:
            self.ollama_host = env_host.rstrip("/")
        elif self.api_key:
            self.ollama_host = "https://ollama.com"
        else:
            self.ollama_host = "http://localhost:11434"

        self.embedding_model = embedding_model
        self.alfa_hibrido = alfa_hibrido
        self.documentos: List[Dict[str, Any]] = []
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[Any] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.vectorizer_doc: Optional[TfidfVectorizer] = None
        self.tfidf_matrix_doc = None
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
        """Indexa los documentos clínicos extrayendo fragmentos enriquecidos, BM25 y representaciones vectoriales."""
        self.documentos = docs

        # 1. TF-IDF a nivel de documento completo para la guardia de cobertura
        textos_doc = [f"{d['titulo']} {d.get('atc', '')}\n{d['contenido']}" for d in docs]
        if SKLEARN_AVAILABLE and TfidfVectorizer is not None:
            self.vectorizer_doc = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=10000,
                strip_accents="unicode",
                stop_words=SPANISH_STOPWORDS
            )
            self.tfidf_matrix_doc = self.vectorizer_doc.fit_transform(textos_doc)

        # 2. Segmentación en fragmentos enriquecidos con metadatos
        self.chunks = crear_chunks_enriquecidos(docs)
        textos_chunks = [c["texto_indexable"] for c in self.chunks]

        # 3. Índice léxico BM25
        if BM25_AVAILABLE and BM25Okapi is not None:
            corpus_tokens = [tokenizar_clinico(t) for t in textos_chunks]
            self.bm25 = BM25Okapi(corpus_tokens)
        else:
            self.bm25 = None

        # 4. TF-IDF sobre fragmentos enriquecidos (contingencia y componente híbrido local)
        if SKLEARN_AVAILABLE and TfidfVectorizer is not None:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=10000,
                strip_accents="unicode"
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(textos_chunks)
        else:
            self.vectorizer = None
            self.tfidf_matrix = None
            self.doc_tokens = [set(tokenizar_clinico(t)) for t in textos_chunks]

        # 5. Embeddings densos con Ollama (con caché persistente en disco)
        if self.use_ollama:
            try:
                self.embeddings = self._obtener_o_generar_embeddings_cache(textos_chunks)
                self.modo_indice = "ollama"
            except Exception as exc:
                logger.warning("Fallo al generar embeddings con Ollama; se usa TF-IDF + BM25: %s", exc)
                self.use_ollama = False
                self.embeddings = None
                self.modo_indice = "tfidf" if self.vectorizer is not None else "tokens"
        else:
            self.modo_indice = "tfidf" if self.vectorizer is not None else "tokens"

    def _obtener_o_generar_embeddings_cache(self, textos: List[str]) -> np.ndarray:
        """Carga los embeddings desde disco o los genera con Ollama y persiste la caché."""
        import hashlib
        cache_dir = WORKSPACE_ROOT / "data" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_model = re.sub(r"[^\w\-]", "_", self.embedding_model)
        cache_path = cache_dir / f"embeddings_cache_{safe_model}.npz"

        corpus_hash = hashlib.md5("".join(textos).encode("utf-8")).hexdigest()

        if cache_path.exists():
            try:
                data = np.load(cache_path, allow_pickle=True)
                if (
                    str(data.get("corpus_hash", "")) == corpus_hash
                    and int(data.get("num_chunks", 0)) == len(textos)
                    and "embeddings" in data
                ):
                    logger.info("Caché de embeddings cargada desde %s (%d vectores)", cache_path.name, len(textos))
                    return np.array(data["embeddings"])
            except Exception as exc:
                logger.warning("Error leyendo caché de embeddings (%s); se recalcula", exc)

        # Prefijo asimétrico para documentos según directrices de embeddinggemma/Qwen
        textos_con_prefijo = [f"title: none | text: {t}" for t in textos]
        embs = self._embeber_textos(textos_con_prefijo)
        embs_arr = np.array(embs)

        try:
            np.savez_compressed(
                cache_path,
                embeddings=embs_arr,
                corpus_hash=corpus_hash,
                num_chunks=len(textos),
                model=self.embedding_model
            )
            logger.info("Caché de embeddings guardada en %s", cache_path.name)
        except Exception as exc:
            logger.warning("No se pudo persistir la caché de embeddings: %s", exc)

        return embs_arr

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

    def buscar(
        self,
        query: str,
        top_k: int = 4,
        filtro_tipo: Optional[str] = None,
        alfa: Optional[float] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Recuperación híbrida optimizada:
        1. Pondera señal densa (Ollama) o semántica (TF-IDF) recortada a [0, 1].
        2. Pondera señal léxica BM25 normalizada por el máximo de la consulta.
        3. Agrega chunks de vuelta a nivel de documento padre preservando trazabilidad.
        """
        if not self.chunks:
            return []

        alfa = self.alfa_hibrido if alfa is None else alfa
        candidatos = list(range(len(self.chunks)))
        if filtro_tipo:
            candidatos = [i for i, c in enumerate(self.chunks) if c.get("tipo") == filtro_tipo]
            if not candidatos:
                candidatos = list(range(len(self.chunks)))

        # 1. Señal Léxica BM25
        toks_query = tokenizar_clinico(query)
        if self.bm25 is not None and toks_query:
            raw_bm25 = np.array(self.bm25.get_scores(toks_query))[candidatos]
            max_b = raw_bm25.max()
            s_bm25 = (raw_bm25 / max_b) if max_b > 0 else np.zeros_like(raw_bm25)
        else:
            s_bm25 = np.zeros(len(candidatos), dtype=float)

        # 2. Señal Semántica (Ollama con prefijo asimétrico o TF-IDF)
        s_semantica = None
        if self.modo_indice == "ollama" and self.use_ollama and self.embeddings is not None and cosine_similarity is not None:
            try:
                # Prefijo asimétrico para consultas: task: search result | query: {query}
                query_con_prefijo = f"task: search result | query: {query}"
                res = requests.post(
                    f"{self.ollama_host}/api/embeddings",
                    headers=self._get_headers(),
                    json={"model": self.embedding_model, "prompt": query_con_prefijo},
                    timeout=5
                )
                if res.status_code == 200:
                    q_emb = np.array(res.json()["embedding"]).reshape(1, -1)
                    sims = cosine_similarity(q_emb, self.embeddings[candidatos])[0]
                    s_semantica = np.clip(sims, 0.0, 1.0)
            except Exception:
                s_semantica = None

        if s_semantica is None:
            # Fallback a TF-IDF sobre fragmentos
            if self.vectorizer is not None and self.tfidf_matrix is not None and cosine_similarity is not None:
                q_vec = self.vectorizer.transform([query])
                sims = cosine_similarity(q_vec, self.tfidf_matrix[candidatos])[0]
                s_semantica = np.clip(sims, 0.0, 1.0)
            else:
                # Fallback de solapamiento de tokens
                q_tokens = set(tokenizar_clinico(query))
                sims = []
                for i in candidatos:
                    doc_t = getattr(self, "doc_tokens", [])[i] if hasattr(self, "doc_tokens") and i < len(self.doc_tokens) else set()
                    inter = len(q_tokens & doc_t)
                    union = len(q_tokens | doc_t)
                    score = (inter / union) if union > 0 else 0.0
                    sims.append(score)
                s_semantica = np.array(sims, dtype=float)

        # 3. Fusión Híbrida Ponderada
        tiene_bm25 = bool(s_bm25.max() > 0)
        tiene_sem = bool(s_semantica.max() > 0)

        if tiene_bm25 and tiene_sem:
            puntaje_total = alfa * s_semantica + (1.0 - alfa) * s_bm25
        elif tiene_sem:
            puntaje_total = s_semantica
        else:
            puntaje_total = s_bm25

        top_chunk_rel = np.argsort(puntaje_total)[::-1][:top_k * 4]
        chunks_evaluados = [(self.chunks[candidatos[i]], float(puntaje_total[i])) for i in top_chunk_rel]

        # 4. Agregación a nivel de documento padre
        return agregar_chunks_a_documentos(chunks_evaluados, top_k=top_k)


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
            self.ollama_host = ollama_host.rstrip("/")
        elif env_host:
            self.ollama_host = env_host.rstrip("/")
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
            sec_destacada = (
                f"Sección Relevante Destacada: {doc['seccion_relevante']}\n"
                if doc.get("seccion_relevante") and doc.get("seccion_relevante") != "Documento Completo"
                else ""
            )
            bloques.append(
                f"--- DOCUMENTO CLÍNICO [{doc_id}] ---\n"
                f"ID de Citación: [{doc_id}]\n"
                f"Título: {doc['titulo']}\n"
                f"Código ATC: {doc.get('atc', 'N/D')}\n"
                f"Tipo / Categoría: {doc.get('tipo', 'General')}\n"
                f"{sec_destacada}"
                f"Score de Relevancia: {score:.4f}\n"
                f"Contenido:\n{doc['contenido']}\n"
            )
        return "\n\n".join(bloques)

    def consultar(self, pregunta: str, top_k: int = 3) -> Dict[str, Any]:
        """Ejecuta el pipeline completo de recuperación, ensamblado de contexto y generación."""
        t0_ret = time.time()
        docs_recuperados = self.vector_store.buscar(pregunta, top_k=top_k)
        lat_recuperacion = time.time() - t0_ret

        # Guardia de cobertura: con recuperación léxica (TF-IDF/BM25)
        # Evaluamos la similitud documental para garantizar que el corpus contiene
        # vocabulario clínico relevante para la consulta antes de proceder.
        if self.vector_store.modo_indice != "ollama":
            top_score_cobertura = 0.0
            if (
                self.vector_store.vectorizer_doc is not None
                and self.vector_store.tfidf_matrix_doc is not None
                and cosine_similarity is not None
            ):
                q_vec_doc = self.vector_store.vectorizer_doc.transform([pregunta])
                sims_doc = cosine_similarity(q_vec_doc, self.vector_store.tfidf_matrix_doc)[0]
                top_score_cobertura = float(sims_doc.max()) if sims_doc.size > 0 else 0.0
            elif docs_recuperados:
                top_score_cobertura = docs_recuperados[0][1]

            if not docs_recuperados or top_score_cobertura < MIN_SCORE_TFIDF:
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
            # Si /api/generate devuelve 404 (típico en Ollama Cloud o proxies), intentar /api/chat
            if res.status_code == 404:
                chat_payload = {
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": SOFIA_CLINICAL_PROMPT},
                        {"role": "user", "content": prompt_usuario}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "num_predict": NUM_PREDICT
                    }
                }
                res_chat = requests.post(
                    f"{self.ollama_host}/api/chat",
                    headers=self.vector_store._get_headers(),
                    json=chat_payload,
                    timeout=75
                )
                if res_chat.status_code == 200:
                    res = res_chat

            if res.status_code == 200:
                datos = res.json()
                if "message" in datos:
                    raw_resp = datos.get("message", {}).get("content", "").strip()
                else:
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
                detalle = res.text.strip() if hasattr(res, "text") and res.text else "sin detalle"
                logger.warning(
                    "Ollama (%s) devolvió status %s (%s); se usa contingencia",
                    self.ollama_host, res.status_code, detalle[:200]
                )
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
