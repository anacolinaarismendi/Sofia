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
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Directorio base portable
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


SOFIA_CLINICAL_PROMPT = """Eres Sofia, médica especialista en Psicofarmacología Clínica, Psiquiatría Basada en la Evidencia y Farmacovigilancia.
Tu función es asesorar a médicos adjuntos, residentes y profesionales de la salud con el máximo rigor científico, basándote EXCLUSIVAMENTE en el CONTEXTO CLÍNICO RECUPERADO.

### DIRECTRICES DE RAZONAMIENTO Y SEGURIDAD CLÍNICA:
1. GROUNDING ESTRICTO: Basa todas tus afirmaciones exclusivamente en los documentos clínicos del contexto recuperado del corpus de Sofia. Si el contexto no contiene datos suficientes para responder con seguridad, declara explícitamente la limitación sin inventar datos.
2. TRAZABILIDAD Y CITACIÓN POR IDENTIFICADOR ([DOC-ID]): No atribuyas citas textuales directas a autores personales (NO uses «Según la Dra. Jufe...» ni «El Dr. Stahl describe...»). En su lugar, debes respaldar cada afirmación farmacológica, posológica o de seguridad citando entre corchetes el código identificador único del documento clínico correspondiente (ej: [MONO-SERT-001], [CYP-2D6-001], [CASC-HAL-BIP-001], [CANMAT-BIP-MANIA-001]).
3. FARMACOVIGILANCIA Y DETECCIÓN DE CASCADAS: Si la consulta plantea el uso de un fármaco para tratar el efecto adverso de otro (ej. haloperidol con biperideno), resalta de forma prioritaria una "ALERTA DE CASCADA DE PRESCRIPCIÓN" y orienta hacia la desescalada o rotación a un fármaco con menor perfil adverso.
4. METABOLISMO ENZIMÁTICO Y CITOCROMOS: Especifica siempre las vías metabólicas y los citocromos involucrados (CYP1A2, CYP2D6, CYP3A4, CYP2B6, glucuronidación UGT1A4 o eliminación renal de litio).

### FORMATO DE RESPUESTA MÉDICA OBLIGATORIO:
Estructura tu respuesta en los siguientes 4 apartados numerados:
- **1. Juicio Clínico Inmediato:** Conclusión directa, asertiva y aplicable a la consulta en 2-3 líneas.
- **2. Mecanismo Farmacológico y Cinética:** Explicación fisiopatológica, receptores involucrados (D2, 5-HT, etc.) y/o perfil enzimático CYP450.
- **3. Pauta Operativa y Recomendaciones:** Dosis sugeridas, velocidad de titulación/desescalada y parámetros obligatorios de monitorización analítica o clínica.
- **4. Alertas de Seguridad y Trazabilidad Comercial:** Advertencias críticas (caja negra, litemia, agranulocitosis) y nombres comerciales / laboratorios aplicables.
"""


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
    """Almacén vectorial clínico híbrido con Ollama y fallback local TF-IDF."""

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        embedding_model: str = "embeddinggemma:latest"
    ):
        self.ollama_host = ollama_host
        self.embedding_model = embedding_model
        self.documentos: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.use_ollama = False
        self._verificar_ollama()

    def _verificar_ollama(self) -> bool:
        """Comprueba disponibilidad del host de Ollama."""
        try:
            r = requests.get(f"{self.ollama_host}/api/tags", timeout=1.5)
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

        # Siempre inicializar TF-IDF como contingencia inmediata
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
        self.tfidf_matrix = self.vectorizer.fit_transform(textos)

        if self.use_ollama:
            try:
                emb_list = []
                for t in textos:
                    res = requests.post(
                        f"{self.ollama_host}/api/embeddings",
                        json={"model": self.embedding_model, "prompt": t[:2000]},
                        timeout=10
                    )
                    if res.status_code == 200:
                        emb_list.append(res.json()["embedding"])
                    else:
                        raise RuntimeError(f"Error status {res.status_code}")
                self.embeddings = np.array(emb_list)
            except Exception:
                self.use_ollama = False
                self.embeddings = None

    def buscar(self, query: str, top_k: int = 3, filtro_tipo: Optional[str] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Recupera los top_k documentos clínicos más relevantes para la consulta."""
        if not self.documentos:
            return []

        docs_candidatos = self.documentos
        indices_candidatos = list(range(len(docs_candidatos)))

        if filtro_tipo:
            indices_candidatos = [i for i, d in enumerate(docs_candidatos) if d.get("tipo") == filtro_tipo]
            if not indices_candidatos:
                indices_candidatos = list(range(len(docs_candidatos)))

        # Búsqueda densa con Ollama si está disponible
        if self.use_ollama and self.embeddings is not None:
            try:
                res = requests.post(
                    f"{self.ollama_host}/api/embeddings",
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

        # Fallback determinista local con TF-IDF
        q_vec = self.vectorizer.transform([query])
        sub_matrix = self.tfidf_matrix[indices_candidatos]
        sims = cosine_similarity(q_vec, sub_matrix)[0]
        top_idx_rel = np.argsort(sims)[::-1][:top_k]
        return [(self.documentos[indices_candidatos[i]], float(sims[i])) for i in top_idx_rel]


class ClinicalRAGEngine:
    """Motor integral de RAG clínico de Sofia con soporte multimodal y groundedness estricto."""

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        embedding_model: str = "embeddinggemma:latest",
        llm_model: str = "qwen3.5:2b-mlx"
    ):
        self.ollama_host = ollama_host
        self.llm_model = llm_model
        self.vector_store = ClinicalVectorStore(ollama_host=ollama_host, embedding_model=embedding_model)
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
        docs_recuperados = self.vector_store.buscar(pregunta, top_k=top_k)
        contexto = self.construir_contexto(docs_recuperados)

        # Seleccionar el mejor Voice Snippet de los documentos recuperados
        voice_snippet = ""
        for doc, _ in docs_recuperados:
            if doc.get("voice_snippet"):
                voice_snippet = doc["voice_snippet"]
                break

        prompt_usuario = (
            f"### CONTEXTO CLÍNICO RECUPERADO:\n{contexto}\n\n"
            f"### CONSULTA MÉDICA:\n{pregunta}\n\n"
            f"Genera tu respuesta clínica estructurada respetando estrictamente las 4 secciones numeradas "
            f"y citando los códigos identificadores de documento [DOC-ID] del contexto:"
        )

        respuesta_texto = ""
        generado_con_llm = False

        # Intentar llamada a Ollama si está disponible
        try:
            payload = {
                "model": self.llm_model,
                "system": SOFIA_CLINICAL_PROMPT,
                "prompt": prompt_usuario,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_predict": 500
                }
            }
            res = requests.post(f"{self.ollama_host}/api/generate", json=payload, timeout=60)
            if res.status_code == 200:
                respuesta_texto = res.json().get("response", "").strip()
                generado_con_llm = True
        except Exception:
            pass

        # Si Ollama no está disponible, estructurar respuesta determinista a partir del mejor documento
        if not respuesta_texto and docs_recuperados:
            top_doc = docs_recuperados[0][0]
            top_id = top_doc.get("id", "DOC-001")
            respuesta_texto = (
                f"**1. Juicio Clínico Inmediato:**\n"
                f"{voice_snippet or top_doc['titulo']} [{top_id}].\n\n"
                f"**2. Mecanismo Farmacológico y Cinética:**\n"
                f"Perfil farmacológico y enzimático validado en [{top_id}] (Código ATC: {top_doc.get('atc', 'N/D')}).\n\n"
                f"**3. Pauta Operativa y Recomendaciones:**\n"
                f"Seguir la dosificación, rango terapéutico y pauta de monitorización descritos en [{top_id}].\n\n"
                f"**4. Alertas de Seguridad y Trazabilidad Comercial:**\n"
                f"Advertencias críticas y precauciones farmacovigiladas documentadas en [{top_id}]."
            )

        return {
            "pregunta": pregunta,
            "respuesta": respuesta_texto,
            "voice_snippet": voice_snippet,
            "generado_con_llm": generado_con_llm,
            "documentos_recuperados": docs_recuperados
        }
