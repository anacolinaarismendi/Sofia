#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_rag_with_ollama.py
Prueba end-to-end del motor RAG de Sofia:
1. Ingesta e indexación del corpus pre_training/
2. Generación de embeddings con Ollama (embeddinggemma:latest) o fallback local
3. Generación de respuesta con el nuevo Prompt Template ('Soul' de Sofia)
4. Prueba de los 4 casos clínicos benchmark
"""

import os
import sys
import json
import re
from pathlib import Path
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
PRE_TRAINING_DIR = BASE_DIR / "pre_training"

PROMPT_SISTEMA_CLINICO = """Eres Sofia, médica especialista en Psicofarmacología Clínica, Psiquiatría Basada en la Evidencia y Farmacovigilancia.
Tu función es asesorar a médicos adjuntos, residentes y profesionales de la salud con el máximo rigor científico, basándote EXCLUSIVAMENTE en el CONTEXTO CLÍNICO RECUPERADO.

### DIRECTRICES DE RAZONAMIENTO Y SEGURIDAD CLÍNICA:
1. GROUNDING ESTRICTO: Basa todas tus afirmaciones en los documentos del contexto recuperado (Dra. Gabriela Jufe, Dr. Stephen Stahl, Guías oficiales CANMAT/ISBD y catálogo de marcas oficiales). Si el contexto no contiene datos suficientes para responder con seguridad, declara explícitamente la limitación sin inventar datos.
2. CITACIÓN DE FUENTES: Identifica de qué fuente proviene cada recomendación clínica (ej: "Según la Dra. Gabriela Jufe...", "El Dr. Stephen Stahl describe...", "De acuerdo con las guías CANMAT (Evidencia Level 1)...").
3. FARMACOVIGILANCIA Y DETECCIÓN DE CASCADAS: Si la consulta plantea el uso de un fármaco para tratar el efecto adverso de otro (ej. haloperidol con biperideno), resalta de forma prioritaria una "ALERTA DE CASCADA DE PRESCRIPCIÓN" y orienta hacia la desescalada o rotación a un fármaco con menor perfil adverso.
4. METABOLISMO ENZIMÁTICO Y CITOCROMOS: Especifica siempre las vías metabólicas y los citocromos involucrados (CYP1A2, CYP2D6, CYP3A4, CYP2B6, glucuronidación UGT1A4 o eliminación renal de litio).

### FORMATO DE RESPUESTA MÉDICA OBLIGATORIO:
Estructura tu respuesta en los siguientes 4 apartados numerados:
- **1. Juicio Clínico Inmediato:** Conclusión directa, asertiva y aplicable a la consulta en 2-3 líneas.
- **2. Mecanismo Farmacológico y Cinética:** Explicación fisiopatológica, receptores involucrados (D2, 5-HT, etc.) y/o perfil enzimático CYP450.
- **3. Pauta Operativa y Recomendaciones:** Dosis sugeridas, velocidad de titulación/desescalada y parámetros obligatorios de monitorización analítica o clínica.
- **4. Alertas de Seguridad y Trazabilidad Comercial:** Advertencias críticas (caja negra, litemia, agranulocitosis) y nombres comerciales / laboratorios aplicables.
"""

def cargar_documentos_pretraining(directorio: Path):
    documentos = []
    if not directorio.exists():
        print(f"Error: {directorio} no existe.")
        return documentos
    
    for f in sorted(directorio.glob("**/*.md")):
        if f.name.startswith("00_"):
            continue
        with open(f, "r", encoding="utf-8") as archivo:
            texto = archivo.read()
        
        parts = texto.split("---", 2)
        meta = {}
        cuerpo = texto
        if len(parts) >= 3:
            raw_yaml = parts[1]
            cuerpo = parts[2].strip()
            for line in raw_yaml.strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"').strip("'")
        
        voice_snippet = ""
        match_voice = re.search(r"## 1\. Resumen Ejecutivo Verbalizable.*?\n>(.*?)(?=\n##|\Z)", cuerpo, re.DOTALL)
        if match_voice:
            voice_snippet = match_voice.group(1).replace("\n>", " ").replace('"', '').strip()
            
        doc_id = meta.get("id", f.stem)
        tipo = meta.get("tipo", f.parent.name)
        titulo = meta.get("titulo", meta.get("farmaco", f.stem.replace("_", " ").title()))
        atc = meta.get("atc_codigo", "N/D")
        
        documentos.append({
            "id": doc_id,
            "tipo": tipo,
            "titulo": titulo,
            "atc": atc,
            "voice_snippet": voice_snippet,
            "contenido": cuerpo,
            "archivo_origen": str(f.relative_to(BASE_DIR))
        })
    return documentos

class ClinicalVectorStore:
    def __init__(self, ollama_host="http://localhost:11434", embedding_model="embeddinggemma:latest"):
        self.ollama_host = ollama_host
        self.embedding_model = embedding_model
        self.documentos = []
        self.embeddings = None
        self.use_ollama = False
        self._verificar_ollama()
        
    def _verificar_ollama(self):
        try:
            r = requests.get(f"{self.ollama_host}/api/tags", timeout=2)
            if r.status_code == 200:
                self.use_ollama = True
                print(f"✓ Conectado a Ollama en {self.ollama_host}")
            else:
                self.use_ollama = False
        except Exception:
            self.use_ollama = False
            print("ℹ Ollama no está accesible. Se utilizará TF-IDF local con Scikit-Learn.")

    def indexar_documentos(self, docs):
        self.documentos = docs
        textos = [f"{d['titulo']} {d.get('atc', '')}\n{d['contenido']}" for d in docs]
        
        if self.use_ollama:
            try:
                print(f"Generando embeddings con Ollama ({self.embedding_model}) para {len(docs)} documentos...")
                emb_list = []
                for idx, t in enumerate(textos):
                    res = requests.post(
                        f"{self.ollama_host}/api/embeddings",
                        json={"model": self.embedding_model, "prompt": t[:2000]},
                        timeout=15
                    )
                    emb_list.append(res.json()["embedding"])
                self.embeddings = np.array(emb_list)
                print(f"✓ Embeddings calculados exitosamente con {self.embedding_model}.")
                return
            except Exception as e:
                print(f"Aviso: Error con Ollama ({e}). Conmutando a fallback Scikit-Learn.")
                self.use_ollama = False
                
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500)
        self.embeddings = self.vectorizer.fit_transform(textos).toarray()
        print(f"✓ Indexados {len(self.documentos)} documentos con TF-IDF semántico.")

    def buscar(self, query, top_k=2):
        if self.use_ollama:
            try:
                res = requests.post(
                    f"{self.ollama_host}/api/embeddings",
                    json={"model": self.embedding_model, "prompt": query},
                    timeout=5
                )
                query_vec = np.array(res.json()["embedding"]).reshape(1, -1)
                sims = cosine_similarity(query_vec, self.embeddings)[0]
            except Exception:
                query_vec = self.vectorizer.transform([query]).toarray()
                sims = cosine_similarity(query_vec, self.embeddings)[0]
        else:
            query_vec = self.vectorizer.transform([query]).toarray()
            sims = cosine_similarity(query_vec, self.embeddings)[0]
            
        indices_top = np.argsort(sims)[::-1][:top_k]
        
        resultados = []
        for idx in indices_top:
            doc = self.documentos[idx].copy()
            doc["score_similitud"] = round(float(sims[idx]), 4)
            resultados.append(doc)
            
        return resultados

def generar_respuesta_rag(pregunta, vector_store, modelo_ollama="qwen3.5:2b-mlx", top_k=2):
    docs_recuperados = vector_store.buscar(pregunta, top_k=top_k)
    
    contexto_str = "\n\n---\n\n".join([
        f"### [{d['titulo']}] (Tipo: {d.get('tipo', 'clinico')} | ATC: {d.get('atc', 'N/D')})\n{d['contenido']}"
        for d in docs_recuperados
    ])
    
    prompt_completo = f"""{PROMPT_SISTEMA_CLINICO}

==================================================
CONTEXTO CLÍNICO RECUPERADO:
{contexto_str}
==================================================
PREGUNTA DEL MÉDICO / USUARIO:
{pregunta}
==================================================
RESPUESTA CLÍNICA DE SOFIA:"""

    if vector_store.use_ollama:
        try:
            res = requests.post(
                f"{vector_store.ollama_host}/api/generate",
                json={
                    "model": modelo_ollama,
                    "prompt": prompt_completo,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.85
                    }
                },
                timeout=180
            )
            return res.json()["response"], docs_recuperados
        except Exception as e:
            print(f"Aviso: Error consultando Ollama para generación ({e}). Usando síntesis local.")

    # Síntesis local determinista
    sintesis = f"**1. Juicio Clínico Inmediato:** Para la consulta sobre '{pregunta}', se recuperaron documentos clave:\n"
    for d in docs_recuperados:
        sintesis += f"- {d['titulo']} (Relevancia: {d['score_similitud']})\n"
        if d.get("voice_snippet"):
            sintesis += f"  > Resumen Verbal: \"{d['voice_snippet'][:180]}...\"\n"
    return sintesis, docs_recuperados

def run_benchmarks():
    print("=" * 70)
    print("INICIANDO PRUEBA DEL MOTOR RAG SOFIA CON PRE_TRAINING Y OLLAMA")
    print("=" * 70)
    
    docs = cargar_documentos_pretraining(PRE_TRAINING_DIR)
    print(f"Total de documentos cargados desde pre_training/: {len(docs)}")
    
    vs = ClinicalVectorStore()
    vs.indexar_documentos(docs)
    
    casos = [
        ("Caso 1: Marcas y Posología", "¿Qué laboratorios fabrican la Sertralina, cuáles son sus nombres comerciales y posología según Jufe?"),
        ("Caso 2: Interacciones CYP2D6", "¿Qué interacciones y precauciones enzimáticas con CYP2D6 describe Stahl para Paroxetina y Bupropión?"),
        ("Caso 3: CANMAT Manía y Viraje", "¿Qué fármacos recomienda CANMAT en primera línea para manía aguda y por qué desaconseja los antidepresivos solos en depresión bipolar?"),
        ("Caso 4: Cascada de Prescripción", "Un paciente en tratamiento con Haloperidol comienza con rigidez y temblor y le indican medicación antiparkinsoniana como biperideno. ¿Qué dicen Jufe y Stahl sobre este riesgo?")
    ]
    
    for titulo, pregunta in casos:
        print("\n" + "=" * 70)
        print(f"EVALUANDO {titulo}")
        print(f"Pregunta: {pregunta}")
        print("-" * 70)
        
        respuesta, docs_rec = generar_respuesta_rag(pregunta, vs)
        print("DOCUMENTOS RECUPERADOS:")
        for d in docs_rec:
            print(f"  • [{d['id']}] {d['titulo']} (Score: {d['score_similitud']}) - Origen: {d['archivo_origen']}")
        print("\nRESPUESTA GENERADA POR SOFIA:")
        print(respuesta)
        print("=" * 70)

if __name__ == "__main__":
    run_benchmarks()
