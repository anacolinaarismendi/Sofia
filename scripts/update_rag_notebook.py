#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_rag_notebook.py
Actualiza notebooks/04_rag_psicofarmacologia.ipynb para:
1. Indexar el corpus de pre_training/ (documentos Markdown con YAML frontmatter y Voice Snippets).
2. Incorporar el nuevo 'Soul' (Prompt Template optimizado con IA).
3. Asegurar opciones de inferencia calibradas (temperature: 0.2, num_predict: 500).
4. Exportar la base de conocimiento completa a data/processed/rag_knowledge_base.json.
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/04_rag_psicofarmacologia.ipynb")

def update_notebook():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Actualizar Celda 1 (Rutas)
    nb["cells"][1]["source"] = [
        "import os\n",
        "import sys\n",
        "import json\n",
        "import re\n",
        "from pathlib import Path\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import requests\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.metrics.pairwise import cosine_similarity\n",
        "\n",
        "# Detección dinámica y robusta de rutas\n",
        "current_dir = Path.cwd()\n",
        "project_root = current_dir if (current_dir / 'data').exists() else current_dir.parent\n",
        "if str(project_root) not in sys.path:\n",
        "    sys.path.insert(0, str(project_root))\n",
        "\n",
        "DATA_PROCESSED = project_root / 'data' / 'processed'\n",
        "DIR_PRETRAINING = project_root / 'pre_training'\n",
        "print(f'Entorno RAG configurado.\\n- Raíz: {project_root}\\n- Datos Procesados: {DATA_PROCESSED}\\n- Corpus pre_training: {DIR_PRETRAINING}')\n"
    ]

    # 2. Actualizar Celda 2 (Markdown)
    nb["cells"][2]["source"] = [
        "## 1. Carga de Datasets Procesados y del Corpus Clínico `pre_training/`\n",
        "\n",
        "Para maximizar la precisión diagnóstica y terapéutica, el motor RAG ahora ingesta dos niveles de información:\n",
        "1. **Datasets Tabulares Procesados:** CSVs limpios de Jufe, Stahl y CANMAT para consulta relacional.\n",
        "2. **Corpus Clínico Estructurado `pre_training/`:** 26 monografías narrativas, guías CANMAT, interacciones CYP450 y protocolos de desprescripción con metadatos YAML y resúmenes verbalizables (*Voice Snippets*).\n"
    ]

    # 3. Actualizar Celda 3 (Carga y Parser de pre_training)
    nb["cells"][3]["source"] = [
        "# 1. Cargar Datasets Tabulares de Soporte\n",
        "df_jufe = pd.read_csv(DATA_PROCESSED / 'jufe_farmacos_clean.csv')\n",
        "df_jufe_efectos = pd.read_csv(DATA_PROCESSED / 'jufe_efectos_adversos_clean.csv')\n",
        "df_stahl = pd.read_csv(DATA_PROCESSED / 'stahl_farmacos_clean.csv')\n",
        "df_stahl_eventos = pd.read_csv(DATA_PROCESSED / 'stahl_eventos_adversos_clean.csv')\n",
        "df_canmat = pd.read_csv(DATA_PROCESSED / 'canmat_recomendaciones_clean.csv')\n",
        "\n",
        "# 2. Función Lectora del Corpus pre_training/\n",
        "def cargar_corpus_pretraining(directorio: Path):\n",
        "    docs = []\n",
        "    if not directorio.exists():\n",
        "        print(f'Aviso: Directorio {directorio} no encontrado.')\n",
        "        return docs\n",
        "        \n",
        "    for f in sorted(directorio.glob('**/*.md')):\n",
        "        if f.name.startswith('00_'):\n",
        "            continue  # Omitir el manifiesto técnico\n",
        "        with open(f, 'r', encoding='utf-8') as arch:\n",
        "            texto = arch.read()\n",
        "        \n",
        "        parts = texto.split('---', 2)\n",
        "        meta = {}\n",
        "        cuerpo = texto\n",
        "        if len(parts) >= 3:\n",
        "            for line in parts[1].strip().splitlines():\n",
        "                if ':' in line:\n",
        "                    k, v = line.split(':', 1)\n",
        "                    meta[k.strip()] = v.strip().strip('\"').strip(\"'\")\n",
        "            cuerpo = parts[2].strip()\n",
        "            \n",
        "        voice_snippet = ''\n",
        "        match_voice = re.search(r'## 1\\. Resumen Ejecutivo Verbalizable.*?\\n>(.*?)(?=\\n##|\\Z)', cuerpo, re.DOTALL)\n",
        "        if match_voice:\n",
        "            voice_snippet = match_voice.group(1).replace('\\n>', ' ').replace('\"', '').strip()\n",
        "            \n",
        "        docs.append({\n",
        "            'id': meta.get('id', f.stem),\n",
        "            'tipo': meta.get('tipo', f.parent.name),\n",
        "            'titulo': meta.get('titulo', meta.get('farmaco', f.stem.replace('_', ' ').title())),\n",
        "            'atc': meta.get('atc_codigo', 'N/D'),\n",
        "            'voice_snippet': voice_snippet,\n",
        "            'contenido': cuerpo,\n",
        "            'archivo_origen': str(f.relative_to(project_root))\n",
        "        })\n",
        "    return docs\n",
        "\n",
        "docs_pretraining = cargar_corpus_pretraining(DIR_PRETRAINING)\n",
        "print(f'Datos cargados exitosamente:')\n",
        "print(f'- Tablas CSV: Jufe ({len(df_jufe)} fármacos) | Stahl ({len(df_stahl)} fármacos) | CANMAT ({len(df_canmat)} directrices)')\n",
        "print(f'- Corpus pre_training: {len(docs_pretraining)} documentos Markdown ricos parseados.')\n"
    ]

    # 4. Actualizar Celda 6 (Markdown de Construcción de Chunks)
    nb["cells"][6]["source"] = [
        "## 3. Construcción y Fusión de la Base de Conocimiento RAG\n",
        "\n",
        "Integramos los 26 documentos de alta densidad semántica provenientes de `pre_training/` junto con las directrices específicas de CANMAT y el catálogo de marcas comerciales.\n"
    ]

    # 5. Actualizar Celda 7 (Integración de documentos_rag)
    nb["cells"][7]["source"] = [
        "documentos_rag = []\n",
        "\n",
        "# 1. Incorporar los documentos curados del corpus pre_training/ (Monografías, CYP450, Cascadas, CANMAT, Alertas)\n",
        "documentos_rag.extend(docs_pretraining)\n",
        "\n",
        "# 2. Añadir fichas directas de guías CANMAT (por si se requiere granularidad por fármaco individual)\n",
        "for _, row_c in df_canmat.iterrows():\n",
        "    doc_id = f\"canmat_{row_c['patologia'][:3].lower()}_{row_c['farmaco'].split()[0].lower()}\"\n",
        "    # Evitar duplicados si ya existe un documento más completo\n",
        "    if not any(d['id'] == doc_id for d in documentos_rag):\n",
        "        contenido_guia = f\"\"\"# GUÍA CLÍNICA CANMAT: {row_c['patologia'].upper()} ({row_c['fase'].upper()})\n",
        "- Recomendación: {row_c['linea']}\n",
        "- Estrategia: {row_c['estrategia']}\n",
        "- Fármaco / Esquema: {row_c['farmaco']} (ATC: {row_c['atc']})\n",
        "- Nivel de Evidencia Científica: {row_c['nivel_evidencia']}\n",
        "- Directriz Clínica: {row_c['comentarios']}\n",
        "- Fuente: Canadian Network for Mood and Anxiety Treatments (CANMAT)\n",
        "\"\"\"\n",
        "        documentos_rag.append({\n",
        "            'id': doc_id,\n",
        "            'tipo': 'guia_canmat',\n",
        "            'titulo': f\"CANMAT: {row_c['patologia']} - {row_c['farmaco']}\",\n",
        "            'atc': row_c['atc'],\n",
        "            'voice_snippet': f\"CANMAT recomienda {row_c['farmaco']} en {row_c['linea']} para {row_c['patologia']} ({row_c['fase']}) con evidencia {row_c['nivel_evidencia']}.\",\n",
        "            'contenido': contenido_guia.strip(),\n",
        "            'archivo_origen': 'data/processed/canmat_recomendaciones_clean.csv'\n",
        "        })\n",
        "\n",
        "print(f'Total de documentos clínicos indexables en el RAG: {len(documentos_rag)}')\n"
    ]

    # 6. Actualizar Celda 9 (Vector Store)
    nb["cells"][9]["source"] = [
        "class ClinicalVectorStore:\n",
        "    def __init__(self, ollama_host='http://localhost:11434', embedding_model='embeddinggemma:latest'):\n",
        "        self.ollama_host = ollama_host\n",
        "        self.embedding_model = embedding_model\n",
        "        self.documentos = []\n",
        "        self.embeddings = None\n",
        "        self.use_ollama = False\n",
        "        self._verificar_ollama()\n",
        "        \n",
        "    def _verificar_ollama(self):\n",
        "        try:\n",
        "            r = requests.get(f'{self.ollama_host}/api/tags', timeout=2)\n",
        "            if r.status_code == 200:\n",
        "                self.use_ollama = True\n",
        "                print(f'✓ Conectado a Ollama local ({self.ollama_host}).')\n",
        "            else:\n",
        "                self.use_ollama = False\n",
        "        except Exception:\n",
        "            self.use_ollama = False\n",
        "            print('ℹ Ollama no está activo. Se usará el vectorizador TF-IDF local de Scikit-Learn.')\n",
        "\n",
        "    def indexar_documentos(self, docs):\n",
        "        self.documentos = docs\n",
        "        textos = [f\"{d['titulo']} {d.get('atc', '')}\\n{d['contenido']}\" for d in docs]\n",
        "        \n",
        "        if self.use_ollama:\n",
        "            try:\n",
        "                print(f'Generando embeddings con Ollama ({self.embedding_model}) para {len(docs)} documentos...')\n",
        "                emb_list = []\n",
        "                for t in textos:\n",
        "                    res = requests.post(\n",
        "                        f'{self.ollama_host}/api/embeddings',\n",
        "                        json={'model': self.embedding_model, 'prompt': t[:2000]},\n",
        "                        timeout=15\n",
        "                    )\n",
        "                    emb_list.append(res.json()['embedding'])\n",
        "                self.embeddings = np.array(emb_list)\n",
        "                print(f'✓ Embeddings calculados exitosamente con Ollama ({self.embedding_model}).')\n",
        "                return\n",
        "            except Exception as e:\n",
        "                print(f'Error generando embeddings en Ollama: {e}. Conmutando a fallback local.')\n",
        "                self.use_ollama = False\n",
        "                \n",
        "        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500)\n",
        "        self.embeddings = self.vectorizer.fit_transform(textos).toarray()\n",
        "        print(f'✓ Indexados {len(self.documentos)} documentos con vectorizador semántico local.')\n",
        "\n",
        "    def buscar(self, query, top_k=2):\n",
        "        if self.use_ollama:\n",
        "            try:\n",
        "                res = requests.post(\n",
        "                    f'{self.ollama_host}/api/embeddings',\n",
        "                    json={'model': self.embedding_model, 'prompt': query},\n",
        "                    timeout=5\n",
        "                )\n",
        "                query_vec = np.array(res.json()['embedding']).reshape(1, -1)\n",
        "                sims = cosine_similarity(query_vec, self.embeddings)[0]\n",
        "            except Exception:\n",
        "                query_vec = self.vectorizer.transform([query]).toarray()\n",
        "                sims = cosine_similarity(query_vec, self.embeddings)[0]\n",
        "        else:\n",
        "            query_vec = self.vectorizer.transform([query]).toarray()\n",
        "            sims = cosine_similarity(query_vec, self.embeddings)[0]\n",
        "            \n",
        "        indices_top = np.argsort(sims)[::-1][:top_k]\n",
        "        \n",
        "        resultados = []\n",
        "        for idx in indices_top:\n",
        "            doc = self.documentos[idx].copy()\n",
        "            doc['score_similitud'] = round(float(sims[idx]), 4)\n",
        "            resultados.append(doc)\n",
        "            \n",
        "        return resultados\n",
        "\n",
        "# Inicializar e indexar la base unificada\n",
        "vector_store = ClinicalVectorStore()\n",
        "vector_store.indexar_documentos(documentos_rag)\n"
    ]

    # 7. Actualizar Celda 10 (Markdown del Soul)
    nb["cells"][10]["source"] = [
        "## 5. Módulo de Generación Asistida con el Nuevo 'Soul' de Sofia (RAG Prompting & Ollama)\n",
        "\n",
        "El 'Soul' de Sofia ha sido optimizado con técnicas de IA para incorporar:\n",
        "- **Identidad Profesional:** Médica especialista en psicofarmacología clínica y farmacovigilancia.\n",
        "- **Grounding Estricto y Citaciones:** Obligación de citar a Jufe, Stahl o CANMAT sin alucinaciones.\n",
        "- **Alerta Activa de Farmacovigilancia:** Detección de cascadas de prescripción e interacciones enzimáticas CYP450.\n",
        "- **Formato Médico Estructurado en 4 Secciones:** Juicio Clínico Inmediato, Mecanismo/Cinética, Pauta Operativa y Alertas.\n"
    ]

    # 8. Actualizar Celda 11 (Soul Prompt y Generador)
    nb["cells"][11]["source"] = [
        "PROMPT_SISTEMA_CLINICO = \"\"\"Eres Sofia, médica especialista en Psicofarmacología Clínica, Psiquiatría Basada en la Evidencia y Farmacovigilancia.\n",
        "Tu función es asesorar a médicos adjuntos, residentes y profesionales de la salud con el máximo rigor científico, basándote EXCLUSIVAMENTE en el CONTEXTO CLÍNICO RECUPERADO.\n",
        "\n",
        "### DIRECTRICES DE RAZONAMIENTO Y SEGURIDAD CLÍNICA:\n",
        "1. GROUNDING ESTRICTO: Basa todas tus afirmaciones en los documentos del contexto recuperado (Dra. Gabriela Jufe, Dr. Stephen Stahl, Guías oficiales CANMAT/ISBD y catálogo de marcas oficiales). Si el contexto no contiene datos suficientes para responder con seguridad, declara explícitamente la limitación sin inventar datos.\n",
        "2. CITACIÓN DE FUENTES: Identifica de qué fuente proviene cada recomendación clínica (ej: 'Según la Dra. Gabriela Jufe...', 'El Dr. Stephen Stahl describe...', 'De acuerdo con las guías CANMAT (Evidencia Level 1)...').\n",
        "3. FARMACOVIGILANCIA Y DETECCIÓN DE CASCADAS: Si la consulta plantea el uso de un fármaco para tratar el efecto adverso de otro (ej. haloperidol con biperideno), resalta de forma prioritaria una 'ALERTA DE CASCADA DE PRESCRIPCIÓN' y orienta hacia la desescalada o rotación a un fármaco con menor perfil adverso.\n",
        "4. METABOLISMO ENZIMÁTICO Y CITOCROMOS: Especifica siempre las vías metabólicas y los citocromos involucrados (CYP1A2, CYP2D6, CYP3A4, CYP2B6, glucuronidación UGT1A4 o eliminación renal de litio).\n",
        "\n",
        "### FORMATO DE RESPUESTA MÉDICA OBLIGATORIO:\n",
        "Estructura tu respuesta en los siguientes 4 apartados numerados:\n",
        "- **1. Juicio Clínico Inmediato:** Conclusión directa, asertiva y aplicable a la consulta en 2-3 líneas.\n",
        "- **2. Mecanismo Farmacológico y Cinética:** Explicación fisiopatológica, receptores involucrados (D2, 5-HT, etc.) y/o perfil enzimático CYP450.\n",
        "- **3. Pauta Operativa y Recomendaciones:** Dosis sugeridas, velocidad de titulación/desescalada y parámetros obligatorios de monitorización analítica o clínica.\n",
        "- **4. Alertas de Seguridad y Trazabilidad Comercial:** Advertencias críticas (caja negra, litemia, agranulocitosis) y nombres comerciales / laboratorios aplicables.\n",
        "\"\"\"\n",
        "\n",
        "def generar_respuesta_rag(pregunta, vector_store, modelo_ollama=\"qwen3.5:2b-mlx\", top_k=2):\n",
        "    docs_recuperados = vector_store.buscar(pregunta, top_k=top_k)\n",
        "    \n",
        "    contexto_str = \"\\n\\n---\\n\\n\".join([\n",
        "        f\"### [{d['titulo']}] (Tipo: {d.get('tipo', 'clinico')} | ATC: {d.get('atc', 'N/D')})\\n{d['contenido']}\"\n",
        "        for d in docs_recuperados\n",
        "    ])\n",
        "    \n",
        "    prompt_completo = f\"\"\"{PROMPT_SISTEMA_CLINICO}\n",
        "\n",
        "==================================================\n",
        "CONTEXTO CLÍNICO RECUPERADO:\n",
        "{contexto_str}\n",
        "==================================================\n",
        "PREGUNTA DEL MÉDICO / USUARIO:\n",
        "{pregunta}\n",
        "==================================================\n",
        "RESPUESTA CLÍNICA DE SOFIA:\"\"\"\n",
        "\n",
        "    if vector_store.use_ollama:\n",
        "        try:\n",
        "            res = requests.post(\n",
        "                f\"{vector_store.ollama_host}/api/generate\",\n",
        "                json={\n",
        "                    \"model\": modelo_ollama,\n",
        "                    \"prompt\": prompt_completo,\n",
        "                    \"stream\": False,\n",
        "                    \"options\": {\n",
        "                        \"temperature\": 0.2,\n",
        "                        \"top_p\": 0.85,\n",
        "                        \"num_predict\": 500\n",
        "                    }\n",
        "                },\n",
        "                timeout=120\n",
        "            )\n",
        "            return res.json()[\"response\"], docs_recuperados\n",
        "        except Exception as e:\n",
        "            print(f\"Aviso: Error con Ollama ({e}). Mostrando síntesis clínica local.\")\n",
        "\n",
        "    # Fallback determinista local\n",
        "    sintesis = f\"**1. Juicio Clínico Inmediato:** Para la consulta sobre '{pregunta}', se recuperaron documentos clave:\\n\"\n",
        "    for d in docs_recuperados:\n",
        "        sintesis += f\"- {d['titulo']} (Relevancia: {d['score_similitud']})\\n\"\n",
        "        if d.get('voice_snippet'):\n",
        "            sintesis += f\"  > Resumen Verbal: \\\"{d['voice_snippet'][:180]}...\\\"\\n\"\n",
        "    return sintesis, docs_recuperados\n"
    ]

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"✓ Notebook {NOTEBOOK_PATH} actualizado exitosamente.")

if __name__ == "__main__":
    update_notebook()
