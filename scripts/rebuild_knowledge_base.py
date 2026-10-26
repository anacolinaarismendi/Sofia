#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rebuild_knowledge_base.py
Regenera data/processed/rag_knowledge_base.json integrando:
1. Los 27 documentos Markdown actualizados de pre_training/.
2. Las 22 fichas estructuradas de recomendaciones CANMAT (CSV).
3. Las 9 fichas de marcas comerciales y laboratorios.

Garantiza que la base de conocimiento no quede obsoleta (kb_obsoleta == False).
"""

import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.rag.engine import cargar_corpus_pretraining, ClinicalRAGEngine


def rebuild():
    kb_path = WORKSPACE_ROOT / "data" / "processed" / "rag_knowledge_base.json"
    print(f"Leyendo base de conocimiento existente en: {kb_path}")

    existing_docs = []
    if kb_path.exists():
        with open(kb_path, "r", encoding="utf-8") as f:
            existing_docs = json.load(f)

    # Conservar fichas generadas a partir de CSVs y catálogo de marcas
    csv_and_marcas = [
        d for d in existing_docs
        if d.get("tipo") in ("guia_canmat_csv", "catalogo_marcas")
    ]
    print(f"Fichas CSV y marcas conservadas: {len(csv_and_marcas)}")

    # Cargar documentos frescos desde pre_training/
    pre_docs = cargar_corpus_pretraining()
    print(f"Documentos pre_training cargados: {len(pre_docs)}")

    # Unir asegurando unicidad de IDs
    documentos_rag = []
    ids_vistos = set()

    for doc in pre_docs:
        if doc["id"] not in ids_vistos:
            ids_vistos.add(doc["id"])
            documentos_rag.append(doc)

    for doc in csv_and_marcas:
        if doc["id"] not in ids_vistos:
            ids_vistos.add(doc["id"])
            documentos_rag.append(doc)

    print(f"Total documentos consolidados: {len(documentos_rag)}")

    # Escribir con codificación utf-8 e indentación de 2 espacios
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(documentos_rag, f, indent=2, ensure_ascii=False)

    print(f"✓ Guardado exitoso en {kb_path}")

    # Verificar con ClinicalRAGEngine
    rag = ClinicalRAGEngine()
    if rag.kb_obsoleta:
        print("⚠️ Advertencia: kb_obsoleta sigue siendo True.")
    else:
        print("✓ Verificación: kb_obsoleta es False. Base de conocimiento plenamente sincronizada.")


if __name__ == "__main__":
    rebuild()
