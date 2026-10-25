"""
chunker.py
Módulo de segmentación estructural y enriquecimiento de fragmentos clínicos para el RAG de Sofia.
Principios de segmentación semántica y enriquecimiento de fragmentos clínicos:
- Segmentación respetando encabezados H2 (secciones clínicas temáticas).
- Enriquecimiento de fragmentos con metadatos de trazabilidad ([ID], Principio Activo, ATC, Sección).
- Tokenización clínica especializada con normalización de acentos para BM25.
- Preservación de la referencia al documento padre para agregación jerárquica.
"""

import re
import unicodedata
from typing import List, Dict, Any, Tuple


def tokenizar_clinico(texto: str) -> List[str]:
    """
    Tokenizador clínico para BM25 y análisis léxico:
    - Normalización de acentos (NFD) para que 'posologia' y 'Posología' coincidan exactamente.
    - Conversión a minúsculas.
    - Extracción de secuencias alfanuméricas (preserva términos como 'cyp2d6', '5ht2a', '100mg').
    """
    if not texto:
        return []
    texto_sin_acentos = "".join(
        c for c in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(c) != "Mn"
    )
    return re.findall(r"[a-z0-9]+", texto_sin_acentos)


def crear_chunks_enriquecidos(
    documentos: List[Dict[str, Any]],
    min_chars_split: int = 1200
) -> List[Dict[str, Any]]:
    """
    Transforma documentos clínicos extensos en fragmentos enriquecidos respetando
    la estructura de secciones (H2).
    
    Para documentos breves (< min_chars_split), preserva el documento intacto enriqueciendo
    su cabecera. Para documentos extensos con múltiples secciones (monografías, protocolos),
    genera un chunk por sección temática manteniendo la referencia al documento padre.
    """
    chunks: List[Dict[str, Any]] = []

    for doc in documentos:
        doc_id = doc.get("id", "DOC")
        titulo = doc.get("titulo", "Sin Título")
        atc = doc.get("atc", "N/D")
        tipo = doc.get("tipo", "general")
        contenido = doc.get("contenido", "").strip()

        # Separar por encabezados de segundo nivel (##)
        partes_h2 = re.split(r"\n(?=##\s+)", contenido)

        if len(partes_h2) <= 1 or len(contenido) < min_chars_split:
            # Documento atómico o breve (ej. guía CANMAT CSV o ficha de marcas)
            cabecera = f"[{doc_id}] {titulo} | Código ATC: {atc} | Tipo: {tipo}"
            texto_indexable = f"{cabecera}\n\n{contenido}"
            chunks.append({
                "chunk_id": doc_id,
                "doc_id": doc_id,
                "titulo": titulo,
                "seccion": "Documento Completo",
                "atc": atc,
                "tipo": tipo,
                "contenido": contenido,
                "texto_indexable": texto_indexable,
                "doc_padre": doc
            })
        else:
            # Documento estructurado: generar sub-chunks temáticos enriquecidos
            # Si la primera parte no comienza con ## es un preámbulo o título H1
            secciones = partes_h2
            if not partes_h2[0].strip().startswith("##"):
                secciones = partes_h2[1:]

            for idx, parte in enumerate(secciones, 1):
                parte_limpia = parte.strip()
                if not parte_limpia:
                    continue

                # Extraer título de la sección si existe
                match_h2 = re.match(r"^##\s+(.+)$", parte_limpia, re.MULTILINE)
                sec_titulo = match_h2.group(1).strip() if match_h2 else f"Sección {idx}"

                # Enriquecimiento del fragmento: inyectar contexto institucional y clínico
                cabecera = f"[{doc_id}] {titulo} | ATC: {atc} | Sección: {sec_titulo}"
                texto_indexable = f"{cabecera}\n\n{parte_limpia}"

                chunks.append({
                    "chunk_id": f"{doc_id}__sec_{idx}",
                    "doc_id": doc_id,
                    "titulo": titulo,
                    "seccion": sec_titulo,
                    "atc": atc,
                    "tipo": tipo,
                    "contenido": parte_limpia,
                    "texto_indexable": texto_indexable,
                    "doc_padre": doc
                })

    return chunks


def agregar_chunks_a_documentos(
    chunks_con_score: List[Tuple[Dict[str, Any], float]],
    top_k: int = 4
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Agrega los resultados de chunks al nivel de documento padre.
    Si múltiples fragmentos del mismo documento coinciden, se conserva el mejor puntaje
    y se anota la sección de máxima coincidencia en doc['seccion_relevante'].
    Retorna una lista de tuplas (documento_padre, score) deduplicada por doc_id.
    """
    docs_agregados: Dict[str, Tuple[Dict[str, Any], float, str]] = {}

    for chunk, score in chunks_con_score:
        doc_padre = chunk.get("doc_padre")
        if not doc_padre:
            continue
        doc_id = doc_padre.get("id", "")
        seccion = chunk.get("seccion", "")

        if doc_id not in docs_agregados:
            docs_agregados[doc_id] = (doc_padre, score, seccion)
        else:
            # Mantener el máximo score alcanzado
            doc_existente, mejor_score, sec_existente = docs_agregados[doc_id]
            if score > mejor_score:
                docs_agregados[doc_id] = (doc_existente, score, seccion)

    # Ordenar por score descendente
    ordenados = sorted(docs_agregados.values(), key=lambda x: x[1], reverse=True)

    resultado: List[Tuple[Dict[str, Any], float]] = []
    for doc, score, sec in ordenados[:top_k]:
        doc_copia = dict(doc)
        doc_copia["seccion_relevante"] = sec
        resultado.append((doc_copia, score))

    return resultado
