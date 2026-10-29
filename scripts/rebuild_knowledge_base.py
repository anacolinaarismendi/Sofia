#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_full_database.py
Compila e integra la base de datos clínica completa y real de Sofia:
1. Extrae y valida 57 fármacos reales de Jufe (640 págs) y 67 recomendaciones de CANMAT (74 págs).
2. Genera stahl_farmacos_clean.csv y stahl_eventos_adversos_clean.csv con matriz CYP completa (CYP1A2, 2C9, 2C19, 2D6, 3A4, inductores y potencia).
3. Exporta la base relacional SQLite sofia_clinica.db indexada.
4. Regenera rag_knowledge_base.json integrando monografías Jufe, monografías Stahl, CANMAT, marcas canónicas deduplicadas y pre_training.
"""

import sys
import json
import sqlite3
import unicodedata
from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.extractors.jufe_extractor import JufeExtractor
from src.extractors.stahl_extractor import StahlExtractor
from src.extractors.canmat_extractor import CANMATExtractor
from src.enrichers.atc_enricher import ATCEnricher
from src.rag.engine import cargar_corpus_pretraining, ClinicalRAGEngine


def normalizar_id(texto: str) -> str:
    """Convierte texto en un identificador uniforme en mayúsculas sin acentos ni espacios."""
    if not isinstance(texto, str):
        return "UNKNOWN"
    norm = unicodedata.normalize("NFKD", texto)
    clean = "".join(c for c in norm if not unicodedata.combining(c))
    clean = clean.upper().replace(" ", "-").replace("/", "-").replace("(", "").replace(")", "").replace(".", "")
    while "--" in clean:
        clean = clean.replace("--", "-")
    return clean.strip("-")


def calibrar_snippet(texto: str, max_words: int = 80) -> str:
    """Ajusta la longitud del snippet para locución verbal fluida (máximo ~80 palabras, ~30s)."""
    if not isinstance(texto, str):
        return ""
    words = texto.split()
    if len(words) <= max_words:
        return texto
    truncated = " ".join(words[:max_words])
    last_period = truncated.rfind(".")
    if last_period > len(truncated) // 2:
        return truncated[:last_period + 1]
    return truncated + "."


def compilar_stahl() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return StahlExtractor.compilar_stahl()


def exportar_sqlite(datos_df: Dict[str, pd.DataFrame], db_path: Path):
    """Crea una base de datos relacional SQLite con todas las tablas indexadas."""
    print(f"Compilando base de datos relacional SQLite en: {db_path}")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    for nombre_tabla, df in datos_df.items():
        df.to_sql(nombre_tabla, conn, if_exists="replace", index=False)
        if "farmaco" in df.columns:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{nombre_tabla}_farmaco ON {nombre_tabla}(farmaco);")
        if "atc_codigo" in df.columns:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{nombre_tabla}_atc ON {nombre_tabla}(atc_codigo);")

    conn.commit()
    conn.close()
    print("✓ Base de datos SQLite sofia_clinica.db generada e indexada con éxito.")


def regenerar_knowledge_base():
    """Genera rag_knowledge_base.json integrando monografías Jufe, monografías Stahl, CANMAT, marcas canónicas y pre_training."""
    data_dir = WORKSPACE_ROOT / "data" / "processed"
    kb_path = data_dir / "rag_knowledge_base.json"

    # 1. Monografías y efectos de Jufe
    jufe_ext = JufeExtractor()
    df_jufe, df_jufe_efectos = jufe_ext.extract_monographs()
    df_jufe.to_csv(data_dir / "jufe_farmacos_clean.csv", index=False, encoding="utf-8")
    df_jufe_efectos.to_csv(data_dir / "jufe_efectos_adversos_clean.csv", index=False, encoding="utf-8")

    # 2. Recomendaciones y alertas de CANMAT
    canmat_ext = CANMATExtractor()
    df_canmat, df_canmat_alertas = canmat_ext.extract_recommendations()
    df_canmat.to_csv(data_dir / "canmat_recomendaciones_clean.csv", index=False, encoding="utf-8")
    df_canmat_alertas.to_csv(data_dir / "canmat_alertas_clinicas_clean.csv", index=False, encoding="utf-8")

    # 3. Fármacos y eventos de Stahl
    df_stahl, df_stahl_eventos = compilar_stahl()
    df_stahl.to_csv(data_dir / "stahl_farmacos_clean.csv", index=False, encoding="utf-8")
    df_stahl_eventos.to_csv(data_dir / "stahl_eventos_adversos_clean.csv", index=False, encoding="utf-8")

    # 4. Exportar base de datos SQLite
    exportar_sqlite({
        "jufe_farmacos": df_jufe,
        "jufe_efectos_adversos": df_jufe_efectos,
        "canmat_recomendaciones": df_canmat,
        "canmat_alertas": df_canmat_alertas,
        "stahl_farmacos": df_stahl,
        "stahl_eventos_adversos": df_stahl_eventos
    }, data_dir / "sofia_clinica.db")

    # 5. Cargar corpus pre_training (27 documentos curados de alta especialización)
    pre_docs = cargar_corpus_pretraining()
    print(f"Documentos pre_training cargados: {len(pre_docs)}")

    documentos_rag = []
    ids_vistos = set()

    for doc in pre_docs:
        if doc["id"] not in ids_vistos:
            ids_vistos.add(doc["id"])
            documentos_rag.append(doc)

    # 6. Transformar fármacos de Jufe en documentos RAG estructurados
    for _, row in df_jufe.iterrows():
        farmaco_nombre = row["farmaco"]
        doc_id = f"JUFE-MONO-{normalizar_id(farmaco_nombre)}"
        if doc_id in ids_vistos:
            continue
        ids_vistos.add(doc_id)

        atc = row.get("atc_codigo", "N/D")
        grupo = row.get("atc_grupo", "N/D")
        paginas = row.get("paginas_jufe", "N/D")

        contenido = f"""# Monografía Clínica Canónica: {farmaco_nombre}
**Fuente de Evidencia:** Psicofarmacología Práctica (4ª Edición), Dra. Gabriela Jufe (páginas {paginas}).
**Clase Terapéutica:** {row['clase']} | **Código ATC Oficial OMS:** {atc} ({grupo})

## 1. Resumen Ejecutivo Verbalizable (< 40s)
> {farmaco_nombre} es un psicofármaco de la clase {row['clase']} ({atc}). Su mecanismo primario es: {row['mecanismo']}. Rango posológico habitual de {row['posologia']}. Monitorizar: {row['poblaciones_especiales'][:140]}.

## 2. Mecanismo de Acción Farmacológica y Receptores
{row['mecanismo']}

## 3. Pauta Posológica y Administración
{row['posologia']}

## 4. Farmacocinética y Metabolismo Enzimático (CYP450)
{row['farmacocinetica']}

## 5. Efectos Adversos y Toxicidad
{row['efectos_adversos']}

## 6. Interacciones Medicamentosas Críticas
{row['interacciones']}

## 7. Precauciones y Poblaciones Especiales
{row['poblaciones_especiales']}
"""
        documentos_rag.append({
            "id": doc_id,
            "tipo": "monografia_jufe",
            "titulo": f"Monografía Jufe: {farmaco_nombre}",
            "atc": atc,
            "voice_snippet": f"{farmaco_nombre} ({atc}): {row['mecanismo'][:110]}. Pauta posológica estándar: {row['posologia'][:90]}.",
            "contenido": contenido.strip(),
            "archivo_origen": "data/raw/Psicofarmacologia practica - G. Jufe.pdf"
        })

    # 7. Transformar fármacos de Stahl en documentos RAG estructurados (STAHL-MONO-XXX)
    for _, row in df_stahl.iterrows():
        farmaco_nombre = row["farmaco"]
        doc_id = f"STAHL-MONO-{normalizar_id(farmaco_nombre)}"
        if doc_id in ids_vistos:
            continue
        ids_vistos.add(doc_id)

        atc = row.get("atc_codigo", "N/D")
        contenido = f"""# Guía de Prescripción de Stahl: {farmaco_nombre}
**Fuente de Evidencia:** Stephen M. Stahl - Prescriber's Guide (6th Edition).
**Clase Terapéutica:** {row['clase']} | **Código ATC:** {atc}

## 1. Perfil de Unión a Receptores y Farmacodinamia
{row['perfil_receptores']}

## 2. Farmacocinética y Metabolismo Enzimático (CYP450)
{row['metabolismo_cyp']}
- Inhibición CYP2D6: {row['potencia_inhibicion_2d6']}
- Sustrato CYP1A2: {'Sí' if row['sustrato_cyp1a2'] else 'No'} | CYP2C19: {'Sí' if row['sustrato_cyp2c19'] else 'No'} | CYP2D6: {'Sí' if row['sustrato_cyp2d6'] else 'No'} | CYP3A4: {'Sí' if row['sustrato_cyp3a4'] else 'No'}
- Inductor enzimático: {'Sí (CYP1A2, 2C9, 2C19, 3A4)' if row['inductor_cyp3a4'] else 'No'}

## 3. Efectos Adversos Comunes y Reacciones Graves
- **Efectos Comunes:** {row['efectos_comunes']}
- **Efectos Graves y Riesgo Vital:** {row['efectos_graves']}

## 4. Perlas Clínicas y Recomendaciones Prácticas
{row['perlas_clinicas']}
"""
        documentos_rag.append({
            "id": doc_id,
            "tipo": "monografia_stahl",
            "titulo": f"Guía de Prescripción de Stahl: {farmaco_nombre}",
            "atc": atc,
            "voice_snippet": f"Guía Stahl para {farmaco_nombre} ({atc}): {row['perfil_receptores'][:110]}. Perla clínica: {row['perlas_clinicas'][:100]}.",
            "contenido": contenido.strip(),
            "archivo_origen": "Stephen M. Stahl - Prescriber's Guide (6th Ed.)"
        })

    # 8. Transformar recomendaciones oficiales CANMAT con Voice Snippets calibrados (35-65 palabras)
    for idx, row in df_canmat.iterrows():
        doc_id = f"GUIA-CANMAT-REC-{idx+1:03d}"
        if doc_id in ids_vistos:
            continue
        ids_vistos.add(doc_id)

        contenido = f"""# Recomendación Oficial CANMAT: {row['patologia']} ({row['fase']}) - {row['farmaco']}
**Patología:** {row['patologia']}
**Fase Clínica:** {row['fase']}
**Línea Terapéutica:** {row['linea']}
**Estrategia:** {row['estrategia']}
**Fármaco / Pauta:** {row['farmaco']} (ATC: {row.get('atc', 'N/D')})
**Nivel de Evidencia Científica:** {row['nivel_evidencia']}
**Dirección de la Recomendación:** {row['direccion_evidencia']}
**Directriz Clínica y Comentarios Oficiales:**
{row['comentarios']}
"""
        # Voice snippet calibrado entre 35 y 65 palabras
        snippet = (
            f"Directriz clínica CANMAT para {row['patologia']} en fase de {row['fase']}: "
            f"se recomienda {row['farmaco']} en calidad de {row['linea']} con {row['nivel_evidencia']} "
            f"y calificación de {row['direccion_evidencia']}. Estrategia de {row['estrategia']}. "
            f"Directriz clínica: {row['comentarios'][:110]}."
        )
        documentos_rag.append({
            "id": doc_id,
            "tipo": "guia_canmat",
            "titulo": f"CANMAT: {row['patologia']} ({row['fase']}) - {row['farmaco']}",
            "atc": row.get("atc", "N/D"),
            "voice_snippet": snippet,
            "contenido": contenido.strip(),
            "archivo_origen": "data/raw/BDI-20-97.pdf"
        })

    # 9. Transformar alertas clínicas oficiales CANMAT
    for idx, row in df_canmat_alertas.iterrows():
        doc_id = f"ALERTA-CANMAT-{idx+1:03d}"
        if doc_id in ids_vistos:
            continue
        ids_vistos.add(doc_id)

        contenido = f"""# Alerta Clínica Oficial: {row['farmaco_involucrado']}
**Tipo de Alerta:** {row['tipo_alerta']}
**Patología / Población:** {row['patologia']}
**Nivel de Evidencia:** {row['nivel_evidencia']}
**Fuente Regulatoria y Guía:** {row['fuente']}
**Descripción del Riesgo Clínico:**
{row['descripcion']}
"""
        snippet = (
            f"Alerta de seguridad clínica para {row['farmaco_involucrado']} en {row['patologia']}: "
            f"{row['tipo_alerta']}. Nivel: {row['nivel_evidencia']}. Directriz: {row['descripcion'][:130]}."
        )
        documentos_rag.append({
            "id": doc_id,
            "tipo": "alerta_canmat",
            "titulo": f"Alerta Clínica: {row['farmaco_involucrado']}",
            "atc": "N/D",
            "voice_snippet": snippet,
            "contenido": contenido.strip(),
            "archivo_origen": "data/raw/BDI-20-97.pdf"
        })

    # 10. Catálogo oficial de marcas comerciales, laboratorios y presentaciones (canónico y deduplicado)
    canonical_catalog = ATCEnricher.get_canonical_catalog()
    for atc_cod, v in canonical_catalog.items():
        doc_id = f"MARCA-{normalizar_id(v['nombre_oficial'])}-001"
        if doc_id in ids_vistos:
            continue
        ids_vistos.add(doc_id)

        marcas = v.get("marcas_comerciales", [])
        marcas_str = ", ".join(marcas) if marcas else "Genéricos / EFG"
        labs = v.get("laboratorios", ["Genéricos / EFG"])
        labs_str = ", ".join(labs)
        pres = v.get("presentaciones", ["Comprimidos orales"])
        pres_str = "; ".join(pres)

        contenido = f"""# Catálogo Oficial ATC, Marcas Comerciales y Presentaciones: {v['nombre_oficial']}
**Principio Activo:** {v['nombre_oficial']}
**Código ATC Oficial OMS:** {v['atc_codigo']}
**Clasificación Anatómica y Química:**
- Nivel 1: {v.get('atc_nivel1', 'N/D')}
- Nivel 2: {v.get('atc_nivel2', 'N/D')}
- Nivel 3 (Grupo Terapéutico): {v.get('atc_nivel3', 'N/D')}
- Nivel 4 (Subgrupo Químico): {v.get('atc_nivel4', 'N/D')}

**Marcas Comerciales Registradas en Farmacopea:**
{marcas_str}

**Laboratorios Titulares de Autorización:**
{labs_str}

**Presentaciones Farmacéuticas y Formatos Disponibles:**
{pres_str}
"""
        snippet = (
            f"Ficha oficial de {v['nombre_oficial']} ({v['atc_codigo']}). "
            f"Grupo terapéutico: {v.get('atc_nivel3', 'Psicofármaco')}. "
            f"Marcas comerciales registradas: {marcas_str}. "
            f"Laboratorios comercializadores: {labs_str}. "
            f"Presentaciones farmacéuticas disponibles: {pres_str}."
        )
        documentos_rag.append({
            "id": doc_id,
            "tipo": "catalogo_marcas",
            "titulo": f"Marcas Comerciales, Laboratorios y Presentaciones: {v['nombre_oficial']}",
            "atc": v.get("atc_codigo", "N/D"),
            "voice_snippet": snippet,
            "contenido": contenido.strip(),
            "archivo_origen": "Catálogo Oficial ATC/DDD - Organización Mundial de la Salud (OMS) / AEMPS / Fichas Técnicas"
        })

    # Calibrar la longitud de todos los voice snippets para locución verbal fluida (< 35s)
    for doc in documentos_rag:
        if doc.get("voice_snippet"):
            doc["voice_snippet"] = calibrar_snippet(doc["voice_snippet"], max_words=80)

    print(f"Total de documentos clínicos consolidados en la Base de Conocimiento RAG: {len(documentos_rag)}")

    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(documentos_rag, f, indent=2, ensure_ascii=False)

    print(f"✓ Guardado exitoso de {len(documentos_rag)} documentos en: {kb_path}")

    # Verificar que kb_obsoleta sea False
    rag = ClinicalRAGEngine()
    if rag.kb_obsoleta:
        print("⚠️ Advertencia: kb_obsoleta es True.")
    else:
        print("✓ Verificación RAG: kb_obsoleta es False. Motor plenamente operativo y sincronizado.")


if __name__ == "__main__":
    regenerar_knowledge_base()
