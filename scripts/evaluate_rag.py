#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_rag.py
Evaluación sistemática del Motor RAG Clínico de Sofia:
- Benchmark formal de 20 casos clínicos representativos (Monografías, Citocromos CYP450,
  Guías CANMAT, Cascadas de Prescripción, Marcas Comerciales y Consultas Fuera de Cobertura).
- Métricas estándar de recuperación y generación (Recall@1, Recall@3, Recall@5, MRR, Answer Relevance,
  Groundedness y Trazabilidad Ética [DOC-ID]).
- Evaluación cuantitativa de ranking, cobertura y precisión de citas documentales.
"""

import sys
import time
import json
import re
from pathlib import Path
from typing import Dict, List, Any

# Configurar directorio raíz portable
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.rag.engine import ClinicalRAGEngine


CASOS_BENCHMARK = [
    {
        "id": "CASO_01_POSOLOGIA_Y_MARCAS",
        "categoria": "Monografía y Posología (Sertralina)",
        "pregunta": "¿Qué laboratorios comercializan la Sertralina, cuáles son sus marcas comerciales principales y cuál es su posología recomendada de inicio y mantenimiento?",
        "entidades_esperadas": ["Sertralina", "50", "200", "Zoloft", "Serlift"],
        "documentos_esperados": ["MONO-SERTRALINA-001", "MARCA-SERTRALINA-001"],
        "es_fuera_de_alcance": False,
        "criterio_grounding": "Debe citar [MONO-SERTRALINA-001], detallar rango 50-200 mg/día y marcas comerciales (Zoloft, Serlift, Aremis) sin atribuciones personales."
    },
    {
        "id": "CASO_02_INTERACCIONES_CYP450",
        "categoria": "Farmacocinética y Citocromos (CYP2D6)",
        "pregunta": "¿Qué interacciones farmacocinéticas y precauciones enzimáticas con el citocromo CYP2D6 se describen para Paroxetina y Bupropión?",
        "entidades_esperadas": ["Paroxetina", "Bupropión", "CYP2D6", "inhibidor", "sustrato"],
        "documentos_esperados": ["INTER-CYP2D6-001", "MONO-BUPROPION-004", "MONO-PAROXETINA-003"],
        "es_fuera_de_alcance": False,
        "criterio_grounding": "Debe citar [INTER-CYP2D6-001], identificar la inhibición potente de CYP2D6 por Paroxetina y Bupropión y el riesgo sobre sustratos."
    },
    {
        "id": "CASO_03_DIRECTRICES_CANMAT",
        "categoria": "Medicina Basada en la Evidencia (CANMAT)",
        "pregunta": "¿Qué fármacos se recomiendan en primera línea para un episodio de manía aguda y por qué se desaconseja el uso de antidepresivos en monoterapia en depresión bipolar?",
        "entidades_esperadas": ["Litio", "Quetiapina", "viraje", "manía", "Level 1"],
        "documentos_esperados": ["GUIA-CANMAT-MANIA-001", "GUIA-CANMAT-DEP-002", "ALERTA-ISRS-VIRAJE-004"],
        "es_fuera_de_alcance": False,
        "criterio_grounding": "Debe citar directrices CANMAT ([GUIA-CANMAT-MANIA-001] y/o [GUIA-CANMAT-DEP-002]) con Litio/Quetiapina para manía y advertir riesgo de viraje maníaco con antidepresivos en monoterapia."
    },
    {
        "id": "CASO_04_FARMACOVIGILANCIA_CASCADA",
        "categoria": "Farmacovigilancia y Cascadas de Prescripción (Haloperidol - Biperideno)",
        "pregunta": "Un paciente en tratamiento con Haloperidol comienza con rigidez y temblor y se le prescribe medicación antiparkinsoniana como biperideno. ¿Cuál es el riesgo de cascada de prescripción y qué pauta de desprescripción se aconseja?",
        "entidades_esperadas": ["Haloperidol", "Biperideno", "cascada", "extrapiramidal", "desprescripción"],
        "documentos_esperados": ["CASC-HAL-BIP-001", "MONO-HALOPERIDOL-008"],
        "es_fuera_de_alcance": False,
        "criterio_grounding": "Debe citar [CASC-HAL-BIP-001], emitir Alerta de Cascada de Prescripción (Haloperidol + Biperideno) y advertir sobre discinesia tardía."
    },
    {
        "id": "CASO_05_LITIO_MONITORIZACION_Y_RANGOS",
        "categoria": "Monitorización Estrecha (Carbonato de Litio)",
        "pregunta": "¿Cuál es el rango terapéutico plasmático de la litemia en fase de mantenimiento, cuándo debe extraerse la muestra de sangre y qué pruebas analíticas semestrales son obligatorias?",
        "entidades_esperadas": ["litio", "litemia", "0.6", "12 horas", "creatinina"],
        "documentos_esperados": ["MONO-LITIO-005", "ALERTA-TOX-LITIO-003"],
        "es_fuera_de_alcance": False,
        "criterio_grounding": "Debe citar [MONO-LITIO-005] o [ALERTA-TOX-LITIO-003], especificando rango 0.6-0.8 (hasta 1.0) mEq/L, toma a las 12h y control de función renal/tiroidea."
    },
    {
        "id": "CASO_06_LITIO_INTERACCION_AINES",
        "categoria": "Interacciones Críticas Renales (Litio y AINEs)",
        "pregunta": "¿Por qué están contraindicados el ibuprofeno y los diuréticos tiazídicos en pacientes que reciben litio y qué analgésico se recomienda para el dolor?",
        "entidades_esperadas": ["litio", "ibuprofeno", "renal", "intoxicación", "paracetamol"],
        "documentos_esperados": ["INTER-LIT-RENAL-004", "MONO-LITIO-005"],
        "es_fuera_de_alcance": False,
        "criterio_grounding": "Debe citar [INTER-LIT-RENAL-004] o [MONO-LITIO-005], explicar reducción de aclaramiento renal y sugerir paracetamol."
    },
    {
        "id": "CASO_07_CLOZAPINA_AGRANULOCITOSIS",
        "categoria": "Seguridad Hematológica (Clozapina)",
        "pregunta": "¿Cuál es el protocolo de recuento absoluto de neutrófilos (RAN) para clozapina y con qué umbral de neutrófilos debe interrumpirse inmediatamente el tratamiento?",
        "entidades_esperadas": ["clozapina", "neutrófilos", "1000", "agranulocitosis", "hemograma"],
        "documentos_esperados": ["ALERTA-CLOZ-AGRAN-001", "MONO-CLOZAPINA-006"],
        "es_fuera_de_alcance": False,
        "criterio_grounding": "Debe citar [ALERTA-CLOZ-AGRAN-001] o [MONO-CLOZAPINA-006] con umbral de suspensión en RAN < 1000/µL."
    },
    {
        "id": "CASO_08_CLOZAPINA_TABACO_CYP1A2",
        "categoria": "Farmacocinética y Tabaco (CYP1A2)",
        "pregunta": "Un paciente estabilizado con clozapina ingresa en un hospital y deja de fumar abruptamente. ¿Qué ocurre con sus niveles plasmáticos y qué citocromo está involucrado?",
        "entidades_esperadas": ["clozapina", "CYP1A2", "tabaco", "humo", "reducir"],
        "documentos_esperados": ["INTER-CYP1A2-002", "MONO-CLOZAPINA-006"],
        "es_fuera_de_alcance": False,
        "criterio_grounding": "Debe citar [INTER-CYP1A2-002] o [MONO-CLOZAPINA-006], explicar la pérdida de inducción de CYP1A2 y el riesgo de duplicación de niveles."
    },
    {
        "id": "CASO_09_LAMOTRIGINA_VALPROATO",
        "categoria": "Interacción Enzimática Crítica (Lamotrigina y Valproato)",
        "pregunta": "¿Qué ocurre al coadministrar lamotrigina con ácido valproico y cómo debe ajustarse la dosis inicial de lamotrigina?",
        "entidades_esperadas": ["lamotrigina", "valproato", "Stevens-Johnson", "reducir", "UGT1A4"],
        "documentos_esperados": ["INTER-LAM-VAL-003", "MONO-LAMOTRIGINA-009", "ALERTA-LAM-SJS-002"],
        "es_fuera_de_alcance": False,
        "criterio_grounding": "Debe citar [INTER-LAM-VAL-003], explicar la inhibición de glucuronidación y la pauta de inicio reducida al 50% o días alternos."
    },
    {
        "id": "CASO_10_LAMOTRIGINA_TITULACION",
        "categoria": "Seguridad Cutánea (Titulación de Lamotrigina)",
        "pregunta": "¿Cuál es el cronograma de escalado de dosis de lamotrigina en monoterapia y qué debe hacerse si el paciente interrumpe el tratamiento más de 5 días?",
        "entidades_esperadas": ["lamotrigina", "25", "titulación", "reiniciar", "Stevens-Johnson"],
        "documentos_esperados": ["MONO-LAMOTRIGINA-009", "ALERTA-LAM-SJS-002", "INTER-LAM-VAL-003"],
        "es_fuera_de_alcance": False,
        "criterio_grounding": "Debe citar [MONO-LAMOTRIGINA-009], indicando inicio con 25 mg/día y necesidad de reiniciar la titulación si hubo interrupción prolongada."
    },
    {
        "id": "CASO_11_FLUOXETINA_VIDA_MEDIA_LAVADO",
        "categoria": "Farmacocinética y Lavado (Fluoxetina)",
        "pregunta": "¿Cuál es la vida media de la norfluoxetina y cuánto tiempo de lavado debe esperarse antes de introducir un IMAO?",
        "entidades_esperadas": ["fluoxetina", "norfluoxetina", "semanas", "IMAO", "serotoninérgico"],
        "documentos_esperados": ["MONO-FLUOXETINA-002"],
        "es_fuera_de_alcance": False,
        "criterio_grounding": "Debe citar [MONO-FLUOXETINA-002] señalando 5 semanas de lavado obligatorio."
    },
    {
        "id": "CASO_12_RISPERIDONA_HIPERPROLACTINEMIA",
        "categoria": "Neuroendocrinología (Risperidona)",
        "pregunta": "¿Por qué la risperidona provoca hiperprolactinemia y galactorrea con mayor frecuencia que otros antipsicóticos atípicos?",
        "entidades_esperadas": ["risperidona", "prolactina", "D2", "tuberoinfundibular", "atípico"],
        "documentos_esperados": ["MONO-RISPERIDONA-007"],
        "es_fuera_de_alcance": False,
        "criterio_grounding": "Debe citar [MONO-RISPERIDONA-007] y referir el bloqueo dopaminérgico D2 en la vía tuberoinfundibular."
    },
    {
        "id": "CASO_13_VALPROATO_TERATOGENICIDAD",
        "categoria": "Poblaciones Especiales (Ácido Valproico)",
        "pregunta": "¿Cuáles son las advertencias de seguridad y riesgos teratogénicos del ácido valproico en mujeres en edad fértil?",
        "entidades_esperadas": ["valproato", "teratogénico", "espina bífida", "tubo neural", "anticoncepción"],
        "documentos_esperados": ["MONO-VALPROATO-010"],
        "es_fuera_de_alcance": False,
        "criterio_grounding": "Debe citar [MONO-VALPROATO-010] indicando defectos del tubo neural y plan de prevención de embarazos."
    },
    {
        "id": "CASO_14_CASCADA_METABOLICA",
        "categoria": "Farmacovigilancia (Cascada Metabólica)",
        "pregunta": "¿Qué cascada de prescripción ocurre frecuentemente tras iniciar antipsicóticos atípicos con alta ganancia de peso como olanzapina o clozapina?",
        "entidades_esperadas": ["cascada", "peso", "metabólico", "antipsicóticos", "glucosa"],
        "documentos_esperados": ["CASC-METAB-002"],
        "es_fuera_de_alcance": False,
        "criterio_grounding": "Debe citar [CASC-METAB-002] y alertar sobre la cascada con hipolipemiantes o antidiabéticos sin abordar el antipsicótico."
    },
    {
        "id": "CASO_15_CANMAT_TDM_RESISTENCIA",
        "categoria": "Medicina Basada en Evidencia (Depresión Resistente)",
        "pregunta": "Según las directrices CANMAT, ¿cuándo se aconseja potenciar un antidepresivo y qué agentes de primera línea se utilizan como add-on en TDM resistente?",
        "entidades_esperadas": ["CANMAT", "potenciación", "litio", "atípico", "Level 1"],
        "documentos_esperados": ["GUIA-CANMAT-TDM-RESIST-005", "GUIA-CANMAT-CSV-011"],
        "es_fuera_de_alcance": False,
        "criterio_grounding": "Debe citar [GUIA-CANMAT-TDM-RESIST-005] recomendando adición de Litio o antipsicóticos atípicos (quetiapina, aripiprazol)."
    },
    {
        "id": "CASO_16_CANMAT_MANTENIMIENTO_BIPOLAR",
        "categoria": "Medicina Basada en Evidencia (Mantenimiento Bipolar)",
        "pregunta": "¿Qué fármacos se sitúan en primera línea de mantenimiento según CANMAT 2018 y cuál es la polaridad de prevención del litio versus lamotrigina?",
        "entidades_esperadas": ["CANMAT", "mantenimiento", "litio", "lamotrigina", "bipolar"],
        "documentos_esperados": ["GUIA-CANMAT-MANT-003", "MONO-LITIO-005", "MONO-LAMOTRIGINA-009"],
        "es_fuera_de_alcance": False,
        "criterio_grounding": "Debe citar [GUIA-CANMAT-MANT-003], indicando litio para manía/ambos polos y lamotrigina preferente para polo depresivo."
    },
    {
        "id": "CASO_17_MARCAS_HALOPERIDOL",
        "categoria": "Catálogo Comercial y Formatos (Haloperidol)",
        "pregunta": "¿Bajo qué nombres comerciales, laboratorios y presentaciones se comercializa el haloperidol en el mercado clínico?",
        "entidades_esperadas": ["Haloperidol", "Haldol", "gotas", "decanoato", "ampollas"],
        "documentos_esperados": ["MARCA-HALOPERIDOL-001", "MONO-HALOPERIDOL-008"],
        "es_fuera_de_alcance": False,
        "criterio_grounding": "Debe citar [MARCA-HALOPERIDOL-001] y referir Haldol y sus formulaciones oral e inyectable."
    },
    {
        "id": "CASO_18_MARCAS_BUPROPION",
        "categoria": "Catálogo Comercial y Formatos (Bupropión)",
        "pregunta": "¿Cuáles son las marcas comerciales y laboratorios de referencia para el bupropión en depresión mayor y cesación tabáquica?",
        "entidades_esperadas": ["Bupropión", "Wellbutrin", "Elontril", "GSK", "Zyban"],
        "documentos_esperados": ["MARCA-BUPROPION-001", "MONO-BUPROPION-004"],
        "es_fuera_de_alcance": False,
        "criterio_grounding": "Debe citar [MARCA-BUPROPION-001] y detallar Wellbutrin, Elontril y Zyban."
    },
    {
        "id": "CASO_19_FUERA_DE_ALCANCE_GASTRONOMIA",
        "categoria": "Guardia de Cobertura y Seguridad (Consulta Fuera de Dominio)",
        "pregunta": "¿Cómo hacer una tarta de manzana casera paso a paso?",
        "entidades_esperadas": [],
        "documentos_esperados": [],
        "es_fuera_de_alcance": True,
        "criterio_grounding": "El motor debe activar la guardia de cobertura, declarar falta de respaldo documental y no inventar recetas."
    },
    {
        "id": "CASO_20_FUERA_DE_ALCANCE_NO_PSIQUIATRICO",
        "categoria": "Guardia de Cobertura y Seguridad (Medicina General No Psiquiátrica)",
        "pregunta": "¿Cuáles son los antibióticos de primera elección para una neumonía bacteriana adquirida en la comunidad?",
        "entidades_esperadas": [],
        "documentos_esperados": [],
        "es_fuera_de_alcance": True,
        "criterio_grounding": "El motor debe abstenerse por falta de cobertura en psicofarmacología y no emitir recomendación infectológica sin base."
    }
]


def evaluar_caso(rag_engine: ClinicalRAGEngine, caso: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
    res = rag_engine.consultar(caso["pregunta"], top_k=top_k)

    respuesta = res["respuesta"]
    docs_recuperados = res["documentos_recuperados"]
    voice_snippet = res["voice_snippet"]
    es_fuera = res.get("fuera_de_alcance", False)
    lat_recuperacion = res.get("latencia_recuperacion_seg", 0.0)
    lat_generacion = res.get("latencia_generacion_seg", 0.0)
    lat_total = res.get("latencia_total_seg", lat_recuperacion + lat_generacion)

    # Evaluación para consultas fuera de alcance
    if caso.get("es_fuera_de_alcance"):
        abstencion_correcta = bool(es_fuera and res["modo_respuesta"] == "sin_cobertura")
        return {
            "caso_id": caso["id"],
            "categoria": caso["categoria"],
            "pregunta": caso["pregunta"],
            "es_fuera_de_alcance": True,
            "abstencion_correcta": abstencion_correcta,
            "context_recall@1": 1.0 if abstencion_correcta else 0.0,
            "context_recall@3": 1.0 if abstencion_correcta else 0.0,
            "context_recall@5": 1.0 if abstencion_correcta else 0.0,
            "mrr": 1.0 if abstencion_correcta else 0.0,
            "relevance_score": 1.0 if abstencion_correcta else 0.0,
            "groundedness_score": 1.0 if abstencion_correcta else 0.0,
            "top_doc_id": "Ninguno (Abstención Segura)",
            "top_documento": "Sin cobertura",
            "top_score_similitud": 0.0,
            "latencia_recuperacion_segundos": round(lat_recuperacion, 4),
            "latencia_generacion_segundos": round(lat_generacion, 4),
            "latencia_total_segundos": round(lat_total, 4),
            "trazabilidad_etica_valida": True,
            "voice_snippet_valido": True,
            "generado_con_llm": res["generado_con_llm"]
        }

    # 1. Context Recall@k y MRR (Mean Reciprocal Rank)
    ids_recuperados = [d.get("id", "").upper() for d, _ in docs_recuperados]
    docs_esperados_upper = [d.upper() for d in caso["documentos_esperados"]]

    # Encontrar la primera posición (1-indexed) de un documento esperado
    primer_rango = None
    for idx, doc_id in enumerate(ids_recuperados, 1):
        if any(esp in doc_id or doc_id in esp for esp in docs_esperados_upper):
            primer_rango = idx
            break

    recall_1 = 1.0 if (primer_rango is not None and primer_rango <= 1) else 0.0
    recall_3 = 1.0 if (primer_rango is not None and primer_rango <= 3) else 0.0
    recall_5 = 1.0 if (primer_rango is not None and primer_rango <= 5) else 0.0
    mrr = (1.0 / primer_rango) if primer_rango is not None else 0.0

    top_score = docs_recuperados[0][1] if docs_recuperados else 0.0
    top_doc = docs_recuperados[0][0] if docs_recuperados else {}

    # 2. Answer Relevance & Cobertura de Entidades Clínicas
    entidades_presentes = [
        ent for ent in caso["entidades_esperadas"]
        if ent.lower() in respuesta.lower() or ent.lower() in voice_snippet.lower()
    ]
    relevance_score = len(entidades_presentes) / len(caso["entidades_esperadas"]) if caso["entidades_esperadas"] else 1.0

    # 3. Groundedness / Fidelidad de Estructura y Citas
    secciones_requeridas = [
        "1. Juicio Clínico Inmediato",
        "2. Mecanismo Farmacológico",
        "3. Pauta Operativa",
        "4. Alertas de Seguridad"
    ]
    secciones_presentes = sum(1 for sec in secciones_requeridas if sec.lower() in respuesta.lower())
    groundedness_score = (secciones_presentes / len(secciones_requeridas)) * 0.5 + (0.5 if relevance_score >= 0.6 else 0.2)

    # 4. Calidad del Voice Snippet (conciso y verbalizable)
    voice_ok = bool(voice_snippet and len(voice_snippet.split()) <= 110)

    # 5. Auditoría Ética: Trazabilidad [DOC-ID] y Cero Citas Apócrifas
    citas_brutas = re.findall(r"\[([A-ZÁÉÍÓÚÑ0-9_\-]+)\]", respuesta)
    citas_doc_id = [
        c for c in citas_brutas
        if any(c.startswith(p) for p in ["MONO-", "INTER-", "CASC-", "GUIA-", "ALERTA-", "MARCA-"])
    ]
    ids_recuperados_set = set(ids_recuperados)
    citas_validas = [c for c in citas_doc_id if c.upper() in ids_recuperados_set]

    cita_apocrifa_detectada = bool(
        "según la dra. jufe" in respuesta.lower() or
        "según el dr. stahl" in respuesta.lower() or
        "afirma la dra. jufe" in respuesta.lower() or
        "afirma el dr. stahl" in respuesta.lower()
    )

    trazabilidad_etica = bool(
        len(citas_validas) >= 1 and
        len(citas_validas) == len(citas_doc_id) and
        not cita_apocrifa_detectada
    )

    return {
        "caso_id": caso["id"],
        "categoria": caso["categoria"],
        "pregunta": caso["pregunta"],
        "es_fuera_de_alcance": False,
        "latencia_recuperacion_segundos": round(lat_recuperacion, 4),
        "latencia_generacion_segundos": round(lat_generacion, 4),
        "latencia_total_segundos": round(lat_total, 4),
        "top_documento": top_doc.get("titulo", "Ninguno"),
        "top_doc_id": top_doc.get("id", "N/D"),
        "top_score_similitud": round(top_score, 4),
        "primer_rango": primer_rango,
        "context_recall@1": recall_1,
        "context_recall@3": recall_3,
        "context_recall@5": recall_5,
        "mrr": round(mrr, 4),
        "entidades_halladas": entidades_presentes,
        "cobertura_entidades": f"{len(entidades_presentes)}/{len(caso['entidades_esperadas'])}",
        "relevance_score": round(relevance_score, 3),
        "groundedness_score": round(groundedness_score, 3),
        "citas_identificadores_halladas": [f"[{c}]" for c in citas_doc_id],
        "citas_validas_recuperadas": [f"[{c}]" for c in citas_validas],
        "trazabilidad_etica_valida": trazabilidad_etica,
        "voice_snippet_valido": voice_ok,
        "generado_con_llm": res["generado_con_llm"]
    }


def main():
    print("==================================================================")
    print("🧪 BENCHMARK FORMAL DEL MOTOR RAG SOFIA (N=20 CASOS CLÍNICOS)")
    print("==================================================================")

    rag = ClinicalRAGEngine()
    print(f"✓ Motor RAG cargado con {len(rag.docs)} documentos clínicos padre.")
    print(f"✓ Total fragmentos indexados (chunks enriquecidos): {len(rag.vector_store.chunks)}")
    print(f"✓ Motor de recuperación activo: {'Ollama denso + BM25' if rag.vector_store.use_ollama else 'TF-IDF híbrido + BM25'}")

    resultados = []
    print(f"\nEjecutando evaluación sobre los {len(CASOS_BENCHMARK)} escenarios clínicos...")
    for idx, caso in enumerate(CASOS_BENCHMARK, 1):
        eval_res = evaluar_caso(rag, caso, top_k=5)
        resultados.append(eval_res)
        if eval_res.get("es_fuera_de_alcance"):
            estado = "✓ Abstención Correcta" if eval_res["abstencion_correcta"] else "✗ Fallo de Abstención"
            print(f"[{idx:02d}/20] {caso['id']} ({caso['categoria']}) -> {estado}")
        else:
            rank_str = f"Rank #{eval_res['primer_rango']}" if eval_res["primer_rango"] else "No recuperado"
            print(f"[{idx:02d}/20] {caso['id']} -> {rank_str} | R@3: {int(eval_res['context_recall@3'])} | MRR: {eval_res['mrr']:.2f} | Top: {eval_res['top_doc_id']}")

    # Promedios y Métricas Agregadas
    casos_in = [r for r in resultados if not r.get("es_fuera_de_alcance")]
    avg_r1 = sum(r["context_recall@1"] for r in casos_in) / len(casos_in)
    avg_r3 = sum(r["context_recall@3"] for r in casos_in) / len(casos_in)
    avg_r5 = sum(r["context_recall@5"] for r in casos_in) / len(casos_in)
    avg_mrr = sum(r["mrr"] for r in casos_in) / len(casos_in)

    avg_relevance = sum(r["relevance_score"] for r in resultados) / len(resultados)
    avg_groundedness = sum(r["groundedness_score"] for r in resultados) / len(resultados)
    avg_ethical = sum(1.0 for r in resultados if r["trazabilidad_etica_valida"]) / len(resultados)
    avg_lat_rec = sum(r["latencia_recuperacion_segundos"] for r in resultados) / len(resultados)
    avg_lat_gen = sum(r["latencia_generacion_segundos"] for r in resultados) / len(resultados)
    avg_lat_total = sum(r["latencia_total_segundos"] for r in resultados) / len(resultados)

    print("\n==================================================================")
    print("📊 RESULTADOS AGREGADOS DEL BENCHMARK CLÍNICO (N=20)")
    print("==================================================================")
    print(f"• Context Recall@1:             {avg_r1 * 100:.1f}%")
    print(f"• Context Recall@3:             {avg_r3 * 100:.1f}%")
    print(f"• Context Recall@5:             {avg_r5 * 100:.1f}%")
    print(f"• Mean Reciprocal Rank (MRR):   {avg_mrr:.4f}")
    print(f"• Answer Relevance Promedio:    {avg_relevance * 100:.1f}%")
    print(f"• Groundedness / Fidelidad:     {avg_groundedness * 100:.1f}%")
    print(f"• Trazabilidad Ética [DOC-ID]:  {avg_ethical * 100:.1f}%")
    print(f"• Latencia Recuperación Media:  {avg_lat_rec * 1000:.2f} ms")
    print(f"• Latencia Generación Media:    {avg_lat_gen:.3f} s")
    print(f"• Latencia Total Media:         {avg_lat_total:.3f} s")
    print("==================================================================")

    # Guardar informe estructurado
    out_report = WORKSPACE_ROOT / "data" / "processed" / "rag_evaluation_report.json"
    report_data = {
        "timestamp_evaluacion": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_casos_evaluados": len(resultados),
        "modo_ejecucion": "Ollama denso + BM25 híbrido" if any(r["generado_con_llm"] for r in resultados) else "TF-IDF híbrido + BM25 (sin LLM activo)",
        "nota_metodologica": "Benchmark formal de 20 casos clínicos estructurados con métricas Recall@1, Recall@3, Recall@5, MRR y Tríada de RAG.",
        "metricas_clave": {
            "recall_at_1": round(avg_r1, 4),
            "recall_at_3": round(avg_r3, 4),
            "recall_at_5": round(avg_r5, 4),
            "mrr": round(avg_mrr, 4),
            "answer_relevance": round(avg_relevance, 4),
            "faithfulness_groundedness": round(avg_groundedness, 4),
            "trazabilidad_etica_doc_id": round(avg_ethical, 4),
            "latencia_recuperacion_promedio_seg": round(avg_lat_rec, 4),
            "latencia_generacion_promedio_seg": round(avg_lat_gen, 4),
            "latencia_segundos_promedio": round(avg_lat_total, 4)
        },
        "detalle_por_caso": resultados
    }

    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    rel_out = out_report.relative_to(WORKSPACE_ROOT)
    print(f"\n✓ Informe completo exportado a: {rel_out}")


if __name__ == "__main__":
    main()
