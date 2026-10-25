#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_rag.py
Evaluación sistemática de la Tríada de RAG (Fidelidad, Relevancia y Precisión de Contexto)
sobre los 4 casos clínicos benchmark de Sofia (Jufe, Stahl, CANMAT y Farmacovigilancia).
Diseñado con estricta portabilidad relativa y cero rutas de usuario.
"""

import sys
import time
import json
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
        "documentos_esperados": ["sertralina"],
        "criterio_grounding": "Debe citar [MONO-SERTRALINA-001], detallar rango 50-200 mg/día y marcas comerciales (Zoloft, Serlift, Aremis) sin atribuciones personales."
    },
    {
        "id": "CASO_02_INTERACCIONES_CYP450",
        "categoria": "Farmacocinética y Citocromos (CYP2D6)",
        "pregunta": "¿Qué interacciones farmacocinéticas y precauciones enzimáticas con el citocromo CYP2D6 se describen para Paroxetina y Bupropión?",
        "entidades_esperadas": ["Paroxetina", "Bupropión", "CYP2D6", "inhibidor", "sustrato"],
        "documentos_esperados": ["cyp2d6"],
        "criterio_grounding": "Debe citar [INTER-CYP2D6-001], identificar la inhibición potente de CYP2D6 por Paroxetina y Bupropión y el riesgo sobre sustratos."
    },
    {
        "id": "CASO_03_DIRECTRICES_CANMAT",
        "categoria": "Medicina Basada en la Evidencia (CANMAT)",
        "pregunta": "¿Qué fármacos se recomiendan en primera línea para un episodio de manía aguda y por qué se desaconseja el uso de antidepresivos en monoterapia en depresión bipolar?",
        "entidades_esperadas": ["Litio", "Quetiapina", "viraje", "manía", "Level 1"],
        "documentos_esperados": ["canmat_bipolar_mania", "canmat_bipolar_depresion", "canmat"],
        "criterio_grounding": "Debe citar directrices CANMAT ([GUIA-CANMAT-MANIA-001] y/o [GUIA-CANMAT-DEP-002]) con Litio/Quetiapina para manía y advertir riesgo de viraje maníaco con antidepresivos en monoterapia."
    },
    {
        "id": "CASO_04_FARMACOVIGILANCIA_CASCADA",
        "categoria": "Farmacovigilancia y Cascadas de Prescripción (Haloperidol - Biperideno)",
        "pregunta": "Un paciente en tratamiento con Haloperidol comienza con rigidez y temblor y se le prescribe medicación antiparkinsoniana como biperideno. ¿Cuál es el riesgo de cascada de prescripción y qué pauta de desprescripción se aconseja?",
        "entidades_esperadas": ["Haloperidol", "Biperideno", "cascada", "extrapiramidal", "desprescripción"],
        "documentos_esperados": ["haloperidol"],
        "criterio_grounding": "Debe citar [CASC-HAL-BIP-001], emitir Alerta de Cascada de Prescripción (Haloperidol + Biperideno) y advertir sobre discinesia tardía."
    }
]


def evaluar_caso(rag_engine: ClinicalRAGEngine, caso: Dict[str, Any]) -> Dict[str, Any]:
    import re
    res = rag_engine.consultar(caso["pregunta"], top_k=3)  # mismo top_k que la app

    respuesta = res["respuesta"]
    docs_recuperados = res["documentos_recuperados"]
    voice_snippet = res["voice_snippet"]
    lat_recuperacion = res.get("latencia_recuperacion_seg", 0.0)
    lat_generacion = res.get("latencia_generacion_seg", 0.0)
    lat_total = res.get("latencia_total_seg", lat_recuperacion + lat_generacion)

    # 1. Context Precision & Recall: ¿El documento canónico esperado está en el top-k?
    titulos_recuperados = [d["titulo"].lower() for d, _ in docs_recuperados]
    archivos_recuperados = [d["archivo_origen"].lower() for d, _ in docs_recuperados]
    ids_recuperados = [d.get("id", "").lower() for d, _ in docs_recuperados]
    cadena_docs = " ".join(titulos_recuperados + archivos_recuperados + ids_recuperados)

    docs_esp = caso.get("documentos_esperados", [caso.get("documento_esperado", "")])
    aciertos = sum(
        1 for esp in docs_esp
        if esp.lower() in cadena_docs or any(esp.lower() in t for t in titulos_recuperados)
    )
    context_recall = 1.0 if aciertos > 0 else 0.0
    top_score = docs_recuperados[0][1] if docs_recuperados else 0.0

    # 2. Answer Relevance & Coverage de Entidades Clínicas
    entidades_presentes = [
        ent for ent in caso["entidades_esperadas"]
        if ent.lower() in respuesta.lower() or ent.lower() in voice_snippet.lower()
    ]
    relevance_score = len(entidades_presentes) / len(caso["entidades_esperadas"])

    # 3. Groundedness / Fidelidad (Estructura en 4 partes + Ausencia de Alucinación)
    secciones_requeridas = [
        "1. Juicio Clínico Inmediato",
        "2. Mecanismo Farmacológico",
        "3. Pauta Operativa",
        "4. Alertas de Seguridad"
    ]
    secciones_presentes = sum(1 for sec in secciones_requeridas if sec.lower() in respuesta.lower())
    groundedness_score = (secciones_presentes / len(secciones_requeridas)) * 0.5 + (0.5 if relevance_score >= 0.6 else 0.2)

    # 4. Voice Snippet Quality (lectura fluida <= 110 palabras)
    voice_ok = bool(voice_snippet and len(voice_snippet.split()) <= 110)

    # 5. Auditoría Bioética: Trazabilidad por [DOC-ID] y Cero Citas Apócrifas
    citas_brutas = re.findall(r"\[([A-ZÁÉÍÓÚÑ0-9_\-]+)\]", respuesta)
    citas_doc_id = [
        c for c in citas_brutas
        if any(c.startswith(p) for p in ["MONO-", "INTER-", "CASC-", "GUIA-", "ALERTA-", "MARCA-"])
    ]
    ids_recuperados_set = {d.get("id", "").upper() for d, _ in docs_recuperados}
    citas_validas = [c for c in citas_doc_id if c.upper() in ids_recuperados_set]

    cita_apocrifa_detectada = bool(
        "según la dra. jufe" in respuesta.lower() or
        "según el dr. stahl" in respuesta.lower() or
        "afirma la dra. jufe" in respuesta.lower() or
        "afirma el dr. stahl" in respuesta.lower()
    )

    # Exigencia estricta: al menos un ID citado, todos los citados deben pertenecer a los recuperados, y cero citas apócrifas
    trazabilidad_etica = bool(
        len(citas_validas) >= 1 and
        len(citas_validas) == len(citas_doc_id) and
        not cita_apocrifa_detectada
    )

    return {
        "caso_id": caso["id"],
        "categoria": caso["categoria"],
        "pregunta": caso["pregunta"],
        "latencia_recuperacion_segundos": round(lat_recuperacion, 4),
        "latencia_generacion_segundos": round(lat_generacion, 4),
        "latencia_total_segundos": round(lat_total, 4),
        "latencia_segundos": round(lat_total, 4),
        "top_documento": docs_recuperados[0][0]["titulo"] if docs_recuperados else "Ninguno",
        "top_doc_id": docs_recuperados[0][0].get("id", "N/D") if docs_recuperados else "N/D",
        "top_score_similitud": round(top_score, 4),
        "context_recall": context_recall,
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
    print("🧪 EVALUACIÓN FORMAL DEL MOTOR RAG SOFIA (BENCHMARK CLÍNICO N=4)")
    print("==================================================================")

    rag = ClinicalRAGEngine()
    print(f"✓ Motor RAG inicializado con {len(rag.docs)} documentos clínicos.")
    print(f"✓ Modo de recuperación: {'Ollama denso' if rag.vector_store.use_ollama else 'TF-IDF híbrido de contingencia'}")

    resultados = []
    print("\nEjecutando pruebas sobre los 4 escenarios clínicos...")
    for caso in CASOS_BENCHMARK:
        print(f"\n▶ Evaluando: {caso['id']} ({caso['categoria']})")
        eval_res = evaluar_caso(rag, caso)
        resultados.append(eval_res)
        print(f"  - Top Match: {eval_res['top_documento']} (Score: {eval_res['top_score_similitud']})")
        print(f"  - Context Recall: {eval_res['context_recall'] * 100:.1f}%")
        print(f"  - Relevancia / Entidades: {eval_res['cobertura_entidades']} ({eval_res['relevance_score'] * 100:.1f}%)")
        print(f"  - Groundedness / Fidelidad: {eval_res['groundedness_score'] * 100:.1f}%")
        print(f"  - Citas [DOC-ID]: {eval_res['citas_identificadores_halladas']} | Válidas: {eval_res['citas_validas_recuperadas']} | Ética: {'✓ Conforme' if eval_res['trazabilidad_etica_valida'] else '✗ No conforme'}")
        print(f"  - Latencias: Recuperación {eval_res['latencia_recuperacion_segundos']*1000:.1f}ms | Generación {eval_res['latencia_generacion_segundos']:.3f}s | Total {eval_res['latencia_total_segundos']:.3f}s")
        print(f"  - Voice Snippet: {'✓' if eval_res['voice_snippet_valido'] else '✗'}")

    # Promedios Globales
    avg_recall = sum(r["context_recall"] for r in resultados) / len(resultados)
    avg_relevance = sum(r["relevance_score"] for r in resultados) / len(resultados)
    avg_groundedness = sum(r["groundedness_score"] for r in resultados) / len(resultados)
    avg_lat_rec = sum(r["latencia_recuperacion_segundos"] for r in resultados) / len(resultados)
    avg_lat_gen = sum(r["latencia_generacion_segundos"] for r in resultados) / len(resultados)
    avg_lat_total = sum(r["latencia_total_segundos"] for r in resultados) / len(resultados)
    avg_ethical = sum(1.0 for r in resultados if r["trazabilidad_etica_valida"]) / len(resultados)

    print("\n==================================================================")
    print("📊 RESUMEN CUANTITATIVO DE LA TRÍADA DE RAG SOFIA (Smoke Test N=4)")
    print("==================================================================")
    print(f"• Context Recall Promedio:      {avg_recall * 100:.1f}%")
    print(f"• Answer Relevance Promedio:    {avg_relevance * 100:.1f}%")
    print(f"• Groundedness (heurístico):    {avg_groundedness * 100:.1f}% (estructura 4 apartados + cobertura de entidades; no es fidelidad semántica)")
    print(f"• Trazabilidad Ética [DOC-ID]:  {avg_ethical * 100:.1f}% (Cero citas apócrifas, IDs en contexto)")
    print(f"• Latencia Recuperación Media:  {avg_lat_rec * 1000:.2f} ms")
    print(f"• Latencia Generación Media:    {avg_lat_gen:.3f} s")
    print(f"• Latencia Total Media:         {avg_lat_total:.3f} s")
    n_voice = sum(1 for r in resultados if r["voice_snippet_valido"])
    print(f"• Cobertura Voice Snippet:      {n_voice / len(resultados) * 100:.0f}% ({n_voice}/{len(resultados)} casos)")
    print("==================================================================")

    # Guardar informe estructurado portable en data/processed/
    out_report = WORKSPACE_ROOT / "data" / "processed" / "rag_evaluation_report.json"
    report_data = {
        "timestamp_evaluacion": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_casos_evaluados": len(resultados),
        "modo_ejecucion": "Ollama denso + LLM" if any(r["generado_con_llm"] for r in resultados) else "TF-IDF híbrido determinista (sin LLM activo)",
        "nota_metodologica": "Smoke test preliminar (N=4 casos representativos de Jufe, Stahl, CANMAT y Farmacovigilancia). La métrica faithfulness_groundedness es heurística (estructura en 4 apartados + cobertura de entidades), no mide fidelidad semántica. No constituye ensayo clínico formal.",
        "promedios_triada_rag": {
            "context_recall": round(avg_recall, 4),
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
    print(f"\n✓ Informe completo de evaluación exportado a: {rel_out}")


if __name__ == "__main__":
    main()
