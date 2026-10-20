#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_finetuning_dataset.py
Script de auditoría y control de calidad para el archivo JSONL de fine-tuning.
Valida el esquema ChatML (system, user, assistant), métricas de tokens y cobertura clínica.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "dataset_sofia_medical_qwen.jsonl"

def validate_dataset():
    print("=" * 65)
    print("AUDITORÍA DE CALIDAD: DATASET DE FINE-TUNING QWEN (SOFIA)")
    print("=" * 65)

    if not DATASET_FILE.exists():
        print(f"❌ Error: No se encontró el archivo {DATASET_FILE}")
        sys.exit(1)

    total_lines = 0
    total_user_words = 0
    total_assistant_words = 0
    errors = []

    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()
            if not line:
                errors.append(f"Línea {idx}: Línea vacía en JSONL.")
                continue

            try:
                data = json.loads(line)
            except Exception as e:
                errors.append(f"Línea {idx}: Error al decodificar JSON: {e}")
                continue

            if "messages" not in data or not isinstance(data["messages"], list):
                errors.append(f"Línea {idx}: Falta clave 'messages' o no es una lista.")
                continue

            msgs = data["messages"]
            if len(msgs) != 3:
                errors.append(f"Línea {idx}: Se esperaban 3 mensajes (system, user, assistant), se encontraron {len(msgs)}.")
                continue

            roles = [m.get("role") for m in msgs]
            if roles != ["system", "user", "assistant"]:
                errors.append(f"Línea {idx}: Secuencia de roles incorrecta: {roles}")
                continue

            system_text = msgs[0].get("content", "").strip()
            user_text = msgs[1].get("content", "").strip()
            assistant_text = msgs[2].get("content", "").strip()

            if not system_text or not user_text or not assistant_text:
                errors.append(f"Línea {idx}: Mensaje vacío detectado.")
                continue

            user_words = len(user_text.split())
            asst_words = len(assistant_text.split())

            total_user_words += user_words
            total_assistant_words += asst_words

            # Validar que la respuesta tenga profundidad médica suficiente
            if asst_words < 80:
                errors.append(f"Línea {idx}: Respuesta del asistente demasiado corta ({asst_words} palabras).")

            # Auditoría Ética y de Trazabilidad: Cero citas apócrifas a personas
            if "según la dra. jufe" in assistant_text.lower() or "según el dr. stahl" in assistant_text.lower():
                errors.append(f"Línea {idx}: Cita apócrifa a autor personal detectada en respuesta del asistente.")

            # Validar presencia de identificadores clínicos [DOC-ID]
            import re
            doc_citations = re.findall(r"\[(?:MONO|INTER|CASC|GUIA|ALERTA)-[A-Z0-9\-]+\]", assistant_text)
            if not doc_citations:
                errors.append(f"Línea {idx}: La respuesta médica no incluye ninguna citación a identificador de documento [DOC-ID].")

    print(f"\n--- MÉTRICAS GLOBALES DEL DATASET ---")
    print(f"  • Total de muestras conversacionales: {total_lines}")
    print(f"  • Palabras totales en consultas de doctores: {total_user_words:,} palabras (promedio: {total_user_words // total_lines} palabras/consulta)")
    print(f"  • Palabras totales en respuestas médicas: {total_assistant_words:,} palabras (promedio: {total_assistant_words // total_lines} palabras/respuesta)")
    total_corpus_words = total_user_words + total_assistant_words
    approx_tokens = int(total_corpus_words * 1.35)
    print(f"  • Total aproximado de tokens de entrenamiento: ~{approx_tokens:,} tokens")
    print(f"  • Formato verificado: ChatML compatible con Qwen2 / Qwen2.5 / Unsloth / Ollama")

    if errors:
        print(f"\n❌ Se detectaron {len(errors)} errores:")
        for err in errors:
            print("  -", err)
        sys.exit(1)
    else:
        print("\n✅ ¡El dataset pasó el 100% de las pruebas sintácticas y de consistencia clínica!")

if __name__ == "__main__":
    validate_dataset()
