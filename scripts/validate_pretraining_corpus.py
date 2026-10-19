#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_pretraining_corpus.py
Script de verificación y control de calidad automatizado para el corpus pre_training/.
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PRE_TRAINING_DIR = BASE_DIR / "pre_training"

def parse_yaml_simple(raw_yaml: str):
    """Parser liviano para validar la sintaxis YAML del frontmatter sin requerir PyYAML externo."""
    data = {}
    for line in raw_yaml.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            # Parser básico de listas y strings
            if v.startswith("[") and v.endswith("]"):
                items = [item.strip().strip('"').strip("'") for item in v[1:-1].split(",") if item.strip()]
                data[k] = items
            else:
                data[k] = v.strip('"').strip("'")
    return data

def validate():
    print("=" * 60)
    print("VALIDACIÓN DEL CORPUS pre_training/")
    print("=" * 60)

    if not PRE_TRAINING_DIR.exists():
        print(f"Error: No se encontró el directorio {PRE_TRAINING_DIR}")
        sys.exit(1)

    all_md_files = sorted(list(PRE_TRAINING_DIR.glob("**/*.md")))
    print(f"Total de archivos Markdown localizados: {len(all_md_files)}")

    errors = []
    stats = {
        "tipos": set(),
        "total_palabras": 0,
        "con_voice_snippet": 0,
        "farmacos_unicos": set(),
        "archivos_por_categoria": {}
    }

    for file_path in all_md_files:
        rel = file_path.relative_to(PRE_TRAINING_DIR)
        cat = rel.parts[0] if len(rel.parts) > 1 else "raiz"
        stats["archivos_por_categoria"][cat] = stats["archivos_por_categoria"].get(cat, 0) + 1

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        words = len(content.split())
        stats["total_palabras"] += words

        # Validar Frontmatter
        if not content.startswith("---"):
            errors.append(f"[{rel}] No comienza con delimitador de frontmatter '---'")
            continue

        parts = content.split("---", 2)
        if len(parts) < 3:
            errors.append(f"[{rel}] Frontmatter incompleto o mal delimitado")
            continue

        yaml_text = parts[1]
        body = parts[2].strip()

        try:
            meta = parse_yaml_simple(yaml_text)
            if "id" not in meta:
                errors.append(f"[{rel}] Falta campo 'id' en frontmatter")
            if "tipo" not in meta:
                errors.append(f"[{rel}] Falta campo 'tipo' en frontmatter")
            else:
                stats["tipos"].add(meta["tipo"])

            # Coleccionar fármacos
            if "farmaco" in meta:
                stats["farmacos_unicos"].add(meta["farmaco"])
            if "farmacos_involucrados" in meta:
                for f in meta["farmacos_involucrados"]:
                    stats["farmacos_unicos"].add(f)

        except Exception as e:
            errors.append(f"[{rel}] Error parseando YAML: {e}")

        # Validar Voice Snippet (excepto en el manifiesto técnico)
        if rel.name != "00_MANIFIESTO_Y_ESPECIFICACION_EMBEDDINGS.md":
            if "Voice Snippet" in body or "Resumen Ejecutivo Verbalizable" in body:
                stats["con_voice_snippet"] += 1
            else:
                errors.append(f"[{rel}] No contiene sección 'Voice Snippet' o 'Resumen Ejecutivo Verbalizable'")

        # Validar longitud mínima
        if words < 200:
            errors.append(f"[{rel}] Longitud insuficiente ({words} palabras)")

    print("\n--- DISTRIBUCIÓN POR CATEGORÍAS ---")
    for cat, count in stats["archivos_por_categoria"].items():
        print(f"  • {cat}: {count} documentos")

    print("\n--- RESUMEN DE MÉTRICAS Y COBERTURA ---")
    print(f"  • Total documentos evaluados: {len(all_md_files)}")
    print(f"  • Total palabras del corpus: {stats['total_palabras']:,} palabras")
    print(f"  • Promedio de palabras por documento: {stats['total_palabras'] // len(all_md_files)} palabras")
    print(f"  • Documentos con Voice Snippet verbalizable: {stats['con_voice_snippet']} / {len(all_md_files) - 1}")
    print(f"  • Tipos de entidad cubiertos: {', '.join(sorted(stats['tipos']))}")
    print(f"  • Fármacos y entidades activas catalogadas: {len(stats['farmacos_unicos'])}")

    if errors:
        print(f"\n❌ Se detectaron {len(errors)} errores:")
        for err in errors:
            print("  -", err)
        sys.exit(1)
    else:
        print("\n✅ ¡Todos los archivos pasaron exitosamente los controles de calidad y validación clínica!")

if __name__ == "__main__":
    validate()
