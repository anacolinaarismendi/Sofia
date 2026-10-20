#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_notebook_in_process.py
Ejecuta notebooks celda por celda en un proceso Python limpio,
capturando salidas y errores para verificar ejecutabilidad al 100%
sin depender de sockets ZMQ bloqueados por sandbox.
"""

import os
import sys
import io
import json
import traceback
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
MPL_TMP = WORKSPACE_ROOT / ".matplotlib_tmp"
MPL_TMP.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_TMP)


def run_notebook(nb_path_str: str) -> bool:
    nb_path = Path(nb_path_str)
    print(f"\n==========================================")
    print(f"▶ Ejecutando: {nb_path.name}")
    print(f"==========================================")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Namespace limpio para la ejecución del notebook
    nb_globals = {
        "__name__": "__main__",
        "__file__": str(nb_path.resolve())
    }

    execution_count = 1
    total_code_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    executed_cells = 0

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        executed_cells += 1
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        print(f"  [Celda {executed_cells}/{total_code_cells}] Ejecutando...")

        # Capturar stdout
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_output = io.StringIO()
        redirected_error = io.StringIO()
        sys.stdout = redirected_output
        sys.stderr = redirected_error

        try:
            # Filtrar comandos mágicos de jupyter si quedara alguno
            clean_lines = []
            for line in source.splitlines():
                if line.strip().startswith("%") or line.strip().startswith("!"):
                    continue
                clean_lines.append(line)
            clean_code = "\n".join(clean_lines)

            exec(clean_code, nb_globals)
            out_text = redirected_output.getvalue()
            err_text = redirected_error.getvalue()

            # Guardar outputs limpios en la celda
            cell["execution_count"] = execution_count
            outputs = []
            clean_out_lines = [l for l in out_text.splitlines(keepends=True) if "/Users/" not in l]
            clean_err_lines = [l for l in err_text.splitlines(keepends=True) if "/Users/" not in l and "matplotlib" not in l.lower()]
            if clean_out_lines:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": clean_out_lines
                })
            if clean_err_lines:
                outputs.append({
                    "name": "stderr",
                    "output_type": "stream",
                    "text": clean_err_lines
                })
            cell["outputs"] = outputs
            execution_count += 1

        except Exception as e:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            print(f"❌ ERROR en celda #{idx} de {nb_path.name}: {e}")
            traceback.print_exc()
            return False
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    # Guardar el notebook con sus nuevos outputs
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    # Validar que NO contenga ninguna ruta local de usuario
    with open(nb_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "/Users/" in content:
        print(f"⚠️ ADVERTENCIA: {nb_path.name} contiene '/Users/' en outputs guardados.")
        for line in content.splitlines():
            if "/Users/" in line:
                print(f"   Línea con ruta: {line[:120]}")
        return False

    print(f"✓ ÉXITO TOTAL: {total_code_cells} celdas ejecutadas sin errores.")
    print(f"✓ Cero rutas de usuario encontradas en {nb_path.name}.")
    return True


if __name__ == "__main__":
    notebooks = [
        "notebooks/00_limpieza_y_eda.ipynb",
        "notebooks/01_preprocesamiento_jufe.ipynb",
        "notebooks/02_preprocesamiento_stahl.ipynb",
        "notebooks/03_preprocesamiento_canmat.ipynb",
        "notebooks/04_rag_psicofarmacologia.ipynb"
    ]

    all_ok = True
    for nb in notebooks:
        ok = run_notebook(nb)
        if not ok:
            all_ok = False
            break

    if not all_ok:
        sys.exit(1)
    print("\n✓ ¡Todos los cuadernos de preprocesamiento se ejecutaron exitosamente al 100%!")
