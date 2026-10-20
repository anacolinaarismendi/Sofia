#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_execute_notebooks.py
Ejecuta de principio a fin los Jupyter Notebooks usando nbclient/nbconvert
para verificar que el 100% de las celdas se ejecutan sin errores y guarda los
outputs actualizados (comprobando que no existan rutas absolutas locales).
"""

import os
import sys
from pathlib import Path

# Configurar directorios locales para que Jupyter no intente acceder a Library/ fuera del sandbox
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
JUPYTER_LOCAL = WORKSPACE_ROOT / ".jupyter_tmp"
JUPYTER_LOCAL.mkdir(parents=True, exist_ok=True)

os.environ["JUPYTER_DATA_DIR"] = str(JUPYTER_LOCAL / "data")
os.environ["JUPYTER_RUNTIME_DIR"] = str(JUPYTER_LOCAL / "runtime")
os.environ["JUPYTER_CONFIG_DIR"] = str(JUPYTER_LOCAL / "config")
os.environ["IPYTHONDIR"] = str(JUPYTER_LOCAL / "ipython")

import nbformat
from nbclient import NotebookClient


NOTEBOOKS = [
    "notebooks/01_preprocesamiento_jufe.ipynb",
    "notebooks/02_preprocesamiento_stahl.ipynb",
    "notebooks/03_preprocesamiento_canmat.ipynb",
]


def execute_notebook(nb_path_str: str) -> bool:
    nb_path = WORKSPACE_ROOT / nb_path_str
    print(f"\n--- Ejecutando {nb_path.name} ---")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    try:
        client.execute()
        print(f"✓ {nb_path.name} ejecutado con éxito (todas las celdas completadas).")

        # Guardar notebook con los outputs actualizados
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        # Comprobar si quedó alguna ruta de usuario
        with open(nb_path, "r", encoding="utf-8") as f:
            content = f.read()

        if "/Users/" in content:
            print(f"⚠️ ADVERTENCIA: {nb_path.name} contiene '/Users/' en outputs.")
            # Buscar dónde aparece
            for line in content.splitlines():
                if "/Users/" in line:
                    print(f"   Línea con /Users/: {line[:120]}")
            return False
        else:
            print(f"✓ CERO rutas locales de usuario en {nb_path.name}.")
            return True

    except Exception as e:
        print(f"❌ ERROR ejecutando {nb_path.name}: {e}")
        return False


def main():
    success_count = 0
    for nb in NOTEBOOKS:
        if execute_notebook(nb):
            success_count += 1

    print(f"\nResumen: {success_count}/{len(NOTEBOOKS)} notebooks ejecutados limpiamente.")
    if success_count < len(NOTEBOOKS):
        sys.exit(1)


if __name__ == "__main__":
    main()
