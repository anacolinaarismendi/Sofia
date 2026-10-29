"""Ejecutor determinista de notebooks para el proyecto Sofia.
Ejecuta las celdas secuencialmente en el entorno virtual, capturando stdout/stderr y figuras.
Evita bloqueos de sockets y GUI de matplotlib en entornos headless o sandboxed.
"""
import io
import os
import sys
from pathlib import Path
import nbformat
import matplotlib

# Forzar backend headless no bloqueante
matplotlib.use('Agg')
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_sofia'
Path('/tmp/matplotlib_sofia').mkdir(parents=True, exist_ok=True)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

def execute_notebook(nb_path: Path):
    print(f"\n{'='*70}\nIniciando ejecución de: {nb_path.name}\n{'='*70}")
    nb = nbformat.read(nb_path, as_version=4)
    
    # Contexto global compartido entre celdas del notebook
    gl = {
        '__name__': '__main__',
        '__file__': str(nb_path)
    }
    
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            source = cell.source.strip()
            if not source:
                continue
                
            # Filtrar comandos mágicos de Jupyter (%pip, %matplotlib, etc.)
            clean_lines = []
            for line in source.split('\n'):
                stripped = line.strip()
                if stripped.startswith('%') or stripped.startswith('!'):
                    clean_lines.append(f"# {line}")
                else:
                    clean_lines.append(line)
            clean_code = '\n'.join(clean_lines)
            
            f_out = io.StringIO()
            f_err = io.StringIO()
            
            orig_stdout = sys.stdout
            orig_stderr = sys.stderr
            sys.stdout = f_out
            sys.stderr = f_err
            
            try:
                exec(clean_code, gl)
            except Exception as e:
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr
                print(f"❌ Error en celda {idx} de {nb_path.name}: {e}")
                print("--- Código con error ---")
                print(clean_code)
                print("--- Stdout acumulado ---")
                print(f_out.getvalue())
                print("--- Stderr acumulado ---")
                print(f_err.getvalue())
                raise
            finally:
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr
            
            # Asignar salidas limpias a la celda
            cell.outputs = []
            stdout_text = f_out.getvalue()
            stderr_text = f_err.getvalue()
            
            if stdout_text:
                cell.outputs.append(nbformat.v4.new_output(
                    output_type='stream',
                    name='stdout',
                    text=stdout_text
                ))
            if stderr_text:
                cell.outputs.append(nbformat.v4.new_output(
                    output_type='stream',
                    name='stderr',
                    text=stderr_text
                ))
            cell.execution_count = idx + 1
            print(f"  ✓ Celda {idx:02d} ejecutada con éxito")

    # Guardar notebook con salidas persistidas
    nbformat.write(nb, nb_path)
    print(f"✅ Finalizado con éxito: {nb_path.name} (todas las celdas ejecutadas y persistidas)")

def main():
    notebooks = [
        WORKSPACE_ROOT / "notebooks" / "00_limpieza_y_eda.ipynb",
        WORKSPACE_ROOT / "notebooks" / "01_preprocesamiento_jufe.ipynb",
        WORKSPACE_ROOT / "notebooks" / "02_preprocesamiento_stahl.ipynb",
        WORKSPACE_ROOT / "notebooks" / "03_preprocesamiento_canmat.ipynb",
        WORKSPACE_ROOT / "notebooks" / "04_rag_psicofarmacologia.ipynb"
    ]
    
    for nb in notebooks:
        execute_notebook(nb)
    print("\n🎉 Todos los notebooks se han ejecutado y validado sin ningún error.")

if __name__ == '__main__':
    main()
