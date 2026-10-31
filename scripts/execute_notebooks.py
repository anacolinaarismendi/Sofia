"""Ejecutor determinista de notebooks para el proyecto Sofia.
Ejecuta las celdas secuencialmente en el entorno virtual, capturando stdout/stderr
y persistiendo todas las figuras generadas como imágenes PNG en base64 en los outputs.
"""
import io
import os
import sys
import base64
from pathlib import Path
import nbformat
import matplotlib

# Forzar backend headless y directorio temporal aislado
matplotlib.use('Agg')
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_sofia'
Path('/tmp/matplotlib_sofia').mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
os.chdir(WORKSPACE_ROOT)

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
            
            cell_outputs = []
            
            # Interceptar plt.show() para capturar figuras como PNG base64
            def custom_show(*args, **kwargs):
                for fignum in plt.get_fignums():
                    fig = plt.figure(fignum)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
                    buf.seek(0)
                    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                    cell_outputs.append(nbformat.v4.new_output(
                        output_type='display_data',
                        data={'image/png': img_b64, 'text/plain': f'<Figure size {fig.get_size_inches()} with {len(fig.axes)} Axes>'}
                    ))
                    plt.close(fig)

            orig_show = plt.show
            plt.show = custom_show
            gl['plt'] = plt
            
            try:
                exec(clean_code, gl)
                # Capturar cualquier figura que haya quedado abierta sin show()
                for fignum in plt.get_fignums():
                    fig = plt.figure(fignum)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
                    buf.seek(0)
                    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                    cell_outputs.append(nbformat.v4.new_output(
                        output_type='display_data',
                        data={'image/png': img_b64, 'text/plain': f'<Figure size {fig.get_size_inches()} with {len(fig.axes)} Axes>'}
                    ))
                    plt.close(fig)
            except Exception as e:
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr
                plt.show = orig_show
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
                plt.show = orig_show
            
            # Limpiar stderr filtrando avisos no bloqueantes de backend Agg
            raw_stderr = f_err.getvalue()
            filtered_stderr_lines = [
                l for l in raw_stderr.split('\n')
                if 'FigureCanvasAgg is non-interactive' not in l
                and 'Matplotlib is building the font cache' not in l
                and 'is not a writable directory' not in l
            ]
            clean_stderr = '\n'.join(filtered_stderr_lines).strip()
            
            # Asignar salidas a la celda
            stdout_text = f_out.getvalue()
            
            outputs_final = []
            if stdout_text:
                outputs_final.append(nbformat.v4.new_output(
                    output_type='stream',
                    name='stdout',
                    text=stdout_text
                ))
            if clean_stderr:
                outputs_final.append(nbformat.v4.new_output(
                    output_type='stream',
                    name='stderr',
                    text=clean_stderr + '\n'
                ))
            # Agregar imágenes PNG
            outputs_final.extend(cell_outputs)
            
            cell.outputs = outputs_final
            cell.execution_count = idx + 1
            num_figs = len(cell_outputs)
            fig_str = f" ({num_figs} figuras PNG guardadas)" if num_figs > 0 else ""
            print(f"  ✓ Celda {idx:02d} ejecutada con éxito{fig_str}")

    # Guardar notebook con salidas persistidas
    nbformat.write(nb, nb_path)
    print(f"✅ Finalizado con éxito: {nb_path.name} (todas las celdas ejecutadas y persistidas)")

def main():
    notebooks = [
        WORKSPACE_ROOT / "notebooks" / "01_limpieza_y_preprocesamiento.ipynb",
        WORKSPACE_ROOT / "notebooks" / "02_analisis_exploratorio_eda.ipynb",
        WORKSPACE_ROOT / "notebooks" / "03_rag_psicofarmacologia.ipynb"
    ]
    
    for nb in notebooks:
        execute_notebook(nb)
    print("\n🎉 Todos los notebooks se han ejecutado y validado sin ningún error.")

if __name__ == '__main__':
    main()
