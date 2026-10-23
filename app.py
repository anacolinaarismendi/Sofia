"""
app.py
Punto de entrada compatible para despliegues en la nube (Streamlit Community Cloud,
Hugging Face Spaces, etc.) que busquen 'app.py' por defecto en la raíz del repositorio.
Redirige y ejecuta directamente 'streamlit_app.py'.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "streamlit_app.py"
    runpy.run_path(str(target), run_name="__main__")
