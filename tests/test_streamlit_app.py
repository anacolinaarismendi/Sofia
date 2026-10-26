"""
test_streamlit_app.py
Tests de regresión e importabilidad para la aplicación interactiva Streamlit (streamlit_app.py).
Previene errores de sintaxis, variables o tipos no importados (como NameError: Optional).
"""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_import_streamlit_app():
    """Verifica que streamlit_app se importe sin NameError, ImportError o SyntaxError."""
    import streamlit_app  # noqa: F401


def test_resolver_credenciales_ollama():
    """Verifica la resolución defensiva de credenciales de Ollama."""
    from streamlit_app import resolver_credenciales_ollama

    key, host, model = resolver_credenciales_ollama()
    assert isinstance(key, str)
    assert isinstance(host, str)
    assert isinstance(model, str)


def test_load_datasets():
    """Verifica que los datasets clínicos requeridos por la app se carguen correctamente."""
    from streamlit_app import load_datasets

    data = load_datasets()
    assert isinstance(data, dict)
    assert "canmat" in data
    assert "jufe" in data
    assert "stahl" in data
    assert not data["canmat"].empty


def test_cascadas_clinicas_schema():
    """Verifica la estructura y consistencia del catálogo de cascadas de prescripción."""
    from streamlit_app import CASCADAS_CLINICAS

    required_keys = {
        "nivel",
        "titulo",
        "mecanismo",
        "error_tipico",
        "riesgo_iatrogenico",
        "protocolo",
        "fuente",
    }
    for farmaco, sintomas in CASCADAS_CLINICAS.items():
        assert isinstance(farmaco, str) and farmaco
        assert isinstance(sintomas, dict) and sintomas
        for sintoma, regla in sintomas.items():
            assert isinstance(sintoma, str) and sintoma
            missing = required_keys - set(regla.keys())
            assert not missing, f"Faltan claves {missing} en {farmaco} -> {sintoma}"
            assert isinstance(regla["protocolo"], list) and len(regla["protocolo"]) > 0
            assert regla["nivel"] in {"CRÍTICO", "ADVERTENCIA", "SEGURO"}


def test_md_a_html():
    """Verifica la conversión de negritas y saltos de párrafo para tarjetas HTML."""
    from streamlit_app import _md_a_html

    resultado = _md_a_html("**Texto en negrita**\n\nSegundo párrafo.")
    assert "<b>Texto en negrita</b>" in resultado
    assert "<p>" in resultado and "</p>" in resultado
