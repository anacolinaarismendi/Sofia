"""
test_multiturn_rag.py
Tests unitarios y de integración para la memoria conversacional multiturno
y la guardia semántica de cobertura en el motor RAG de Sofia.
"""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.engine import (
    ClinicalRAGEngine,
    reformular_consulta_con_historial,
    extraer_farmaco_de_texto,
)


def test_extraer_farmaco_de_texto():
    assert extraer_farmaco_de_texto("¿Qué dosis de sertralina?") == "Sertralina"
    assert extraer_farmaco_de_texto("Efectos de Zoloft en ancianos") == "Sertralina"
    assert extraer_farmaco_de_texto("Uso de litio y toxicidad") in {"Carbonato de Litio", "Litio"}
    assert extraer_farmaco_de_texto("Receta de cocina con manzanas") is None


def test_reformular_consulta_sin_historial():
    pregunta = "¿Cuáles son sus efectos adversos?"
    reformulada = reformular_consulta_con_historial(pregunta, historial=None)
    assert reformulada == pregunta


def test_reformular_consulta_con_historial_resuelve_anafora():
    historial = [
        {"role": "user", "content": "¿Cuál es la dosis de inicio de la Sertralina en adultos?"},
        {"role": "assistant", "content": "La Sertralina se inicia a 50 mg/día [MONO-SERTRALINA-001]."}
    ]
    pregunta_seguimiento = "¿Y cuáles son sus interacciones principales?"
    reformulada = reformular_consulta_con_historial(pregunta_seguimiento, historial=historial)
    assert "Sertralina" in reformulada


def test_reformular_consulta_no_sobreescribe_nuevo_farmaco():
    historial = [
        {"role": "user", "content": "¿Cuál es la dosis de Sertralina?"},
        {"role": "assistant", "content": "50 a 200 mg/día."}
    ]
    nueva_pregunta = "¿Qué rango terapéutico tiene el carbonato de litio?"
    reformulada = reformular_consulta_con_historial(nueva_pregunta, historial=historial)
    assert "Sertralina" not in reformulada
    assert "litio" in reformulada.lower()


def test_consultar_multiturno_recupera_documento_correcto(monkeypatch):
    import requests

    def _sin_red(*args, **kwargs):
        raise requests.ConnectionError("sin red en test")

    monkeypatch.setattr(requests, "get", _sin_red)
    monkeypatch.setattr(requests, "post", _sin_red)

    rag = ClinicalRAGEngine()

    # Turno 1
    t1 = rag.consultar("Posología de sertralina en cardiopatía")
    assert not t1["fuera_de_alcance"]

    # Turno 2 con historial
    historial = [
        {"role": "user", "content": "Posología de sertralina en cardiopatía"},
        {"role": "assistant", "content": t1["respuesta"]}
    ]
    t2 = rag.consultar("¿Y cuál es su dosis máxima recomendada?", historial=historial)
    assert not t2["fuera_de_alcance"]
    assert any("SERTRALINA" in d.get("id", "").upper() for d, _ in t2["documentos_recuperados"])
