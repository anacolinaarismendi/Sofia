"""Tests mínimos de regresión para los errores corregidos en la auditoría (E-01..E-11)."""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.enrichers.atc_enricher import ATCEnricher  # noqa: E402
from src.rag import engine as engine_mod  # noqa: E402


# --------------------------------------------------------------------------- ATC (E-07)
@pytest.mark.parametrize("entrada", ["", "  ", "a", "o", "ina", None])
def test_atc_entradas_invalidas_no_devuelven_farmaco(entrada):
    assert ATCEnricher.get_info(entrada) is None


@pytest.mark.parametrize("entrada,codigo", [
    ("Sertralina", "N06AB06"),
    ("  sertralina ", "N06AB06"),
    ("Bupropión", "N06AX12"),
    ("Zoloft", "N06AB06"),
    ("Carbonato de litio", "N05AN01"),
])
def test_atc_coincidencias_validas(entrada, codigo):
    assert ATCEnricher.get_info(entrada)["atc_codigo"] == codigo


# --------------------------------------------------------------------------- Motor RAG
@pytest.fixture
def rag_sin_ollama(monkeypatch):
    """Motor RAG aislado de la red: sin Ollama, recuperación TF-IDF."""
    def _sin_red(*args, **kwargs):
        raise engine_mod.requests.ConnectionError("sin red (test)")

    monkeypatch.setattr(engine_mod.requests, "get", _sin_red)
    monkeypatch.setattr(engine_mod.requests, "post", _sin_red)
    return engine_mod.ClinicalRAGEngine()


@pytest.mark.parametrize("pregunta", [
    "como hacer una tarta de manzana",
    "clonazepam dosis",
    "antibioticos para neumonia",
])
def test_consulta_fuera_de_alcance_declara_falta_de_cobertura(rag_sin_ollama, pregunta):
    res = rag_sin_ollama.consultar(pregunta)
    assert res["fuera_de_alcance"] is True
    assert res["modo_respuesta"] == "sin_cobertura"
    assert res["documentos_recuperados"] == []
    assert res["voice_snippet"] == ""


@pytest.mark.parametrize("pregunta,prefijo", [
    ("Posologia de sertralina", "MONO-SERTRALINA"),
    ("haloperidol biperideno cascada", "CASC-HAL-BIP"),
    ("paroxetina y bupropion CYP2D6", "INTER-CYP2D6"),
])
def test_consulta_dentro_de_alcance_recupera_documento_esperado(rag_sin_ollama, pregunta, prefijo):
    res = rag_sin_ollama.consultar(pregunta)
    assert res["fuera_de_alcance"] is False
    assert res["modo_respuesta"] == "contingencia"  # sin LLM disponible
    ids = [d["id"] for d, _ in res["documentos_recuperados"]]
    assert any(i.startswith(prefijo) for i in ids), ids


def test_ids_de_ejemplo_del_prompt_existen_en_el_corpus():
    kb = json.loads((ROOT / "data" / "processed" / "rag_knowledge_base.json").read_text(encoding="utf-8"))
    ids_reales = {d["id"] for d in kb}
    ejemplos = re.findall(r"\[((?:MONO|INTER|CASC|GUIA|ALERTA)-[A-Z0-9\-]+)\]", engine_mod.SOFIA_CLINICAL_PROMPT)
    assert ejemplos
    assert set(ejemplos) <= ids_reales, set(ejemplos) - ids_reales


def test_citas_fuera_del_contexto_se_marcan(monkeypatch):
    def _sin_red_get(*args, **kwargs):
        raise engine_mod.requests.ConnectionError("sin red (test)")

    class _RespFalsa:
        status_code = 200

        def json(self):
            return {
                "response": "1. Juicio... [MONO-SERTRALINA-001] y además [MONO-INVENTADO-999].",
                "done_reason": "stop",
            }

    monkeypatch.setattr(engine_mod.requests, "get", _sin_red_get)
    rag = engine_mod.ClinicalRAGEngine()
    monkeypatch.setattr(engine_mod.requests, "post", lambda *a, **k: _RespFalsa())

    res = rag.consultar("Posologia de sertralina")
    assert res["modo_respuesta"] == "llm"
    assert res["citas_no_recuperadas"] == ["MONO-INVENTADO-999"]
    assert "Aviso de trazabilidad" in res["respuesta"]
