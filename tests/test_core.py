"""Tests unitarios y de regresión para el catálogo ATC y el motor clínico RAG."""

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


# --------------------------------------------------------------------------- Catálogo ATC y Normalización
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
    "reparar motor de combustion interna",
    "antibioticos para neumonia bacteriana",
])
def test_consulta_fuera_de_alcance_declara_falta_de_cobertura(rag_sin_ollama, pregunta):
    res = rag_sin_ollama.consultar(pregunta)
    assert res["fuera_de_alcance"] is True
    assert res["modo_respuesta"] == "sin_cobertura"
    assert res["documentos_recuperados"] == []
    assert res["voice_snippet"] == ""


@pytest.mark.parametrize("pregunta,prefijo", [
    ("Posologia de sertralina", ("MONO-SERTRALINA", "JUFE-MONO-SERTRALINA")),
    ("haloperidol biperideno cascada", ("CASC-HAL-BIP", "CASC")),
    ("paroxetina y bupropion CYP2D6", ("INTER-CYP2D6", "MONO-BUPROPION", "MONO-PAROXETINA", "JUFE-MONO")),
    ("posologia de clonazepam", ("JUFE-MONO-CLONAZEPAM", "MARCA-CLONAZEPAM")),
])
def test_consulta_dentro_de_alcance_recupera_documento_esperado(rag_sin_ollama, pregunta, prefijo):
    res = rag_sin_ollama.consultar(pregunta)
    assert res["fuera_de_alcance"] is False
    assert res["modo_respuesta"] == "contingencia"  # sin LLM disponible
    ids = [d["id"] for d, _ in res["documentos_recuperados"]]
    prefijos = (prefijo,) if isinstance(prefijo, str) else prefijo
    assert any(any(i.startswith(p) for p in prefijos) for i in ids), ids


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


# --------------------------------------------------------------------------- Chunking y Búsqueda Híbrida (Clase RAG)
def test_tokenizador_clinico_normaliza_acentos():
    from src.rag.chunker import tokenizar_clinico
    toks = tokenizar_clinico("Posología de Bupropión y CYP2D6 a 150mg/día")
    assert "posologia" in toks
    assert "bupropion" in toks
    assert "cyp2d6" in toks
    assert "150mg" in toks
    assert "dia" in toks


def test_chunking_estructural_enriquece_metadatos():
    from src.rag.chunker import crear_chunks_enriquecidos
    doc_muestra = [{
        "id": "MONO-TEST-001",
        "titulo": "Fármaco Test",
        "atc": "N06AB99",
        "tipo": "monografia_farmacologica",
        "contenido": "# Ficha Clínica\n\n## 1. Resumen\n> Texto resumen\n\n## 2. Farmacocinética\nMetabolismo por CYP2D6."
    }]
    chunks = crear_chunks_enriquecidos(doc_muestra, min_chars_split=50)
    assert len(chunks) == 2
    assert "[MONO-TEST-001]" in chunks[0]["texto_indexable"]
    assert "Farmacocinética" in chunks[1]["seccion"]
    assert chunks[1]["doc_padre"] == doc_muestra[0]


def test_recuperacion_hibrida_identifica_seccion_relevante(rag_sin_ollama):
    res = rag_sin_ollama.consultar("paroxetina y bupropion CYP2D6")
    assert res["fuera_de_alcance"] is False
    assert len(res["documentos_recuperados"]) > 0
    top_doc = res["documentos_recuperados"][0][0]
    # Comprobar que anota la sección relevante encontrada por el chunker
    assert "seccion_relevante" in top_doc
    assert top_doc["seccion_relevante"] != ""


def test_generacion_con_groq_mock(monkeypatch):
    """Verifica que el motor use Groq cuando GROQ_API_KEY está configurada."""
    class MockGroqResponse:
        status_code = 200
        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": "Respuesta simulada de Groq con evidencia [MONO-BUPROPION-004]."
                    }
                }]
            }

    def _mock_post(url, *args, **kwargs):
        if "api.groq.com" in url:
            return MockGroqResponse()
        raise engine_mod.requests.ConnectionError("no ollama")

    monkeypatch.setattr(engine_mod.requests, "post", _mock_post)
    rag = engine_mod.ClinicalRAGEngine(groq_api_key="gsk_mock_test_key")
    res = rag.consultar("dosis bupropion")
    assert res["generado_con_llm"] is True
    assert res["modo_respuesta"] == "llm"
    assert "Groq" in res["proveedor_llm"]
    assert "Respuesta simulada de Groq" in res["respuesta"]

