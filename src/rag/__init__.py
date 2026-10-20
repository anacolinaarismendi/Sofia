"""Módulo RAG clínico de Sofia con almacén vectorial y prompt template especializado."""
from .engine import (
    ClinicalRAGEngine,
    ClinicalVectorStore,
    cargar_corpus_pretraining,
    SOFIA_CLINICAL_PROMPT,
    SOFIA_SOUL_PROMPT
)

__all__ = [
    "ClinicalRAGEngine",
    "ClinicalVectorStore",
    "cargar_corpus_pretraining",
    "SOFIA_CLINICAL_PROMPT",
    "SOFIA_SOUL_PROMPT"
]

