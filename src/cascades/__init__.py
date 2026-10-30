"""Módulo de Detección de Cascadas de Prescripción y Farmacovigilancia del proyecto Sofia."""

from src.cascades.detector import (
    PrescriptionCascadeDetector,
    CascadaActiva,
    RiesgoLatente,
    CargaAnticolinergica,
    ResultadoAnalisisRegimen
)
from src.cascades.rules import (
    TABLA_CARGA_ANTICOLINERGICA_ACB,
    CASCADAS_CLINICAS_EXPANDIDAS,
    PAREJAS_CASCADA_POLIFARMACIA
)

__all__ = [
    "PrescriptionCascadeDetector",
    "CascadaActiva",
    "RiesgoLatente",
    "CargaAnticolinergica",
    "ResultadoAnalisisRegimen",
    "TABLA_CARGA_ANTICOLINERGICA_ACB",
    "CASCADAS_CLINICAS_EXPANDIDAS",
    "PAREJAS_CASCADA_POLIFARMACIA"
]
