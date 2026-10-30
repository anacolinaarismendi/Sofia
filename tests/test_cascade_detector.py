"""Tests unitarios y clínicos para el Detector de Cascadas de Prescripción y Carga Anticolinérgica (ACB)."""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cascades import (
    PrescriptionCascadeDetector,
    CASCADAS_CLINICAS_EXPANDIDAS,
    TABLA_CARGA_ANTICOLINERGICA_ACB,
    PAREJAS_CASCADA_POLIFARMACIA
)


@pytest.fixture
def detector():
    return PrescriptionCascadeDetector()


# -----------------------------------------------------------------------------
# 1. Tests de Esquema y Validación Estructural
# -----------------------------------------------------------------------------
def test_schema_catalogo_expandido():
    """Verifica que todas las reglas del catálogo cumplan el esquema clínico requerido."""
    required_keys = {
        "nivel",
        "titulo",
        "mecanismo",
        "error_tipico",
        "riesgo_iatrogenico",
        "protocolo",
        "fuente",
    }
    assert len(CASCADAS_CLINICAS_EXPANDIDAS) >= 8, "Debe contener al menos los 8 fármacos principales"
    for farmaco, sintomas in CASCADAS_CLINICAS_EXPANDIDAS.items():
        assert isinstance(farmaco, str) and farmaco
        assert isinstance(sintomas, dict) and sintomas
        for sintoma, regla in sintomas.items():
            assert isinstance(sintoma, str) and sintoma
            missing = required_keys - set(regla.keys())
            assert not missing, f"Faltan claves {missing} en {farmaco} -> {sintoma}"
            assert regla["nivel"] in {"CRÍTICO", "ADVERTENCIA", "SEGURO"}
            assert isinstance(regla["protocolo"], list) and len(regla["protocolo"]) > 0


def test_schema_parejas_polifarmacia():
    """Verifica que las parejas de polifarmacia contengan todos los campos necesarios."""
    required_fields = {
        "id", "causal_alias", "cascada_alias", "farmaco_causal_nombre",
        "farmaco_cascada_nombre", "nivel", "titulo", "mecanismo",
        "riesgo_iatrogenico", "protocolo_desprescripcion", "doc_id"
    }
    assert len(PAREJAS_CASCADA_POLIFARMACIA) >= 8
    for p in PAREJAS_CASCADA_POLIFARMACIA:
        missing = required_fields - set(p.keys())
        assert not missing, f"Faltan campos {missing} en pareja {p.get('id')}"
        assert p["nivel"] in {"CRÍTICO", "ADVERTENCIA", "SEGURO"}
        assert len(p["causal_alias"]) > 0
        assert len(p["cascada_alias"]) > 0


# -----------------------------------------------------------------------------
# 2. Tests de Detección Activa de Cascadas de Prescripción (Pares en Polifarmacia)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("farmacos,expected_cascada_id", [
    (["Haloperidol 5mg", "Biperideno 2mg"], "CASC-PAIR-01"),
    (["Haldol 10 gotas", "Akineton 2mg"], "CASC-PAIR-01"),
    (["Donepezilo 10mg", "Oxibutinina 5mg"], "CASC-PAIR-02"),
    (["Aricept 10mg", "Ditropan 5mg"], "CASC-PAIR-02"),
    (["Clozapina 200mg", "Biperideno 2mg"], "CASC-PAIR-03"),
    (["Carbonato de Litio 400mg", "Hidroclorotiazida 25mg"], "CASC-PAIR-04"),
    (["Plenur", "Higrotona"], "CASC-PAIR-04"),
    (["Olanzapina 10mg", "Metformina 850mg"], "CASC-PAIR-05"),
    (["Paroxetina 20mg", "Sildenafilo 50mg"], "CASC-PAIR-06"),
    (["Litio 400mg", "Propranolol 20mg"], "CASC-PAIR-07"),
    (["Fluoxetina 20mg", "Lorazepam 1mg"], "CASC-PAIR-08"),
    (["Donepezilo 5mg", "Atenolol 50mg"], "CASC-PAIR-09"),
])
def test_detectar_cascadas_activas_pares(detector, farmacos, expected_cascada_id):
    """Verifica la detección automática de pares de fármacos en cascada."""
    cascadas = detector.detectar_cascadas_activas(farmacos)
    assert len(cascadas) >= 1, f"No se detectó cascada esperada en {farmacos}"
    ids_detectados = [c.id for c in cascadas]
    assert expected_cascada_id in ids_detectados


def test_regimen_seguro_no_genera_falsos_positivos(detector):
    """Verifica que un régimen sin cascadas no dispare alertas activas."""
    farmacos_seguros = ["Sertralina 50mg", "Levotiroxina 50mcg", "Paracetamol 1g"]
    cascadas = detector.detectar_cascadas_activas(farmacos_seguros)
    assert len(cascadas) == 0, f"Falso positivo en régimen seguro: {cascadas}"


# -----------------------------------------------------------------------------
# 3. Tests de Carga Anticolinérgica (Escala ACB)
# -----------------------------------------------------------------------------
def test_carga_anticolinergica_score_alto(detector):
    """Verifica el cálculo de puntaje ACB crítico (>= 3)."""
    # Biperideno (3) + Paroxetina (1) = 4
    acb = detector.calcular_carga_anticolinergica(["Biperideno 2mg", "Paroxetina 20mg"])
    assert acb.score_total == 4
    assert acb.nivel_alerta == "CRÍTICO"
    assert "Alto / Crítico" in acb.estratificacion_riesgo
    assert len(acb.farmacos_contribuyentes) == 2


def test_carga_anticolinergica_score_cero(detector):
    """Verifica régimen limpio sin carga anticolinérgica (Score 0)."""
    acb = detector.calcular_carga_anticolinergica(["Sertralina 100mg", "Escitalopram 10mg", "Bupropión 150mg"])
    assert acb.score_total == 0
    assert acb.nivel_alerta == "SEGURO"
    assert len(acb.farmacos_contribuyentes) == 0


# -----------------------------------------------------------------------------
# 4. Tests de Extracción NER desde Texto Libre
# -----------------------------------------------------------------------------
def test_extraer_farmacos_desde_nota_clinica(detector):
    """Verifica la extracción de fármacos a partir de un párrafo de texto libre."""
    texto = (
        "Paciente mujer de 78 años en tratamiento con Donepezilo 10 mg al día por deterioro cognitivo leve. "
        "Refiere incontinencia urinaria y en urología le han iniciado Oxibutinina 5 mg cada 12 horas."
    )
    farmacos = detector.extraer_farmacos_de_texto(texto)
    assert any("donepezil" in f.lower() for f in farmacos)
    assert any("oxibutinina" in f.lower() for f in farmacos)

    # Análisis integral desde texto
    resultado = detector.analizar_regimen([], texto_clinico=texto)
    assert len(resultado.cascadas_activas) >= 1
    assert resultado.cascadas_activas[0].id == "CASC-PAIR-02"


# -----------------------------------------------------------------------------
# 5. Tests de Evaluación Preventiva por Síntoma Emergente
# -----------------------------------------------------------------------------
def test_evaluar_farmaco_y_sintoma(detector):
    """Verifica la evaluación preventiva de un síntoma emergente."""
    regla = detector.detectar_riesgo_por_sintoma("Haloperidol", "Parkinsonismo")
    assert regla is not None
    assert regla["nivel"] == "CRÍTICO"
    assert "BIPERIDENO" in regla["titulo"]
    assert len(regla["protocolo"]) >= 3


def test_listar_farmacos_catalogo(detector):
    """Verifica que el catálogo devuelva una lista nutrida de principios activos."""
    catalogo = detector.listar_farmacos_catalogo()
    assert len(catalogo) >= 20
    assert "Haloperidol" in catalogo or any("haloperidol" in x.lower() for x in catalogo)
    assert any("biperideno" in x.lower() for x in catalogo)
    assert any("donepezilo" in x.lower() for x in catalogo)
