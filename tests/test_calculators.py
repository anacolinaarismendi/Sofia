"""
test_calculators.py
Tests unitarios para el módulo clínico de cálculo determinista (CKD-EPI, Cockcroft-Gault y ajuste de Litio).
"""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical.calculators import (
    calcular_cockcroft_gault,
    calcular_ckd_epi_2021,
    evaluar_ajuste_renal_farmaco,
)


def test_cockcroft_gault_valores_validos():
    res = calcular_cockcroft_gault(edad_anos=60, peso_kg=72, creatinina_mg_dl=1.0, es_mujer=False)
    # (140-60)*72 / (72*1.0) = 80.0
    assert res["crcl_ml_min"] == 80.0

    res_mujer = calcular_cockcroft_gault(edad_anos=60, peso_kg=72, creatinina_mg_dl=1.0, es_mujer=True)
    assert res_mujer["crcl_ml_min"] == round(80.0 * 0.85, 1)


def test_cockcroft_gault_valores_invalidos():
    with pytest.raises(ValueError):
        calcular_cockcroft_gault(edad_anos=-5, peso_kg=70, creatinina_mg_dl=1.0)
    with pytest.raises(ValueError):
        calcular_cockcroft_gault(edad_anos=40, peso_kg=0, creatinina_mg_dl=1.0)


def test_ckd_epi_2021_estadios():
    # Adulto joven normal
    joven = calcular_ckd_epi_2021(edad_anos=25, creatinina_mg_dl=0.8, es_mujer=False)
    assert joven["estadio_kdigo"] in {"G1", "G2"}
    assert joven["egfr_ml_min_173m2"] > 90

    # Anciano con ERC moderada-grave
    anciano = calcular_ckd_epi_2021(edad_anos=80, creatinina_mg_dl=2.5, es_mujer=True)
    assert anciano["estadio_kdigo"] in {"G4", "G5"}
    assert anciano["alerta_nivel"] in {"CRÍTICO", "CONTRAINDICADO"}


def test_evaluar_ajuste_litio_segun_filtrado():
    # Normal
    normal = evaluar_ajuste_renal_farmaco("Litio", 85.0)
    assert normal["nivel_alerta"] == "SEGURO"
    assert "100%" in normal["ajuste_dosis"]

    # ERC G3b (30-45)
    moderado = evaluar_ajuste_renal_farmaco("Carbonato de Litio", 35.0)
    assert moderado["nivel_alerta"] == "ALERTA"
    assert "50%" in moderado["ajuste_dosis"]

    # Fallo terminal (< 15)
    terminal = evaluar_ajuste_renal_farmaco("Plenur", 10.0)
    assert terminal["nivel_alerta"] == "CONTRAINDICADO"
    assert "SUSPENDER" in terminal["ajuste_dosis"]


def test_evaluar_ajuste_litio_con_litemia():
    res = evaluar_ajuste_renal_farmaco("Litio", 50.0, litemia_actual_meq_l=1.8)
    assert res["analisis_litemia"] is not None
    assert "INTOXICACIÓN" in res["analisis_litemia"]["estado_clinico"]
