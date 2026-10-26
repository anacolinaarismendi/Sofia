"""
test_clinical_consistency.py
Pruebas de regresión clínica automatizadas para garantizar la consistencia,
seguridad posológica y precisión de los 27 documentos de pre_training/
y de la base de conocimiento compilada data/processed/rag_knowledge_base.json.
"""

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
DIR_PRETRAINING = ROOT / "pre_training"
KB_PATH = ROOT / "data" / "processed" / "rag_knowledge_base.json"


def test_todas_las_fichas_tienen_frontmatter_unificado():
    """Verifica que los 27 documentos Markdown tengan versión 2.1.0, fecha 2026-10-26 y revisor unificado."""
    md_files = [f for f in DIR_PRETRAINING.glob("**/*.md") if not f.name.startswith("00_")]
    assert len(md_files) == 27, f"Se esperaban 27 archivos markdown, encontrados {len(md_files)}"

    for f in md_files:
        content = f.read_text(encoding="utf-8")
        parts = content.split("---", 2)
        assert len(parts) >= 3, f"Falta frontmatter YAML en {f.name}"
        meta = {}
        for line in parts[1].strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"').strip("'")
        assert meta.get("version_contenido") == "2.1.0", f"Versión incorrecta en {f.name}"
        assert meta.get("fecha_revision") == "2026-10-26", f"Fecha revisión incorrecta en {f.name}"
        assert meta.get("revisado_por") == "Auditoría Farmacología Clínica", f"Revisor incorrecto en {f.name}"


def test_kb_sincronizada_y_no_obsoleta():
    """Verifica que rag_knowledge_base.json contenga los 59 documentos (27 pre-training + 22 CANMAT + 10 marcas) y no esté obsoleta."""
    from src.rag.engine import ClinicalRAGEngine
    rag = ClinicalRAGEngine()
    assert rag.kb_obsoleta is False, "rag_knowledge_base.json figura como obsoleta respecto a pre_training/"
    assert len(rag.docs) == 59, f"Se esperaban 59 documentos en KB, encontrados {len(rag.docs)}"


def test_clozapina_umbrales_aemps_y_fda():
    """Item 1: Clozapina debe diferenciar umbrales de suspensión AEMPS (1500) vs FDA (1000) y BEN."""
    cloz_path = DIR_PRETRAINING / "01_monografias_farmacologicas" / "clozapina.md"
    alerta_path = DIR_PRETRAINING / "05_alertas_seguridad_y_caja_negra" / "alerta_agranulocitosis_clozapina.md"

    for path in [cloz_path, alerta_path]:
        texto = path.read_text(encoding="utf-8")
        assert "1500" in texto, f"Falta umbral AEMPS 1500 en {path.name}"
        assert "1000" in texto, f"Falta umbral FDA/BEN 1000 en {path.name}"
        assert "AEMPS" in texto or "Europa" in texto, f"Falta mención europea en {path.name}"
        assert "FDA" in texto, f"Falta mención FDA en {path.name}"


def test_clozapina_carbamazepina_contraindicacion_mielosupresion():
    """Item 2: Regla de la Carbamazepina debe contraindicar explícitamente clozapina por riesgo de agranulocitosis."""
    matriz_path = DIR_PRETRAINING / "02_interacciones_farmacologicas_cyp450" / "matriz_clinica_cyp450_psicofarmacos.md"
    texto = matriz_path.read_text(encoding="utf-8")
    assert "carbamazepina" in texto.lower()
    assert "clozapina" in texto.lower()
    assert "mielosupresión" in texto or "mielosupresion" in texto
    assert "agranulocitosis" in texto


def test_clozapina_cese_tabaco_unificado():
    """Item 3: Protocolo de deshabituación tabáquica unificado (reducir 30-40% en 3-5 días con clozapinemia)."""
    cloz_path = DIR_PRETRAINING / "01_monografias_farmacologicas" / "clozapina.md"
    inter_path = DIR_PRETRAINING / "02_interacciones_farmacologicas_cyp450" / "interacciones_cyp1a2_tabaco_clozapina.md"

    for path in [cloz_path, inter_path]:
        texto = path.read_text(encoding="utf-8")
        assert "30-40%" in texto, f"Falta reducción 30-40% en {path.name}"
        assert "3-5 días" in texto or "3 a 5 días" in texto, f"Falta plazo 3-5 días en {path.name}"
        assert "clozapinemia" in texto.lower(), f"Falta clozapinemia en {path.name}"


def test_valproato_riesgo_tea_y_tdah_calibrado():
    """Item 4: Valproato riesgo de TEA ~3x (hasta 4.5-5x autismo infantil) y TDAH 1.5-2.5x."""
    valp_path = DIR_PRETRAINING / "01_monografias_farmacologicas" / "valproato.md"
    texto = valp_path.read_text(encoding="utf-8")
    assert "TDAH" in texto
    assert "1,5 a 2,5" in texto or "1.5 a 2.5" in texto or "1,5-2,5" in texto or "1.5-2.5" in texto
    assert "TEA" in texto
    assert "CYP2C9" in texto
    assert "UGT1A4" in texto


def test_litio_tiazidas_y_aines_rangos_precisos():
    """Item 5: AINEs aumentan litio 30-60% (media 40-50%); tiazidas aumentan 25-40%."""
    lit_path = DIR_PRETRAINING / "01_monografias_farmacologicas" / "litio.md"
    inter_path = DIR_PRETRAINING / "02_interacciones_farmacologicas_cyp450" / "interacciones_litio_aines_diureticos_ieca.md"

    for path in [lit_path, inter_path]:
        texto = path.read_text(encoding="utf-8")
        assert any(p in texto for p in ["30% y un 60%", "30-60%", "30 a 60%"]), f"Falta rango AINEs 30-60% en {path.name}"
        assert any(p in texto for p in ["25-40%", "25 a 40%", "25% a 40%"]), f"Falta rango tiazidas 25-40% en {path.name}"


def test_cyp2d6_porcentaje_farmacos():
    """Item 6: Cobertura enzimática de CYP2D6 corregida (~20-25%)."""
    inter_path = DIR_PRETRAINING / "02_interacciones_farmacologicas_cyp450" / "interacciones_cyp2d6_inhibidores_sustratos.md"
    matriz_path = DIR_PRETRAINING / "02_interacciones_farmacologicas_cyp450" / "matriz_clinica_cyp450_psicofarmacos.md"

    for path in [inter_path, matriz_path]:
        texto = path.read_text(encoding="utf-8")
        assert "20-25%" in texto, f"Falta ~20-25% en {path.name}"
        assert "más del 50%" not in texto, f"Quedó afirmación errónea en {path.name}"


def test_haloperidol_vias_secundarias_2d6():
    """Item 7: Haloperidol aumento moderado 20-50%, co-dependencia CYP3A4 y glucuronidación, cita MONO-FLUOXETINA-002."""
    inter_path = DIR_PRETRAINING / "02_interacciones_farmacologicas_cyp450" / "interacciones_cyp2d6_inhibidores_sustratos.md"
    texto = inter_path.read_text(encoding="utf-8")
    assert "Haloperidol" in texto
    assert "CYP3A4" in texto
    assert "glucuronidación" in texto or "glucuronidacion" in texto
    assert any(p in texto for p in ["20% a 50%", "20-50%", "20 a 50%"])
    assert "[MONO-FLUOXETINA-002]" in texto


def test_isrs_inhibicion_2d6_y_seguridad_farmaco_limpio():
    """Items 8 y 9: ISRS débil a dosis estándar; sertralina preferred, citalopram alerta QTc."""
    inter_path = DIR_PRETRAINING / "02_interacciones_farmacologicas_cyp450" / "interacciones_cyp2d6_inhibidores_sustratos.md"
    matriz_path = DIR_PRETRAINING / "02_interacciones_farmacologicas_cyp450" / "matriz_clinica_cyp450_psicofarmacos.md"

    texto_inter = inter_path.read_text(encoding="utf-8")
    assert "débil o mínima a dosis estándar" in texto_inter or "débil" in texto_inter
    assert "metoprolol" in texto_inter.lower()

    texto_matriz = matriz_path.read_text(encoding="utf-8")
    assert "QTc" in texto_matriz
    assert "Sertralina" in texto_matriz


def test_risperidona_hiperprolactinemia_comparacion():
    """Item 10: Risperidona destaca junto con paliperidona y amisulprida."""
    risp_path = DIR_PRETRAINING / "01_monografias_farmacologicas" / "risperidona.md"
    texto = risp_path.read_text(encoding="utf-8")
    assert "paliperidona" in texto.lower()
    assert "amisulprida" in texto.lower()


def test_bupropion_dosis_maxima_sr_aemps_vs_fda():
    """Item 11: Bupropión SR 300 mg/día en AEMPS vs 400 mg/día en FDA."""
    bup_path = DIR_PRETRAINING / "01_monografias_farmacologicas" / "bupropion.md"
    texto = bup_path.read_text(encoding="utf-8")
    assert "300 mg" in texto
    assert "400 mg" in texto
    assert "AEMPS" in texto
    assert "FDA" in texto


def test_lamotrigina_interrupcion_y_titulacion():
    """Item 12: Lamotrigina interrupción >5 vidas medias (>5-7 días) reinicio a 25 mg/día."""
    lam_path = DIR_PRETRAINING / "01_monografias_farmacologicas" / "lamotrigina.md"
    texto = lam_path.read_text(encoding="utf-8")
    assert "5 vidas medias" in texto
    assert "5-7 días" in texto or "5 a 7 días" in texto
    assert "25 mg" in texto


def test_cascada_haloperidol_biperideno_mermaid_rama_no():
    """Item 14: Diagrama Mermaid incluye rama de duda diagnóstica / No."""
    casc_path = DIR_PRETRAINING / "03_cascadas_prescripcion_y_desprescripcion" / "cascada_haloperidol_biperideno_parkinsonismo.md"
    texto = casc_path.read_text(encoding="utf-8")
    assert "Duda diagnóstica" in texto or "No / Duda diagnóstica" in texto
