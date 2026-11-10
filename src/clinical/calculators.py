"""
calculators.py
Módulo de cálculo clínico determinista para ajuste posológico y farmacocinética:
- Estimación del filtrado glomerular mediante ecuaciones estándar (CKD-EPI 2021 y Cockcroft-Gault).
- Algoritmo de dosificación y monitorización de fármacos de estrecho margen terapéutico
  con eliminación renal (Carbonato de Litio) o metabolismo hepático con fracción libre (Valproato, Lamotrigina).
"""

from typing import Dict, Any, Optional


def calcular_cockcroft_gault(
    edad_anos: int,
    peso_kg: float,
    creatinina_mg_dl: float,
    es_mujer: bool = False
) -> Dict[str, Any]:
    """
    Calcula el Aclaramiento de Creatinina (CrCl en mL/min) mediante la ecuación de Cockcroft-Gault (1976).
    Fórmula: CrCl = [ (140 - edad) * peso ] / (72 * creatinina) * [0.85 si mujer]
    """
    if edad_anos <= 0 or peso_kg <= 0 or creatinina_mg_dl <= 0:
        raise ValueError("Edad, peso y creatinina deben ser valores positivos estrictos.")

    crcl_base = ((140.0 - float(edad_anos)) * float(peso_kg)) / (72.0 * float(creatinina_mg_dl))
    factor_sexo = 0.85 if es_mujer else 1.0
    crcl_final = crcl_base * factor_sexo

    return {
        "crcl_ml_min": round(crcl_final, 1),
        "formula": "Cockcroft-Gault (1976)",
        "es_mujer": es_mujer,
        "edad_anos": edad_anos,
        "peso_kg": peso_kg,
        "creatinina_mg_dl": creatinina_mg_dl
    }


def calcular_ckd_epi_2021(
    edad_anos: int,
    creatinina_mg_dl: float,
    es_mujer: bool = False
) -> Dict[str, Any]:
    """
    Calcula el Filtrado Glomerular Estimado (eGFR en mL/min/1.73m²) mediante la ecuación
    CKD-EPI 2021 (sin factor de raza, recomendación oficial NKF/ASN/KDIGO).
    """
    if edad_anos <= 0 or creatinina_mg_dl <= 0:
        raise ValueError("Edad y creatinina deben ser valores positivos estrictos.")

    scr = float(creatinina_mg_dl)
    edad = float(edad_anos)

    if es_mujer:
        kappa = 0.7
        alpha = -0.241
        factor_sexo = 1.012
    else:
        kappa = 0.9
        alpha = -0.302
        factor_sexo = 1.0

    ratio = scr / kappa
    min_term = min(ratio, 1.0) ** alpha
    max_term = max(ratio, 1.0) ** (-1.200)
    age_term = 0.9938 ** edad

    egfr = 142.0 * min_term * max_term * age_term * factor_sexo

    if egfr >= 90.0:
        estadio = "G1"
        descripcion = "Normal o elevado"
        alerta_nivel = "SEGURO"
    elif egfr >= 60.0:
        estadio = "G2"
        descripcion = "Ligeramente disminuido"
        alerta_nivel = "PRECAUCIÓN"
    elif egfr >= 45.0:
        estadio = "G3a"
        descripcion = "Disminución leve a moderada"
        alerta_nivel = "ADVERTENCIA"
    elif egfr >= 30.0:
        estadio = "G3b"
        descripcion = "Disminución moderada a grave"
        alerta_nivel = "ALERTA"
    elif egfr >= 15.0:
        estadio = "G4"
        descripcion = "Gravemente disminuido"
        alerta_nivel = "CRÍTICO"
    else:
        estadio = "G5"
        descripcion = "Fallo renal terminal"
        alerta_nivel = "CONTRAINDICADO"

    return {
        "egfr_ml_min_173m2": round(egfr, 1),
        "estadio_kdigo": estadio,
        "descripcion_estadio": descripcion,
        "alerta_nivel": alerta_nivel,
        "formula": "CKD-EPI (2021 sin factor de raza)",
        "es_mujer": es_mujer,
        "edad_anos": edad_anos,
        "creatinina_mg_dl": creatinina_mg_dl
    }


def evaluar_ajuste_renal_farmaco(
    farmaco: str,
    egfr_o_crcl: float,
    litemia_actual_meq_l: Optional[float] = None
) -> Dict[str, Any]:
    """
    Evalúa las directrices clínicas de ajuste y monitorización para psicofármacos
    según la función renal estimada.
    """
    f_lower = farmaco.lower().strip()
    valor_filtrado = float(egfr_o_crcl)

    if any(k in f_lower for k in ["litio", "carbonato de litio", "plenur"]):
        if valor_filtrado >= 60.0:
            recomendacion = "Dosis plena habitual. Monitorizar litemia cada 3-6 meses."
            ajuste_dosis = "100% de la dosis estándar"
            nivel = "SEGURO"
            frecuencia_control = "Trimestral o semestral"
        elif valor_filtrado >= 45.0:
            recomendacion = (
                "Disminución leve del filtrado (G3a). Reducir dosis en un 20–30%. "
                "Evitar rigurosamente AINEs, diuréticos tiazídicos e IECAs/ARA-II."
            )
            ajuste_dosis = "70–80% de la dosis habitual"
            nivel = "PRECAUCIÓN"
            frecuencia_control = "Mensual tras ajuste"
        elif valor_filtrado >= 30.0:
            recomendacion = (
                "Disfunción renal moderada a grave (G3b). Reducir dosis en 50%. "
                "Vigilar clínica precoz de neurotoxicidad (temblor distal marcado, disartria, ataxia)."
            )
            ajuste_dosis = "50% de la dosis habitual"
            nivel = "ALERTA"
            frecuencia_control = "Quincenal o mensual estrecha"
        elif valor_filtrado >= 15.0:
            recomendacion = (
                "Insuficiencia renal grave (G4). Contraindicación relativa muy alta. "
                "Si la retirada no es inmediatamente viable, reducir al 25–50% con controles semanales. "
                "Valorar rotación a antipsicótico atípico."
            )
            ajuste_dosis = "25–50% de la dosis con litemia semanal"
            nivel = "CRÍTICO"
            frecuencia_control = "Semanal"
        else:
            recomendacion = (
                "Fallo renal terminal (G5, < 15 mL/min). CONTRAINDICACIÓN FORMAL ABSOLUTA. "
                "Suspender litio de inmediato por riesgo vital de intoxicación irreversible y coma."
            )
            ajuste_dosis = "SUSPENDER TRATAMIENTO"
            nivel = "CONTRAINDICADO"
            frecuencia_control = "Inmediata"

        analisis_litemia = None
        if litemia_actual_meq_l is not None:
            lit = float(litemia_actual_meq_l)
            if lit < 0.6:
                estado_lit = "Subterapéutico (< 0.6 mEq/L). Riesgo de recaída o viraje."
            elif 0.6 <= lit <= 0.8:
                estado_lit = "Óptimo en fase de mantenimiento preventivo (0.6–0.8 mEq/L)."
            elif 0.8 < lit <= 1.0:
                estado_lit = "Rango terapéutico alto / fase de manía aguda (0.8–1.0 mEq/L)."
            elif 1.0 < lit <= 1.2:
                estado_lit = "Límite superior de seguridad (1.0–1.2 mEq/L). Titular a la baja con función renal disminuida."
            elif 1.2 < lit <= 1.5:
                estado_lit = "Riesgo inminente de toxicidad (1.2–1.5 mEq/L). Hidratación forzada y ajuste urgente."
            else:
                estado_lit = "INTOXICACIÓN LÍTICA SEVERA (> 1.5 mEq/L). Urgencia médica hospitalaria; considerar hemodiálisis si > 2.0-2.5 mEq/L."

            analisis_litemia = {
                "litemia_registrada": lit,
                "estado_clinico": estado_lit
            }

        return {
            "farmaco": "Carbonato de Litio",
            "atc": "N05AN01",
            "nivel_alerta": nivel,
            "ajuste_dosis": ajuste_dosis,
            "recomendacion_clinica": recomendacion,
            "frecuencia_monitorizacion": frecuencia_control,
            "interacciones_renales_criticas": "AINEs (ibuprofeno), diuréticos tiazídicos e IECAs/ARA-II",
            "analisis_litemia": analisis_litemia
        }

    elif any(k in f_lower for k in ["valproato", "divalproato", "ácido valproico", "depakine"]):
        ajuste = "100% de dosis habitual (aclaramiento hepático predominante)."
        if valor_filtrado < 30.0:
            rec = (
                "Aclaramiento hepático sin ajuste fijo por eGFR. Sin embargo, en enfermedad renal grave "
                "la menor unión a proteínas plasmáticas (hipoalbuminemia/uremia) eleva la fracción libre "
                "de valproato. Monitorizar signos de toxicidad hepática/neurológica y dosificar según clínica."
            )
            nivel = "PRECAUCIÓN"
        else:
            rec = "Metabolismo hepático. Sin ajuste por función renal en estadios G1-G3."
            nivel = "SEGURO"

        return {
            "farmaco": "Ácido Valproico / Divalproato",
            "atc": "N03AG01",
            "nivel_alerta": nivel,
            "ajuste_dosis": ajuste,
            "recomendacion_clinica": rec,
            "frecuencia_monitorizacion": "Controles habituales (hemograma, plaquetas, transaminasas)",
            "interacciones_renales_criticas": "Ninguna interacción renal directa comparable al litio."
        }

    elif any(k in f_lower for k in ["lamotrigina", "lamictal"]):
        if valor_filtrado < 30.0:
            rec = (
                "Glucuronidación hepática (UGT1A4), pero los metabolitos glucurónidos inactivos se excretan por orina. "
                "En disfunción renal grave (G4-G5), puede ser necesario reducir la dosis de mantenimiento un 20–50%."
            )
            nivel = "PRECAUCIÓN"
            ajuste = "Considerar reducción del 20-50% en fase de mantenimiento"
        else:
            rec = "Sin ajuste por función renal en estadios G1-G3. Titulación lenta estándar para prevenir erupciones cutáneas."
            nivel = "SEGURO"
            ajuste = "Titulación estándar sin reducción"

        return {
            "farmaco": "Lamotrigina",
            "atc": "N03AX09",
            "nivel_alerta": nivel,
            "ajuste_dosis": ajuste,
            "recomendacion_clinica": rec,
            "frecuencia_monitorizacion": "Vigilancia cutánea prioritaria y niveles si hay polifarmacia (valproato)",
            "interacciones_renales_criticas": "Sin colisión renal directa; colisión crítica metabólica con Valproato (UGT1A4)."
        }

    elif any(k in f_lower for k in ["risperidona", "risperdal"]):
        if valor_filtrado < 50.0:
            rec = (
                "El aclaramiento de la fracción antipsicótica activa (risperidona + 9-OH-risperidona) "
                "se reduce aproximadamente un 60% en insuficiencia renal moderada a grave. "
                "Pauta recomendada: dosis inicial de 0.5 mg dos veces al día, titulando en incrementos de 0.5 mg."
            )
            nivel = "ADVERTENCIA"
            ajuste = "Reducir dosis inicial y máxima al 50%"
        else:
            rec = "Dosis habitual para adultos."
            nivel = "SEGURO"
            ajuste = "Dosis estándar"

        return {
            "farmaco": "Risperidona",
            "atc": "N05AX08",
            "nivel_alerta": nivel,
            "ajuste_dosis": ajuste,
            "recomendacion_clinica": rec,
            "frecuencia_monitorizacion": "Control de síntomas extrapiramidales y prolactina",
            "interacciones_renales_criticas": "Mayor riesgo de hipotensión ortostática y parkinsonismo."
        }

    else:
        return {
            "farmaco": farmaco,
            "atc": "N/D",
            "nivel_alerta": "INFORMATIVO",
            "ajuste_dosis": "Consultar monografía específica del fármaco",
            "recomendacion_clinica": (
                f"Para '{farmaco}', verificar el porcentaje de excreción renal en su ficha técnica. "
                f"Con eGFR = {valor_filtrado:.1f} mL/min, valorar el riesgo de acumulación si la eliminación urinaria es > 30%."
            ),
            "frecuencia_monitorizacion": "Según práctica asistencial habitual",
            "interacciones_renales_criticas": "No descritas interacciones renales primarias en este catálogo."
        }
