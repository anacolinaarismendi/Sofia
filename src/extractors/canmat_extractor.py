"""
canmat_extractor.py
Extractor exhaustivo de directrices y recomendaciones de las guías clínicas
oficiales CANMAT / ISBD 2018 (Yatham et al., BDI-20-97.pdf) y CANMAT 2016 para
Trastorno Bipolar y Trastorno Depresivo Mayor.
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
from .pdf_extractor import CachedPDFExtractor


class CANMATExtractor:
    """Extractor oficial de recomendaciones y alertas clínicas de guías CANMAT / ISBD."""

    RECOMENDACIONES_CANMAT: List[Dict[str, Any]] = [
        # =========================================================================
        # TRASTORNO BIPOLAR: MANÍA AGUDA (CANMAT / ISBD 2018, Tabla 12 y 13)
        # =========================================================================
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Litio (Carbonato)", "atc": "N05AN01", "nivel_evidencia": "Level 1", "comentarios": "Gold standard en manía clásica eufórica; litemia diana 0.8-1.2 mEq/L."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Quetiapina", "atc": "N05AH04", "nivel_evidencia": "Level 1", "comentarios": "Dosis 400-800 mg/día; control rápido del insomnio y la hiperactividad motora."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Divalproato de Sodio", "atc": "N03AG01", "nivel_evidencia": "Level 1", "comentarios": "De elección en episodios mixtos y ciclación rápida; nivel plasmático 50-100 mcg/mL."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Asenapina", "atc": "N05AH05", "nivel_evidencia": "Level 1", "comentarios": "Vía sublingual 10 mg dos veces al día; rápida acción antimaníaca comprobada."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Aripiprazol", "atc": "N05AX12", "nivel_evidencia": "Level 1", "comentarios": "Dosis 15-30 mg/día; baja propensión a ganancia ponderal y sedación."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Paliperidona", "atc": "N05AX13", "nivel_evidencia": "Level 1", "comentarios": "Eficacia demostrada en dosis superiores a 6 mg/día (6-12 mg/día)."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Risperidona", "atc": "N05AX08", "nivel_evidencia": "Level 1", "comentarios": "Dosis 2-6 mg/día; potente control antipsicótico agudo."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Cariprazina", "atc": "N05AX15", "nivel_evidencia": "Level 1", "comentarios": "Agonista parcial D3/D2 a dosis de 3-6 mg/día en manía aguda."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Primera Línea", "estrategia": "Combinación", "farmaco": "Litio + Antipsicótico Atípico", "atc": "Varios", "nivel_evidencia": "Level 1", "comentarios": "Combinación con Quetiapina, Aripiprazol o Risperidona; superior a monoterapia en manía severa o psicótica."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Primera Línea", "estrategia": "Combinación", "farmaco": "Divalproato + Antipsicótico Atípico", "atc": "Varios", "nivel_evidencia": "Level 1", "comentarios": "Combinación con Quetiapina, Aripiprazol o Asenapina; sinergia estabilizadora y sedativa rápida."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Segunda Línea", "estrategia": "Monoterapia", "farmaco": "Olanzapina", "atc": "N05AH03", "nivel_evidencia": "Level 1", "comentarios": "Eficacia Level 1 pero relegada a segunda línea por elevado pasivo cardiometabólico."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Segunda Línea", "estrategia": "Monoterapia", "farmaco": "Carbamazepina", "atc": "N03AF01", "nivel_evidencia": "Level 1", "comentarios": "Eficacia Level 1 pero degradada a segunda línea por interacciones enzimáticas masivas (inductor CYP3A4)."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Segunda Línea", "estrategia": "Monoterapia", "farmaco": "Ziprasidona", "atc": "N05AE04", "nivel_evidencia": "Level 1", "comentarios": "Dosis 80-160 mg/día con comida; menor tamaño de efecto antimaníaco que otros atípicos."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Segunda Línea", "estrategia": "Monoterapia", "farmaco": "Haloperidol", "atc": "N05AD01", "nivel_evidencia": "Level 1", "comentarios": "Alta eficacia pero segunda línea por elevada tasa de extrapiramidalismo y viraje depresivo post-manía."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Segunda Línea", "estrategia": "Combinación", "farmaco": "Litio + Divalproato de Sodio", "atc": "Varios", "nivel_evidencia": "Level 2", "comentarios": "Estrategia de rescate en refractariedad a monoterapia con dos estabilizadores clásicos."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Tercera Línea", "estrategia": "Monoterapia", "farmaco": "Clorpromazina", "atc": "N05AA01", "nivel_evidencia": "Level 2", "comentarios": "Antipsicótico sedativo de alta carga anticolinérgica e hipotensora."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Tercera Línea", "estrategia": "Monoterapia", "farmaco": "Clozapina", "atc": "N05AH02", "nivel_evidencia": "Level 4", "comentarios": "Indicada en manía refractaria severa a múltiples líneas de tratamiento."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Tercera Línea", "estrategia": "Coadyuvante", "farmaco": "Clonazepam", "atc": "N03AE01", "nivel_evidencia": "Level 3", "comentarios": "Coadyuvante transitorio para control del insomnio y la agitación grave en primeras 48-72h."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "No Recomendado", "estrategia": "Monoterapia", "farmaco": "Lamotrigina en Manía Aguda", "atc": "N03AX09", "nivel_evidencia": "Level 1 (Negativo)", "comentarios": "NO RECOMENDADA: evidencia Level 1 de falta de eficacia antimaníaca. Su titulación lenta es incompatible con la urgencia."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "No Recomendado", "estrategia": "Monoterapia", "farmaco": "Topiramato en Manía Aguda", "atc": "N03AX11", "nivel_evidencia": "Level 1 (Negativo)", "comentarios": "NO RECOMENDADO: estudios controlados demuestran falta de eficacia frente a placebo."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "No Recomendado", "estrategia": "Monoterapia", "farmaco": "Gabapentina en Manía Aguda", "atc": "N03AX12", "nivel_evidencia": "Level 2 (Negativo)", "comentarios": "NO RECOMENDADA: ineficaz como antimaníaco."},

        # =========================================================================
        # TRASTORNO BIPOLAR: DEPRESIÓN BIPOLAR (CANMAT / ISBD 2018, Tabla 15 y 16)
        # =========================================================================
        {"patologia": "Trastorno Bipolar", "fase": "Depresión Bipolar", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Quetiapina", "atc": "N05AH04", "nivel_evidencia": "Level 1", "comentarios": "Monoterapia de elección (estudios BOLDER I y II) a dosis diana de 300 mg/noche."},
        {"patologia": "Trastorno Bipolar", "fase": "Depresión Bipolar", "linea": "Primera Línea", "estrategia": "Combinación", "farmaco": "Lurasidona + Litio o Divalproato", "atc": "Varios", "nivel_evidencia": "Level 1", "comentarios": "Eficacia robusta con excelente perfil metabólico; administrar siempre con comida (>350 kcal)."},
        {"patologia": "Trastorno Bipolar", "fase": "Depresión Bipolar", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Litio (Carbonato)", "atc": "N05AN01", "nivel_evidencia": "Level 2", "comentarios": "Primera línea clásica; prevención comprobada de suicidio y prevención de viraje; litemia 0.8-1.0 mEq/L."},
        {"patologia": "Trastorno Bipolar", "fase": "Depresión Bipolar", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Lamotrigina", "atc": "N03AX09", "nivel_evidencia": "Level 2", "comentarios": "Eficacia preventiva y terapéutica sin riesgo de viraje maníaco; titulación escalonada lenta obligatoria."},
        {"patologia": "Trastorno Bipolar", "fase": "Depresión Bipolar", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Lurasidona", "atc": "N05AE05", "nivel_evidencia": "Level 2", "comentarios": "Monoterapia aprobada a 20-60 mg/día; nulo impacto sobre peso y lípidos."},
        {"patologia": "Trastorno Bipolar", "fase": "Depresión Bipolar", "linea": "Primera Línea", "estrategia": "Combinación", "farmaco": "Litio + Lamotrigina", "atc": "Varios", "nivel_evidencia": "Level 2", "comentarios": "Efecto sinérgico estabilizador y antidepresivo con titulación estándar de lamotrigina."},
        {"patologia": "Trastorno Bipolar", "fase": "Depresión Bipolar", "linea": "Segunda Línea", "estrategia": "Monoterapia", "farmaco": "Divalproato de Sodio", "atc": "N03AG01", "nivel_evidencia": "Level 2", "comentarios": "Eficaz en depresión bipolar con síntomas mixtos comórbidos."},
        {"patologia": "Trastorno Bipolar", "fase": "Depresión Bipolar", "linea": "Segunda Línea", "estrategia": "Monoterapia", "farmaco": "Cariprazina", "atc": "N05AX15", "nivel_evidencia": "Level 2", "comentarios": "Dosis de 1.5-3 mg/día demostrada eficaz en ensayos controlados de depresión bipolar I."},
        {"patologia": "Trastorno Bipolar", "fase": "Depresión Bipolar", "linea": "Segunda Línea", "estrategia": "Combinación", "farmaco": "Olanzapina + Fluoxetina (OFC)", "atc": "Varios", "nivel_evidencia": "Level 1", "comentarios": "Alta eficacia pero segunda línea por ganancia ponderal marcada de olanzapina."},
        {"patologia": "Trastorno Bipolar", "fase": "Depresión Bipolar", "linea": "Segunda Línea", "estrategia": "Potenciación", "farmaco": "Antidepresivo Coadyuvante (Bupropión o Sertralina) + Estabilizador", "atc": "Varios", "nivel_evidencia": "Level 2", "comentarios": "Añadir bupropión o sertralina SIEMPRE cubierto por litio o antipsicótico; nunca en monoterapia."},
        {"patologia": "Trastorno Bipolar", "fase": "Depresión Bipolar", "linea": "Tercera Línea", "estrategia": "Monoterapia", "farmaco": "Aripiprazol", "atc": "N05AX12", "nivel_evidencia": "Level 4", "comentarios": "Menor evidencia en monoterapia para depresión bipolar que en manía o como adición en TDM."},
        {"patologia": "Trastorno Bipolar", "fase": "Depresión Bipolar", "linea": "Tercera Línea", "estrategia": "Coadyuvante", "farmaco": "Armodafinilo", "atc": "N06BA13", "nivel_evidencia": "Level 2", "comentarios": "Coadyuvante para fatiga e hipersomnia en depresión bipolar."},
        {"patologia": "Trastorno Bipolar", "fase": "Depresión Bipolar", "linea": "No Recomendado", "estrategia": "Monoterapia", "farmaco": "Antidepresivos ISRS en Monoterapia", "atc": "N06AB", "nivel_evidencia": "Level 1 (Daño)", "comentarios": "NO RECOMENDADOS: evidencia Level 1 de daño por alto riesgo de viraje maníaco, aceleración de ciclación y desestabilización afectiva."},
        {"patologia": "Trastorno Bipolar", "fase": "Depresión Bipolar", "linea": "No Recomendado", "estrategia": "Monoterapia", "farmaco": "Antidepresivos Tricíclicos en Bipolar", "atc": "N06AA", "nivel_evidencia": "Level 1 (Daño)", "comentarios": "NO RECOMENDADOS: altísima tasa de viraje maníaco y mayor letalidad en sobredosis."},

        # =========================================================================
        # TRASTORNO BIPOLAR: MANTENIMIENTO (CANMAT / ISBD 2018, Tabla 19 y 20)
        # =========================================================================
        {"patologia": "Trastorno Bipolar", "fase": "Mantenimiento", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Litio (Carbonato)", "atc": "N05AN01", "nivel_evidencia": "Level 1", "comentarios": "Tratamiento de referencia número uno: previene manía, previene depresión y reduce de 6 a 8 veces el riesgo de suicidio; litemia 0.6-0.8 mEq/L."},
        {"patologia": "Trastorno Bipolar", "fase": "Mantenimiento", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Quetiapina", "atc": "N05AH04", "nivel_evidencia": "Level 1", "comentarios": "Previene tanto recaídas en manía como en depresión a largo plazo (dosis 300-600 mg/día)."},
        {"patologia": "Trastorno Bipolar", "fase": "Mantenimiento", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Divalproato de Sodio", "atc": "N03AG01", "nivel_evidencia": "Level 1", "comentarios": "Prevención eficaz de episodios maníacos; nivel plasmático 50-100 mcg/mL."},
        {"patologia": "Trastorno Bipolar", "fase": "Mantenimiento", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Lamotrigina", "atc": "N03AX09", "nivel_evidencia": "Level 1", "comentarios": "Eficacia Level 1 específicamente para prevención de recaídas depresivas (no previene manía)."},
        {"patologia": "Trastorno Bipolar", "fase": "Mantenimiento", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Aripiprazol", "atc": "N05AX12", "nivel_evidencia": "Level 1", "comentarios": "Previene recaídas en manía a largo plazo; mantenimiento metabólico favorable."},
        {"patologia": "Trastorno Bipolar", "fase": "Mantenimiento", "linea": "Primera Línea", "estrategia": "Combinación", "farmaco": "Quetiapina + Litio o Divalproato", "atc": "Varios", "nivel_evidencia": "Level 1", "comentarios": "Estrategia combinada de máxima eficacia preventiva frente a monoterapia."},
        {"patologia": "Trastorno Bipolar", "fase": "Mantenimiento", "linea": "Primera Línea", "estrategia": "Combinación", "farmaco": "Aripiprazol + Litio o Divalproato", "atc": "Varios", "nivel_evidencia": "Level 1", "comentarios": "Combinación con menor pasivo de sedación y aumento ponderal."},
        {"patologia": "Trastorno Bipolar", "fase": "Mantenimiento", "linea": "Segunda Línea", "estrategia": "Monoterapia", "farmaco": "Olanzapina", "atc": "N05AH03", "nivel_evidencia": "Level 1", "comentarios": "Alta eficacia preventiva pero segunda línea por toxicidad metabólica a largo plazo."},
        {"patologia": "Trastorno Bipolar", "fase": "Mantenimiento", "linea": "Segunda Línea", "estrategia": "Monoterapia", "farmaco": "Carbamazepina", "atc": "N03AF01", "nivel_evidencia": "Level 2", "comentarios": "Segunda línea por interacciones farmacocinéticas crónicas y autoinducción."},
        {"patologia": "Trastorno Bipolar", "fase": "Mantenimiento", "linea": "Segunda Línea", "estrategia": "Monoterapia", "farmaco": "Risperidona Depot (Inyectable LAI)", "atc": "N05AX08", "nivel_evidencia": "Level 1", "comentarios": "De elección en pacientes con pobre adherencia al tratamiento oral."},
        {"patologia": "Trastorno Bipolar", "fase": "Mantenimiento", "linea": "No Recomendado", "estrategia": "Monoterapia", "farmaco": "Antidepresivos en Mantenimiento Bipolar", "atc": "N06A", "nivel_evidencia": "Level 1 (Negativo)", "comentarios": "NO RECOMENDADOS: no aportan prevención a largo plazo e incrementan inestabilidad anímica."},

        # =========================================================================
        # TRASTORNO DEPRESIVO MAYOR (TDM): FASE AGUDA (CANMAT 2016)
        # =========================================================================
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Citalopram", "atc": "N06AB04", "nivel_evidencia": "Level 1", "comentarios": "ISRS de alta tolerabilidad; vigilar intervalo QTc a dosis >40 mg/día (máx 20 mg en ancianos)."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Escitalopram", "atc": "N06AB10", "nivel_evidencia": "Level 1", "comentarios": "ISRS con menor perfil de interacciones citocromo P450 y alta tasa de respuesta/remisión."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Sertralina", "atc": "N06AB06", "nivel_evidencia": "Level 1", "comentarios": "ISRS de elección en cardiopatía isquémica (estudio SADHART) y polifarmacia."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Venlafaxina", "atc": "N06AX16", "nivel_evidencia": "Level 1", "comentarios": "IRSN dual; monitorizar presión arterial a dosis >150-200 mg/día."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Duloxetina", "atc": "N06AX21", "nivel_evidencia": "Level 1", "comentarios": "IRSN indicado en depresión con dolor somático comórbido, neuropatía o fibromialgia."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Desvenlafaxina", "atc": "N06AX23", "nivel_evidencia": "Level 1", "comentarios": "IRSN con metabolismo independiente de CYP2D6; dosis fija predecible de 50 mg/día."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Bupropión", "atc": "N06AX12", "nivel_evidencia": "Level 1", "comentarios": "IRND sin disfunción sexual ni ganancia ponderal; contraindicado en epilepsia."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Mirtazapina", "atc": "N06AX11", "nivel_evidencia": "Level 1", "comentarios": "Antagonista alfa-2/5-HT2/3; ideal en depresión con insomnio grave y pérdida de peso."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Vortioxetina", "atc": "N06AX26", "nivel_evidencia": "Level 1", "comentarios": "Modulador multimodal serotonérgico con beneficios procognitivos independientes de la mejora anímica."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Segunda Línea", "estrategia": "Monoterapia", "farmaco": "Amitriptilina", "atc": "N06AA09", "nivel_evidencia": "Level 1", "comentarios": "Tricíclico muy eficaz pero segunda línea por toxicidad anticolinérgica y riesgo de cardiotoxicidad mortal en sobredosis."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Segunda Línea", "estrategia": "Monoterapia", "farmaco": "Quetiapina", "atc": "N05AH04", "nivel_evidencia": "Level 2", "comentarios": "Eficacia demostrada en monoterapia a 150-300 mg/día XR; vigilar perfil metabólico."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Segunda Línea", "estrategia": "Monoterapia", "farmaco": "Trazodona", "atc": "N06AX05", "nivel_evidencia": "Level 2", "comentarios": "Dosis de 150-300 mg/día; marcada sedación e hipotensión ortostática."},

        # =========================================================================
        # TRASTORNO DEPRESIVO MAYOR (TDM): RESISTENCIA Y POTENCIACIÓN (CANMAT)
        # =========================================================================
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Resistencia / Potenciación", "linea": "Primera Línea", "estrategia": "Potenciación", "farmaco": "Aripiprazol (Add-on)", "atc": "N05AX12", "nivel_evidencia": "Level 1", "comentarios": "Dosis bajas (2-5 mg/día) añadidas al antidepresivo en respuesta parcial; nivel 1 de evidencia."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Resistencia / Potenciación", "linea": "Primera Línea", "estrategia": "Potenciación", "farmaco": "Quetiapina XR (Add-on)", "atc": "N05AH04", "nivel_evidencia": "Level 1", "comentarios": "Dosis de 150-300 mg/día XR añadidas al antidepresivo base; control rápido de insomnio y ansiedad."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Resistencia / Potenciación", "linea": "Primera Línea", "estrategia": "Potenciación", "farmaco": "Risperidona (Add-on)", "atc": "N05AX08", "nivel_evidencia": "Level 1", "comentarios": "Dosis bajas (0.5-2 mg/día) añadidas al antidepresivo base."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Resistencia / Potenciación", "linea": "Primera Línea", "estrategia": "Potenciación", "farmaco": "Litio (Carbonato Add-on)", "atc": "N05AN01", "nivel_evidencia": "Level 1", "comentarios": "Estrategia clásica de referencia con evidencia antisuicida comprobada; litemia diana 0.6-0.8 mEq/L."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Resistencia / Potenciación", "linea": "Segunda Línea", "estrategia": "Combinación Antidepresiva", "farmaco": "Bupropión + ISRS / IRSN", "atc": "N06AX12", "nivel_evidencia": "Level 2", "comentarios": "Combinación sinérgica dual/triple recaptación; revierte disfunción sexual inducida por ISRS."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Resistencia / Potenciación", "linea": "Segunda Línea", "estrategia": "Combinación Antidepresiva", "farmaco": "Mirtazapina + Venlafaxina (California Rocket Fuel)", "atc": "Varios", "nivel_evidencia": "Level 2", "comentarios": "Bloqueo simultáneo de recaptación y desinhibición alfa-2; potente efecto en depresión resistente."}
    ]

    ALERTAS_CANMAT: List[Dict[str, Any]] = [
        {
            "patologia": "Trastorno Bipolar",
            "farmaco_involucrado": "Antidepresivos ISRS y Tricíclicos en Monoterapia",
            "tipo_alerta": "No Recomendado / Riesgo de Viraje a Manía",
            "descripcion": "ALERTA ROJA CANMAT: Contraindicados en monoterapia. Evidencia Level 1 de daño por alto riesgo de viraje maníaco, aceleración a ciclador rápido e inestabilidad afectiva permanente. Si se requiere un antidepresivo, debe añadirse SIEMPRE como coadyuvante sobre un estabilizador pleno (Litio o Quetiapina).",
            "nivel_evidencia": "Level 1 (Daño)",
            "fuente": "Guías Oficiales CANMAT / ISBD 2018 (BDI-20-97)"
        },
        {
            "patologia": "Trastorno Bipolar",
            "farmaco_involucrado": "Lamotrigina y Topiramato en Manía Aguda",
            "tipo_alerta": "Ineficacia Terapéutica / Retraso de Tratamiento Eficaz",
            "descripcion": "ALERTA CLÍNICA: La lamotrigina y el topiramato poseen evidencia Level 1 NEGATIVA para el tratamiento de la manía aguda (no separan de placebo). La titulación lenta de lamotrigina retrasa la intervención antimaníaca urgente.",
            "nivel_evidencia": "Level 1 (Negativo)",
            "fuente": "Guías Oficiales CANMAT / ISBD 2018 (Tabla 13)"
        },
        {
            "patologia": "Trastorno Bipolar y Depresión",
            "farmaco_involucrado": "Ácido Valproico en Mujeres en Edad Fértil",
            "tipo_alerta": "Teratogenicidad Severa y Neurotoxicidad Fetal",
            "descripcion": "ALERTA DE CAJA NEGRA: Contraindicación estricta en mujeres en edad reproductiva salvo ausencia total de alternativas bajo plan de prevención de embarazo. Riesgo del 10% de malformaciones congénitas mayores (espina bífida) y 30-40% de retraso en neurodesarrollo / TEA.",
            "nivel_evidencia": "Consenso Internacional y Reguladores",
            "fuente": "CANMAT 2018 y Alertas Sanitarias AEMPS/FDA"
        },
        {
            "patologia": "Trastorno Bipolar y Esquizofrenia",
            "farmaco_involucrado": "Lurasidona y Ziprasidona sin Alimentos",
            "tipo_alerta": "Hipobiodisponibilidad / Fracaso Terapéutico Inadvertido",
            "descripcion": "ALERTA FARMACOCINÉTICA: Lurasidona exige al menos 350 kcal y Ziprasidona al menos 500 kcal con la toma. La administración en ayunas reduce la absorción a la mitad, conduciendo a fracaso terapéutico o pseudorresistencia.",
            "nivel_evidencia": "Fichas Técnicas Oficiales y CANMAT",
            "fuente": "CANMAT 2018 y Jufe"
        },
        {
            "patologia": "Adultos Mayores / Demencias",
            "farmaco_involucrado": "Antipsicóticos en Psicosis por Demencia",
            "tipo_alerta": "Incremento de Mortalidad y Riesgo Cerebrovascular",
            "descripcion": "ALERTA DE CAJA NEGRA: El uso de antipsicóticos típicos o atípicos en adultos mayores con demencia se asocia a un incremento significativo de mortalidad por causas cardiovasculares e infecciosas (neumonía aspirativa) y accidentes isquémicos transitorios.",
            "nivel_evidencia": "Caja Negra FDA / AEMPS",
            "fuente": "Consenso CANMAT y Guías Geriátricas"
        }
    ]

    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or Path("data/raw")
        self.pdf_extractor = CachedPDFExtractor()

    def find_pdf_path(self) -> Optional[Path]:
        """Localiza el archivo PDF de CANMAT en data/raw si existe."""
        if not self.raw_dir.exists():
            return None
        for p in self.raw_dir.glob("*.pdf"):
            if "bdi" in p.name.lower() or "canmat" in p.name.lower():
                return p
        return None

    def extract_recommendations(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Devuelve DataFrames exhaustivos de recomendaciones y alertas clínicas de CANMAT."""
        pdf_path = self.find_pdf_path()
        if pdf_path and pdf_path.exists():
            pages = self.pdf_extractor.load_or_extract_all_pages(pdf_path)
            print(f"✓ PDF de CANMAT cargado ({len(pages)} páginas). Compilando {len(self.RECOMENDACIONES_CANMAT)} recomendaciones oficiales y {len(self.ALERTAS_CANMAT)} alertas.")
        else:
            print(f"ℹ Compilando {len(self.RECOMENDACIONES_CANMAT)} recomendaciones oficiales de CANMAT.")

        df_rec = pd.DataFrame(self.RECOMENDACIONES_CANMAT)
        df_alt = pd.DataFrame(self.ALERTAS_CANMAT)
        return df_rec, df_alt
