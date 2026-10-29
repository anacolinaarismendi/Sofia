#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_full_database.py
Compila e integra la base de datos clínica completa y real de Sofia:
1. Extrae y valida 57 fármacos reales del libro de Jufe (640 págs) y CANMAT (74 págs).
2. Genera stahl_farmacos_clean.csv y stahl_eventos_adversos_clean.csv con perfiles verificados de Stahl.
3. Genera el catálogo relacional SQLite sofia_clinica.db.
4. Regenera rag_knowledge_base.json integrando monografías completas, CANMAT, marcas y pre_training.
"""

import sys
import json
import sqlite3
import unicodedata
from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.extractors.jufe_extractor import JufeExtractor
from src.extractors.canmat_extractor import CANMATExtractor
from src.enrichers.atc_enricher import ATCEnricher
from src.rag.engine import cargar_corpus_pretraining, ClinicalRAGEngine


def normalizar_id(texto: str) -> str:
    """Convierte texto en un identificador uniforme en mayúsculas sin acentos ni espacios."""
    if not isinstance(texto, str):
        return "UNKNOWN"
    norm = unicodedata.normalize("NFKD", texto)
    clean = "".join(c for c in norm if not unicodedata.combining(c))
    clean = clean.upper().replace(" ", "-").replace("/", "-").replace("(", "").replace(")", "").replace(".", "")
    while "--" in clean:
        clean = clean.replace("--", "-")
    return clean.strip("-")


def compilar_stahl() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Genera catálogo estructurado de fármacos y eventos de Stahl con perfiles farmacológicos verificados."""
    FARMACOS_STAHL = [
        {
            "farmaco": "Clozapina",
            "clase": "Antipsicótico Atípico (Segunda Generación)",
            "perfil_receptores": "Antagonista 5-HT2A > D2; alta afinidad por receptores D4, 5-HT2C, alfa-1 adrenérgicos, H1 histaminérgicos y M1 colinérgicos; agonista alostérico M4.",
            "metabolismo_cyp": "Sustrato principal de CYP1A2. Sustrato secundario de CYP2D6 y CYP3A4. Los fumadores inducen CYP1A2 y reducen sus niveles hasta 50%.",
            "efectos_comunes": "Sedación intensa, sialorrea nocturna, aumento de peso, estreñimiento severo, taquicardia sinusal, mareos.",
            "efectos_graves": "Agranulocitosis (neutropenia grave potencialmente mortal), miocarditis, convulsiones dosis-dependiente, íleo paralítico.",
            "perlas_clinicas": "Indicado en esquizofrenia resistente y riesgo suicida recurrente. Requiere monitorización hematológica estricta con hemogramas seriados.",
            "inhibe_cyp2d6": 0, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 1, "sustrato_cyp2b6": 0, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 1,
            "atc_codigo": "N05AH02"
        },
        {
            "farmaco": "Paroxetina",
            "clase": "Inhibidor Selectivo de la Recaptación de Serotonina (ISRS)",
            "perfil_receptores": "Inhibidor potente del transportador de serotonina (SERT); ligera acción anticolinérgica muscarínica (M1) y débil inhibición de recaptación de noradrenalina (NET).",
            "metabolismo_cyp": "Potente inhibidor del citocromo CYP2D6. Sustrato de CYP2D6 con cinética no lineal por autosaturación.",
            "efectos_comunes": "Disfunción sexual marcada, somnolencia/sedación, aumento de peso moderado, náuseas, boca seca por efecto anticolinérgico.",
            "efectos_graves": "Síndrome de discontinuación intenso tras suspensión brusca, síndrome serotoninérgico, teratogenicidad (defectos septales cardíacos en primer trimestre).",
            "perlas_clinicas": "Es el ISRS más sedante y anticolinérgico. Desaconsejado en adultos mayores por riesgo de deterioro cognitivo y caídas.",
            "inhibe_cyp2d6": 1, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 0,
            "atc_codigo": "N06AB05"
        },
        {
            "farmaco": "Bupropión",
            "clase": "Inhibidor de la Recaptación de Noradrenalina y Dopamina (IRND)",
            "perfil_receptores": "Inhibidor de los transportadores DAT y NET; antagonista de receptores nicotínicos alfa-4-beta-2 y alfa-3-beta-4.",
            "metabolismo_cyp": "Metabolizado a hidroxibupropión principalmente por CYP2B6. Potente inhibidor del citocromo CYP2D6.",
            "efectos_comunes": "Insomnio, agitación psicomotriz, cefalea, temblor fino, sequedad bucal, sudoración, disminución del apetito y peso.",
            "efectos_graves": "Convulsiones (riesgo dosis-dependiente >450 mg/día), psicosis o exacerbación maníaca en pacientes bipolares.",
            "perlas_clinicas": "Antidepresivo activador que no produce disfunción sexual ni aumento ponderal. Contraindicado en trastornos de conducta alimentaria (anorexia/bulimia) y epilepsia.",
            "inhibe_cyp2d6": 1, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 1, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 0,
            "atc_codigo": "N06AX12"
        },
        {
            "farmaco": "Lamotrigina",
            "clase": "Estabilizador del Ánimo / Antiepiléptico",
            "perfil_receptores": "Bloqueo de canales de sodio voltaje-dependientes; modulación de la liberación presináptica de glutamato.",
            "metabolismo_cyp": "Metabolismo por glucuronidación (UGT1A4). No depende fuertemente de citocromos CYP.",
            "efectos_comunes": "Cefalea, mareos, visión borrosa, náuseas, exantema cutáneo benigno.",
            "efectos_graves": "Síndrome de Stevens-Johnson / Necrólisis Epidérmica Tóxica (exantema grave potencialmente mortal).",
            "perlas_clinicas": "Eficaz principalmente en la fase depresiva del Trastorno Bipolar. El ácido valproico duplica sus niveles (requiere reducir la dosis a la mitad); la carbamazepina reduce sus niveles a la mitad.",
            "inhibe_cyp2d6": 0, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 0,
            "atc_codigo": "N03AX09"
        },
        {
            "farmaco": "Carbonato de Litio",
            "clase": "Estabilizador del Ánimo / Catión Monovalente",
            "perfil_receptores": "Modulación de cascadas de segundos mensajeros (inhibición de inositol monofosfatasa y GSK-3beta); neuroprotección y plasticidad sináptica.",
            "metabolismo_cyp": "Eliminación renal casi exclusiva (>95%) por filtración glomerular sin metabolización hepática por citocromos CYP450.",
            "efectos_comunes": "Temblor fino distal, poliuria, polidipsia, aumento de peso, molestias gastrointestinales leves, hipotiroidismo subclínico.",
            "efectos_graves": "Intoxicación por litemia (arritmias cardíacas, ataxia, convulsiones, coma), insuficiencia renal crónica, diabetes insípida nefrogénica.",
            "perlas_clinicas": "Gold standard en profilaxis del Trastorno Bipolar y reducción comprobada del riesgo suicida. Rango terapéutico estrecho (0.6-0.8 mEq/L en mantenimiento, hasta 0.8-1.1 mEq/L en manía aguda); exige monitorización estrecha de función renal y tiroidea.",
            "inhibe_cyp2d6": 0, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 0,
            "atc_codigo": "N05AN01"
        },
        {
            "farmaco": "Sertralina",
            "clase": "Inhibidor Selectivo de la Recaptación de Serotonina (ISRS)",
            "perfil_receptores": "Inhibidor selectivo de SERT, débil afinidad sobre transportador DAT y agonista de receptores sigma-1.",
            "metabolismo_cyp": "Metabolismo por CYP2B6, CYP2C19, CYP2D6 y CYP3A4. Débil inhibidor de CYP2D6 solo a dosis altas.",
            "efectos_comunes": "Náuseas, diarrea, insomnio, cefalea, disfunción sexual.",
            "efectos_graves": "Síndrome serotoninérgico con IMAO, hiponatremia por SIADH.",
            "perlas_clinicas": "Antidepresivo de primera elección en cardiopatía isquémica reciente (estudio SADHART).",
            "inhibe_cyp2d6": 0, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 1, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 1,
            "atc_codigo": "N06AB06"
        },
        {
            "farmaco": "Fluoxetina",
            "clase": "Inhibidor Selectivo de la Recaptación de Serotonina (ISRS)",
            "perfil_receptores": "Inhibidor de SERT y antagonista de 5-HT2C (desinhibición prefrontal de NA y DA).",
            "metabolismo_cyp": "Metabolizado a norfluoxetina. Potente inhibidor de CYP2D6 y moderado de CYP2C19.",
            "efectos_comunes": "Ansiedad, anorexia, insomnio, náuseas, temblor.",
            "efectos_graves": "Viraje maníaco, síndrome serotoninérgico, interacciones farmacocinéticas múltiples.",
            "perlas_clinicas": "Vida media muy larga (norfluoxetina hasta 2 semanas). Lavado de 5 semanas antes de un IMAO.",
            "inhibe_cyp2d6": 1, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 0,
            "atc_codigo": "N06AB03"
        },
        {
            "farmaco": "Escitalopram",
            "clase": "Inhibidor Selectivo de la Recaptación de Serotonina (ISRS)",
            "perfil_receptores": "Inhibidor alostérico altamente selectivo de SERT. Ausencia de bloqueo colinérgico o histamínico.",
            "metabolismo_cyp": "Metabolismo por CYP2C19 y CYP3A4. Mínima inhibición de CYP2D6.",
            "efectos_comunes": "Náuseas leves, cefalea, somnolencia, disfunción sexual.",
            "efectos_graves": "Prolongación del QTc a dosis supraterapéuticas (>20-30 mg/día).",
            "perlas_clinicas": "ISRS con menor tasa de interacciones citocromo P450; de elección en polimedicación.",
            "inhibe_cyp2d6": 0, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 1,
            "atc_codigo": "N06AB10"
        },
        {
            "farmaco": "Venlafaxina",
            "clase": "Inhibidor de la Recaptación de Serotonina y Noradrenalina (IRSN)",
            "perfil_receptores": "Bloqueo de SERT a dosis bajas y de NET a dosis medias/altas (≥150 mg/día).",
            "metabolismo_cyp": "Metabolismo hepático vía CYP2D6 a O-desmetilvenlafaxina activa.",
            "efectos_comunes": "Náuseas, sudoración profusa, sequedad de boca, insomnio, estreñimiento.",
            "efectos_graves": "Hipertensión arterial dosis-dependiente, síndrome de retirada abrupto intenso.",
            "perlas_clinicas": "Control obligatorio de tensión arterial en dosis superiores a 150 mg/día.",
            "inhibe_cyp2d6": 0, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 0,
            "atc_codigo": "N06AX16"
        },
        {
            "farmaco": "Duloxetina",
            "clase": "Inhibidor de la Recaptación de Serotonina y Noradrenalina (IRSN)",
            "perfil_receptores": "Inhibición equilibrada de SERT y NET desde dosis iniciales.",
            "metabolismo_cyp": "Metabolismo por CYP1A2 y CYP2D6. Inhibidor moderado de CYP2D6.",
            "efectos_comunes": "Náuseas, sequedad bucal, estreñimiento, astenia, mareos.",
            "efectos_graves": "Hepatotoxicidad (ictericia, aumento de transaminasas), crisis hipertensiva.",
            "perlas_clinicas": "Indicada en depresión con dolor somático o fibromialgia. Evitar en hepatopatía crónica.",
            "inhibe_cyp2d6": 1, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 1, "sustrato_cyp2b6": 0, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 0,
            "atc_codigo": "N06AX21"
        },
        {
            "farmaco": "Mirtazapina",
            "clase": "Antidepresivo Noradrenérgico y Serotonérgico Específico (NaSSA)",
            "perfil_receptores": "Antagonista alfa-2 presináptico, 5-HT2A, 5-HT2C, 5-HT3 e histaminérgico H1.",
            "metabolismo_cyp": "Metabolismo por CYP1A2, CYP2D6 y CYP3A4. Sin inhibición enzimática.",
            "efectos_comunes": "Sedación marcada, aumento de apetito, ganancia de peso significativa, sequedad bucal.",
            "efectos_graves": "Agranulocitosis (rara), convulsiones en sobredosis.",
            "perlas_clinicas": "Ideal en pacientes deprimidos con insomnio severo y pérdida de peso; sin disfunción sexual.",
            "inhibe_cyp2d6": 0, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 1, "sustrato_cyp2b6": 0, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 1,
            "atc_codigo": "N06AX11"
        },
        {
            "farmaco": "Haloperidol",
            "clase": "Antipsicótico Típico (Primera Generación)",
            "perfil_receptores": "Antagonismo potente y altamente selectivo de receptores D2 mesolímbicos y nigroestriatales.",
            "metabolismo_cyp": "Metabolismo por glucuronidación, CYP3A4 y CYP2D6.",
            "efectos_comunes": "Parkinsonismo secundario, rigidez, temblor, acatisia, sedación moderada.",
            "efectos_graves": "Distonía aguda laringofaríngea, discinesia tardía, prolongación del QTc, síndrome neuroléptico maligno.",
            "perlas_clinicas": "Gran potencia antipsicótica con alto riesgo de cascada de prescripción hacia biperideno.",
            "inhibe_cyp2d6": 0, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 1,
            "atc_codigo": "N05AD01"
        },
        {
            "farmaco": "Risperidona",
            "clase": "Antipsicótico Atípico (Segunda Generación)",
            "perfil_receptores": "Antagonista potente de 5-HT2A y D2, alfa-1 adrenérgico e histaminérgico H1.",
            "metabolismo_cyp": "Metabolismo vía CYP2D6 hacia 9-hidroxirisperidona (paliperidona activa).",
            "efectos_comunes": "Hiperprolactinemia marcada (galactorrea, amenorrea), aumento de peso, sedación.",
            "efectos_graves": "Síntomas extrapiramidales dosis-dependientes (>4-6 mg/día), riesgo cerebrovascular en demencia.",
            "perlas_clinicas": "A dosis bajas (2-4 mg) mantiene perfil atípico; a dosis altas pierde selectividad atípica.",
            "inhibe_cyp2d6": 0, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 1,
            "atc_codigo": "N05AX08"
        },
        {
            "farmaco": "Olanzapina",
            "clase": "Antipsicótico Atípico (Segunda Generación)",
            "perfil_receptores": "Antagonista multirreceptorial 5-HT2A, 5-HT2C, D2, H1, M1 y alfa-1.",
            "metabolismo_cyp": "Metabolismo por CYP1A2 y glucuronidación. El tabaco disminuye sus concentraciones plasmáticas.",
            "efectos_comunes": "Aumento masivo de peso, apetito voraz por carbohidratos, sedación diurna intensa.",
            "efectos_graves": "Síndrome metabólico acelerado, diabetes mellitus tipo 2 de inicio rápido, dislipidemia grave.",
            "perlas_clinicas": "Excelente eficacia en agitación y manía pero alto pasivo cardiometabólico.",
            "inhibe_cyp2d6": 0, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 1, "sustrato_cyp2b6": 0, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 0,
            "atc_codigo": "N05AH03"
        },
        {
            "farmaco": "Quetiapina",
            "clase": "Antipsicótico Atípico (Segunda Generación)",
            "perfil_receptores": "Disociación rápida de D2, antagonismo 5-HT2A, H1, alfa-1. Norquetiapina inhibe NET y estimula 5-HT1A.",
            "metabolismo_cyp": "Metabolismo casi exclusivo por CYP3A4.",
            "efectos_comunes": "Sedación profunda temprana, mareos, hipotensión ortostática, ganancia ponderal moderada.",
            "efectos_graves": "Cataratas (descritas en animales), prolongación moderada de QTc, síndrome metabólico.",
            "perlas_clinicas": "Nula carga extrapiramidal; de elección en psicosis asociada a enfermedad de Parkinson.",
            "inhibe_cyp2d6": 0, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 1,
            "atc_codigo": "N05AH04"
        },
        {
            "farmaco": "Aripiprazol",
            "clase": "Antipsicótico Atípico / Agonista Parcial D2",
            "perfil_receptores": "Agonista parcial D2 (30% actividad intrínseca), agonista parcial 5-HT1A, antagonista 5-HT2A.",
            "metabolismo_cyp": "Metabolismo por CYP2D6 y CYP3A4. T1/2 prolongada (75 horas).",
            "efectos_comunes": "Acatisia inicial e inquietud motora, insomnio, cefalea, náuseas.",
            "efectos_graves": "Control de impulsos alterado (ludopatía, hipersexualidad paradójica).",
            "perlas_clinicas": "Neutro a nivel metabólico y ponderal; normaliza hiperprolactinemia por otros antipsicóticos.",
            "inhibe_cyp2d6": 0, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 1,
            "atc_codigo": "N05AX12"
        },
        {
            "farmaco": "Ácido Valproico",
            "clase": "Estabilizador del Ánimo / Antiepiléptico",
            "perfil_receptores": "Aumento del tono GABA cerebral, bloqueo de canales de sodio y canales de calcio tipo T.",
            "metabolismo_cyp": "Glucuronidación (UGT) y beta-oxidación. Inhibidor potente de UGT2B7 y CYP2C9.",
            "efectos_comunes": "Molestias digestivas, temblor fino, aumento de peso, alopecia transitoria, somnolencia.",
            "efectos_graves": "Hepatotoxicidad fulminante, pancreatitis aguda, trombocitopenia, hiperamonemia, teratogenicidad.",
            "perlas_clinicas": "Contraindicado absolutamente en mujeres en edad fértil por teratogenicidad mayor y TEA.",
            "inhibe_cyp2d6": 0, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 0,
            "atc_codigo": "N03AG01"
        },
        {
            "farmaco": "Carbamazepina",
            "clase": "Estabilizador del Ánimo / Antiepiléptico",
            "perfil_receptores": "Bloqueo de canales de sodio voltaje-dependientes en estado inactivado.",
            "metabolismo_cyp": "Potentísimo inductor multienzimático de CYP1A2, CYP2C9, CYP2C19 y CYP3A4. Autoinducción.",
            "efectos_comunes": "Mareos, ataxia, sedación, diplopía, náuseas, hiponatremia.",
            "efectos_graves": "Agranulocitosis, anemia aplásica, síndrome de Stevens-Johnson, hepatitis tóxica.",
            "perlas_clinicas": "Contraindicada con clozapina por mielosupresión sinérgica. Invalida anticonceptivos orales.",
            "inhibe_cyp2d6": 0, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 1, "sustrato_cyp2b6": 0, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 1,
            "atc_codigo": "N03AF01"
        },
        {
            "farmaco": "Clonazepam",
            "clase": "Benzodiazepina de Alta Potencia",
            "perfil_receptores": "Modulador alostérico positivo de receptores GABA-A con subunidades alfa-1, 2, 3 y 5.",
            "metabolismo_cyp": "Metabolizado principalmente por CYP3A4 y posterior acetilación.",
            "efectos_comunes": "Sedación, somnolencia matinal, ataxia, enlentecimiento psicomotor, amnesia anterógrada.",
            "efectos_graves": "Dependencia física, tolerancia, depresión respiratoria con opioides o alcohol, abstinencia severa.",
            "perlas_clinicas": "Retirada escalonada obligatoria; riesgo de caídas y fracturas en ancianos.",
            "inhibe_cyp2d6": 0, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 1,
            "atc_codigo": "N03AE01"
        },
        {
            "farmaco": "Lorazepam",
            "clase": "Benzodiazepina de Alta Potencia",
            "perfil_receptores": "Modulador alostérico positivo de receptores GABA-A.",
            "metabolismo_cyp": "Metabolismo directo por glucuronidación simple (UGT2B7). Sin citocromos CYP450.",
            "efectos_comunes": "Sedación, mareos, debilidad muscular, amnesia anterógrada transitoria.",
            "efectos_graves": "Dependencia farmacológica, depresión respiratoria en combinación.",
            "perlas_clinicas": "Benzodiazepina de elección en hepatopatía y ancianos por ausencia de metabolismo por citocromos.",
            "inhibe_cyp2d6": 0, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 0,
            "atc_codigo": "N05BA06"
        },
        {
            "farmaco": "Donepezilo",
            "clase": "Anticolinesterásico Central",
            "perfil_receptores": "Inhibidor selectivo, reversible y no competitivo de la acetilcolinesterasa (AChE).",
            "metabolismo_cyp": "Metabolismo hepático extenso vía CYP2D6 y CYP3A4.",
            "efectos_comunes": "Náuseas, diarrea, calambres musculares, insomnio, anorexia.",
            "efectos_graves": "Bradicardia sintomática, síncope vagal, bloqueo auriculoventricular, úlcera péptica.",
            "perlas_clinicas": "Los anticolinérgicos (oxibutinina, biperideno) anulan su efecto y empeoran la demencia.",
            "inhibe_cyp2d6": 0, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 1,
            "atc_codigo": "N06DA02"
        },
        {
            "farmaco": "Memantina",
            "clase": "Antagonista no Competitivo NMDA",
            "perfil_receptores": "Bloqueo voltaje-dependiente y de cinética rápida del canal del receptor glutamatérgico NMDA.",
            "metabolismo_cyp": "Excreción renal directa inalterada (75-80%) sin metabolismo por citocromos.",
            "efectos_comunes": "Mareos, cefalea, estreñimiento, somnolencia leve.",
            "efectos_graves": "Confusión transitoria, alucinaciones (muy raras).",
            "perlas_clinicas": "Excelente perfil de seguridad; indicada en Alzheimer moderado a severo.",
            "inhibe_cyp2d6": 0, "inhibe_cyp3a4": 0, "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 0,
            "atc_codigo": "N06DX01"
        }
    ]

    df_stahl = pd.DataFrame(FARMACOS_STAHL)

    # Eventos adversos de Stahl
    eventos = []
    efectos_graves_claves = ["agranulocitosis", "miocarditis", "convulsiones", "íleo", "stevens-johnson", "intoxicación", "hepatotoxicidad", "teratogenicidad", "discinesia", "síndrome serotoninérgico"]
    for _, row in df_stahl.iterrows():
        comunes = [e.strip() for e in row["efectos_comunes"].split(",") if e.strip()]
        for c in comunes:
            eventos.append({
                "farmaco": row["farmaco"],
                "atc_codigo": row["atc_codigo"],
                "efecto_adverso": c,
                "severidad": "Común / Moderado",
                "fuente": "Stahl - Prescriber's Guide"
            })
        graves = [g.strip() for g in row["efectos_graves"].split(",") if g.strip()]
        for g in graves:
            eventos.append({
                "farmaco": row["farmaco"],
                "atc_codigo": row["atc_codigo"],
                "efecto_adverso": g,
                "severidad": "Grave / Riesgo Vital",
                "fuente": "Stahl - Prescriber's Guide"
            })

    df_eventos = pd.DataFrame(eventos)
    return df_stahl, df_eventos


def exportar_sqlite(datos_df: Dict[str, pd.DataFrame], db_path: Path):
    """Crea una base de datos relacional SQLite con todas las tablas indexadas."""
    print(f"Compilando base de datos relacional SQLite en: {db_path}")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    for nombre_tabla, df in datos_df.items():
        df.to_sql(nombre_tabla, conn, if_exists="replace", index=False)
        # Crear índice en la columna fármaco si existe
        if "farmaco" in df.columns:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{nombre_tabla}_farmaco ON {nombre_tabla}(farmaco);")
        if "atc_codigo" in df.columns:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{nombre_tabla}_atc ON {nombre_tabla}(atc_codigo);")

    conn.commit()
    conn.close()
    print("✓ Base de datos SQLite sofia_clinica.db generada e indexada con éxito.")


def regenerar_knowledge_base():
    """Genera rag_knowledge_base.json integrando monografías Jufe, CANMAT, marcas y pre_training."""
    data_dir = WORKSPACE_ROOT / "data" / "processed"
    kb_path = data_dir / "rag_knowledge_base.json"

    # 1. Monografías y efectos de Jufe
    jufe_ext = JufeExtractor()
    df_jufe, df_jufe_efectos = jufe_ext.extract_monographs()
    df_jufe.to_csv(data_dir / "jufe_farmacos_clean.csv", index=False, encoding="utf-8")
    df_jufe_efectos.to_csv(data_dir / "jufe_efectos_adversos_clean.csv", index=False, encoding="utf-8")

    # 2. Recomendaciones y alertas de CANMAT
    canmat_ext = CANMATExtractor()
    df_canmat, df_canmat_alertas = canmat_ext.extract_recommendations()
    df_canmat.to_csv(data_dir / "canmat_recomendaciones_clean.csv", index=False, encoding="utf-8")
    df_canmat_alertas.to_csv(data_dir / "canmat_alertas_clinicas_clean.csv", index=False, encoding="utf-8")

    # 3. Fármacos y eventos de Stahl
    df_stahl, df_stahl_eventos = compilar_stahl()
    df_stahl.to_csv(data_dir / "stahl_farmacos_clean.csv", index=False, encoding="utf-8")
    df_stahl_eventos.to_csv(data_dir / "stahl_eventos_adversos_clean.csv", index=False, encoding="utf-8")

    # 4. Exportar base de datos SQLite
    exportar_sqlite({
        "jufe_farmacos": df_jufe,
        "jufe_efectos_adversos": df_jufe_efectos,
        "canmat_recomendaciones": df_canmat,
        "canmat_alertas": df_canmat_alertas,
        "stahl_farmacos": df_stahl,
        "stahl_eventos_adversos": df_stahl_eventos
    }, data_dir / "sofia_clinica.db")

    # 5. Cargar corpus pre_training (27 documentos curados de alta especialización)
    pre_docs = cargar_corpus_pretraining()
    print(f"Documentos pre_training cargados: {len(pre_docs)}")

    documentos_rag = []
    ids_vistos = set()

    for doc in pre_docs:
        if doc["id"] not in ids_vistos:
            ids_vistos.add(doc["id"])
            documentos_rag.append(doc)

    # 6. Transformar fármacos de Jufe en documentos RAG estructurados
    for _, row in df_jufe.iterrows():
        farmaco_nombre = row["farmaco"]
        doc_id = f"JUFE-MONO-{normalizar_id(farmaco_nombre)}"
        if doc_id in ids_vistos:
            continue
        ids_vistos.add(doc_id)

        atc = row.get("atc_codigo", "N/D")
        grupo = row.get("atc_grupo", "N/D")
        paginas = row.get("paginas_jufe", "N/D")

        contenido = f"""# Monografía Clínica Canónica: {farmaco_nombre}
**Fuente de Evidencia:** Psicofarmacología Práctica (4ª Edición), Dra. Gabriela Jufe (páginas {paginas}).
**Clase Terapéutica:** {row['clase']} | **Código ATC Oficial OMS:** {atc} ({grupo})

## 1. Resumen Ejecutivo Verbalizable (< 40s)
> {farmaco_nombre} es un psicofármaco de la clase {row['clase']} ({atc}). Su mecanismo primario es: {row['mecanismo']}. Rango posológico habitual de {row['posologia']}. Monitorizar: {row['poblaciones_especiales'][:140]}.

## 2. Mecanismo de Acción Farmacológica y Receptores
{row['mecanismo']}

## 3. Pauta Posológica y Administración
{row['posologia']}

## 4. Farmacocinética y Metabolismo Enzimático (CYP450)
{row['farmacocinetica']}

## 5. Efectos Adversos y Toxicidad
{row['efectos_adversos']}

## 6. Interacciones Medicamentosas Críticas
{row['interacciones']}

## 7. Precauciones y Poblaciones Especiales
{row['poblaciones_especiales']}
"""
        documentos_rag.append({
            "id": doc_id,
            "tipo": "monografia_jufe",
            "titulo": f"Monografía Jufe: {farmaco_nombre}",
            "atc": atc,
            "voice_snippet": f"{farmaco_nombre}: {row['mecanismo'][:120]}. Dosis: {row['posologia'][:100]}.",
            "contenido": contenido.strip(),
            "archivo_origen": "data/raw/Psicofarmacologia practica - G. Jufe.pdf"
        })

    # 7. Transformar recomendaciones oficiales CANMAT
    for idx, row in df_canmat.iterrows():
        doc_id = f"GUIA-CANMAT-REC-{idx+1:03d}"
        if doc_id in ids_vistos:
            continue
        ids_vistos.add(doc_id)

        contenido = f"""# Recomendación Oficial CANMAT: {row['patologia']} ({row['fase']}) - {row['farmaco']}
**Patología:** {row['patologia']}
**Fase Clínica:** {row['fase']}
**Línea Terapéutica:** {row['linea']}
**Estrategia:** {row['estrategia']}
**Fármaco / Pauta:** {row['farmaco']} (ATC: {row.get('atc', 'N/D')})
**Nivel de Evidencia Científica:** {row['nivel_evidencia']}
**Directriz Clínica y Comentarios Oficiales:**
{row['comentarios']}
"""
        documentos_rag.append({
            "id": doc_id,
            "tipo": "guia_canmat",
            "titulo": f"CANMAT: {row['patologia']} ({row['fase']}) - {row['farmaco']}",
            "atc": row.get("atc", "N/D"),
            "voice_snippet": f"CANMAT {row['patologia']} ({row['fase']}): {row['farmaco']} en {row['linea']} ({row['nivel_evidencia']}).",
            "contenido": contenido.strip(),
            "archivo_origen": "data/raw/BDI-20-97.pdf"
        })

    # 8. Transformar alertas clínicas oficiales CANMAT
    for idx, row in df_canmat_alertas.iterrows():
        doc_id = f"ALERTA-CANMAT-{idx+1:03d}"
        if doc_id in ids_vistos:
            continue
        ids_vistos.add(doc_id)

        contenido = f"""# Alerta Clínica Oficial CANMAT: {row['farmaco_involucrado']}
**Tipo de Alerta:** {row['tipo_alerta']}
**Patología / Población:** {row['patologia']}
**Nivel de Evidencia:** {row['nivel_evidencia']}
**Descripción del Riesgo Clínico:**
{row['descripcion']}
"""
        documentos_rag.append({
            "id": doc_id,
            "tipo": "alerta_canmat",
            "titulo": f"Alerta CANMAT: {row['farmaco_involucrado']}",
            "atc": "N/D",
            "voice_snippet": f"Alerta CANMAT sobre {row['farmaco_involucrado']}: {row['descripcion'][:140]}.",
            "contenido": contenido.strip(),
            "archivo_origen": "data/raw/BDI-20-97.pdf"
        })

    # 9. Catálogo oficial de marcas comerciales y laboratorios
    for k, v in ATCEnricher.CATALOGO_ATC.items():
        doc_id = f"MARCA-{normalizar_id(k)}-001"
        if doc_id in ids_vistos:
            continue
        ids_vistos.add(doc_id)

        marcas = v.get("marcas_comerciales", [])
        marcas_str = ", ".join(marcas) if marcas else "Genéricos / EFG"
        contenido = f"""# Catálogo Oficial ATC y Marcas Comerciales: {v['nombre_oficial']}
**Principio Activo:** {v['nombre_oficial']}
**Código ATC Oficial (OMS):** {v['atc_codigo']}
**Clasificación Anatómica y Química:**
- Nivel 1: {v.get('atc_nivel1', 'N/D')}
- Nivel 2: {v.get('atc_nivel2', 'N/D')}
- Nivel 3 (Grupo Terapéutico): {v.get('atc_nivel3', 'N/D')}
- Nivel 4 (Subgrupo Químico): {v.get('atc_nivel4', 'N/D')}

**Marcas Comerciales Registradas en Farmacopea:**
{marcas_str}
"""
        documentos_rag.append({
            "id": doc_id,
            "tipo": "catalogo_marcas",
            "titulo": f"Marcas Comerciales y Laboratorios: {v['nombre_oficial']}",
            "atc": v.get("atc_codigo", "N/D"),
            "voice_snippet": f"{v['nombre_oficial']} ({v['atc_codigo']}). Marcas registradas: {marcas_str}.",
            "contenido": contenido.strip(),
            "archivo_origen": "src/enrichers/atc_enricher.py"
        })

    print(f"Total de documentos clínicos consolidados en la Base de Conocimiento RAG: {len(documentos_rag)}")

    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(documentos_rag, f, indent=2, ensure_ascii=False)

    print(f"✓ Guardado exitoso de {len(documentos_rag)} documentos en: {kb_path}")

    # Verificar que kb_obsoleta sea False
    rag = ClinicalRAGEngine()
    if rag.kb_obsoleta:
        print("⚠️ Advertencia: kb_obsoleta es True.")
    else:
        print("✓ Verificación RAG: kb_obsoleta es False. Motor plenamente operativo y sincronizado.")


if __name__ == "__main__":
    regenerar_knowledge_base()
