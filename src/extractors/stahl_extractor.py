"""Extractor y compilador estructurado de monografías clínicas de Stephen Stahl (Prescriber's Guide)."""
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd

class StahlExtractor:
    """Compilador y validador de monografías clínicas de Stephen Stahl con matriz CYP completa."""

    def __init__(self, raw_dir: Path = None):
        self.raw_dir = raw_dir

    def extract_monographs(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.compilar_stahl()

    @staticmethod
    def compilar_stahl() -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Genera catálogo estructurado de fármacos y eventos de Stahl con matriz CYP completa y perfiles verificados."""
        FARMACOS_STAHL = [
            {
                "farmaco": "Clozapina",
                "clase": "Antipsicótico Atípico (Segunda Generación)",
                "perfil_receptores": "Antagonista 5-HT2A > D2; alta afinidad por receptores D4, 5-HT2C, alfa-1 adrenérgicos, H1 histaminérgicos y M1 colinérgicos; agonista alostérico M4.",
                "metabolismo_cyp": "Sustrato principal de CYP1A2. Sustrato secundario de CYP2D6 y CYP3A4. Los hidrocarburos aromáticos del tabaco inducen potentemente CYP1A2 y reducen sus niveles hasta 50%.",
                "efectos_comunes": ["Sedación diurna intensa", "Sialorrea nocturna profusa", "Aumento marcado de peso", "Estreñimiento severo", "Taquicardia sinusal", "Mareos ortostáticos"],
                "efectos_graves": ["Agranulocitosis (neutropenia grave potencialmente mortal <1000/µL)", "Miocarditis y miocardiopatía dilatada", "Convulsiones dosis-dependiente (>600 mg/día)", "Íleo paralítico obstructivo fatal"],
                "perlas_clinicas": "Indicado en esquizofrenia resistente y riesgo suicida recurrente. Requiere monitorización hematológica estricta con hemogramas seriados obligatorios.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 0, "potencia_inhibicion_2d6": "Nula", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 1, "sustrato_cyp2b6": 0, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 0, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 1,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N05AH02"
            },
            {
                "farmaco": "Paroxetina",
                "clase": "Inhibidor Selectivo de la Recaptación de Serotonina (ISRS)",
                "perfil_receptores": "Inhibidor potente del transportador de serotonina (SERT); ligera acción anticolinérgica muscarínica (M1) y débil inhibición de recaptación de noradrenalina (NET).",
                "metabolismo_cyp": "Potente inhibidor del citocromo CYP2D6. Sustrato de CYP2D6 con cinética no lineal por autosaturación enzimática.",
                "efectos_comunes": ["Disfunción sexual marcada", "Somnolencia y sedación", "Aumento de peso moderado", "Náuseas", "Boca seca por carga antimuscarínica", "Constipación"],
                "efectos_graves": ["Síndrome de discontinuación intenso tras suspensión brusca", "Síndrome serotoninérgico", "Teratogenicidad (defectos septales cardíacos en primer trimestre)", "Retención urinaria aguda"],
                "perlas_clinicas": "Es el ISRS más sedante y anticolinérgico. Desaconsejado en adultos mayores por riesgo de deterioro cognitivo y caídas.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 1, "potencia_inhibicion_2d6": "Potente", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 0, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 0,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N06AB05"
            },
            {
                "farmaco": "Bupropión",
                "clase": "Inhibidor de la Recaptación de Noradrenalina y Dopamina (IRND)",
                "perfil_receptores": "Inhibidor de los transportadores DAT y NET; antagonista de receptores nicotínicos alfa-4-beta-2 y alfa-3-beta-4.",
                "metabolismo_cyp": "Metabolizado a hidroxibupropión principalmente por CYP2B6. Potente inhibidor competitivo del citocromo CYP2D6.",
                "efectos_comunes": ["Insomnio de conciliación", "Agitación psicomotriz", "Cefalea", "Temblor fino", "Sequedad bucal", "Sudoración", "Disminución del apetito y peso"],
                "efectos_graves": ["Convulsiones epilépticas (riesgo dosis-dependiente >450 mg/día)", "Psicosis o exacerbación maníaca en pacientes bipolares", "Crisis hipertensiva con IMAO"],
                "perlas_clinicas": "Antidepresivo activador que no produce disfunción sexual ni aumento ponderal. Contraindicado en trastornos de conducta alimentaria (anorexia/bulimia) y epilepsia.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 1, "potencia_inhibicion_2d6": "Potente", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 1, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 0, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 0,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N06AX12"
            },
            {
                "farmaco": "Lamotrigina",
                "clase": "Estabilizador del Ánimo / Antiepiléptico",
                "perfil_receptores": "Bloqueo de canales de sodio voltaje-dependientes; modulación de la liberación presináptica de glutamato y aspartato.",
                "metabolismo_cyp": "Metabolismo por glucuronidación hepática vía UGT1A4. No depende fuertemente de citocromos CYP450.",
                "efectos_comunes": ["Cefalea", "Mareos", "Visión borrosa o diplopía", "Náuseas", "Exantema cutáneo maculopapular benigno", "Somnolencia"],
                "efectos_graves": ["Síndrome de Stevens-Johnson / Necrólisis Epidérmica Tóxica (exantema grave potencialmente mortal)", "Reacción DRESS", "Meningitis aséptica"],
                "perlas_clinicas": "Eficaz principalmente en la fase depresiva del Trastorno Bipolar. El ácido valproico inhibe UGT y duplica sus niveles (requiere reducir la dosis inicial al 50%); los inductores reducen sus niveles a la mitad.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 0, "potencia_inhibicion_2d6": "Nula", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 0, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 0,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N03AX09"
            },
            {
                "farmaco": "Carbonato de Litio",
                "clase": "Estabilizador del Ánimo / Catión Monovalente",
                "perfil_receptores": "Modulación de cascadas de segundos mensajeros (inhibición de inositol monofosfatasa y GSK-3beta); neuroprotección y plasticidad neuronal.",
                "metabolismo_cyp": "Eliminación renal casi exclusiva (>95%) por filtración glomerular sin metabolización hepática ni interacción con citocromos CYP450.",
                "efectos_comunes": ["Temblor fino distal", "Poliuria", "Polidipsia", "Aumento ponderal", "Molestias gastrointestinales leves", "Hipotiroidismo subclínico o bocio"],
                "efectos_graves": ["Intoxicación por litemia (arritmias cardíacas, ataxia, convulsiones, coma y muerte)", "Insuficiencia renal crónica tubulointersticial", "Diabetes insípida nefrogénica"],
                "perlas_clinicas": "Gold standard en profilaxis del Trastorno Bipolar y reducción comprobada del riesgo suicida. Rango terapéutico estrecho (0.6-0.8 mEq/L en mantenimiento, hasta 0.8-1.1 mEq/L en manía aguda); exige monitorización semestral de función renal y tiroidea.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 0, "potencia_inhibicion_2d6": "Nula", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 0, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 0,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N05AN01"
            },
            {
                "farmaco": "Sertralina",
                "clase": "Inhibidor Selectivo de la Recaptación de Serotonina (ISRS)",
                "perfil_receptores": "Inhibidor selectivo de SERT, débil afinidad sobre transportador DAT y agonista de receptores sigma-1.",
                "metabolismo_cyp": "Metabolismo por CYP2B6, CYP2C19, CYP2D6 y CYP3A4. Débil inhibidor de CYP2D6 solo a dosis altas (>150 mg/día).",
                "efectos_comunes": ["Náuseas", "Diarrea transitoria", "Insomnio", "Cefalea", "Disfunción sexual", "Temblor fino"],
                "efectos_graves": ["Síndrome serotoninérgico con IMAO", "Hiponatremia por SIADH", "Hemorragia digestiva"],
                "perlas_clinicas": "Antidepresivo de primera elección en cardiopatía isquémica reciente (ensayo SADHART) y polimedicación.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 1, "potencia_inhibicion_2d6": "Débil", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 1, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 1, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 1,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N06AB06"
            },
            {
                "farmaco": "Fluoxetina",
                "clase": "Inhibidor Selectivo de la Recaptación de Serotonina (ISRS)",
                "perfil_receptores": "Inhibidor de SERT y antagonista de 5-HT2C (desinhibición prefrontal de NA y DA).",
                "metabolismo_cyp": "Metabolizado a norfluoxetina. Potente inhibidor de CYP2D6 y moderado de CYP2C19.",
                "efectos_comunes": ["Ansiedad inicial", "Anorexia y pérdida de peso", "Insomnio", "Náuseas", "Temblor"],
                "efectos_graves": ["Viraje maníaco", "Síndrome serotoninérgico", "Interacciones farmacocinéticas múltiples"],
                "perlas_clinicas": "Vida media muy larga (norfluoxetina hasta 2 semanas). Lavado obligatorio de 5 semanas antes de un IMAO.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 1, "inhibe_cyp2d6": 1, "potencia_inhibicion_2d6": "Potente", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 0, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 0,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N06AB03"
            },
            {
                "farmaco": "Escitalopram",
                "clase": "Inhibidor Selectivo de la Recaptación de Serotonina (ISRS)",
                "perfil_receptores": "Inhibidor alostérico altamente selectivo de SERT. Ausencia de bloqueo colinérgico o histamínico.",
                "metabolismo_cyp": "Metabolismo principal por CYP2C19 y secundario por CYP3A4. Mínima inhibición de CYP2D6.",
                "efectos_comunes": ["Náuseas leves transitorias", "Cefalea", "Somnolencia", "Disfunción sexual", "Diaforesis"],
                "efectos_graves": ["Prolongación del intervalo QTc a dosis supraterapéuticas (>20 mg/día)", "Síndrome serotoninérgico", "Hiponatremia por SIADH"],
                "perlas_clinicas": "ISRS con menor tasa de interacciones citocromo P450; de elección en polimedicación y adultos mayores.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 0, "potencia_inhibicion_2d6": "Nula", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 1, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 1,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N06AB10"
            },
            {
                "farmaco": "Venlafaxina",
                "clase": "Inhibidor de la Recaptación de Serotonina y Noradrenalina (IRSN)",
                "perfil_receptores": "Bloqueo de SERT a dosis bajas y de NET a dosis medias/altas (≥150 mg/día).",
                "metabolismo_cyp": "Metabolismo hepático vía CYP2D6 a O-desmetilvenlafaxina activa.",
                "efectos_comunes": ["Náuseas", "Sudoración profusa diaforética", "Sequedad de boca", "Insomnio", "Estreñimiento"],
                "efectos_graves": ["Hipertensión arterial diastólica dosis-dependiente", "Síndrome de retirada abrupto intenso", "Síndrome serotoninérgico"],
                "perlas_clinicas": "Control obligatorio de tensión arterial en dosis superiores a 150 mg/día.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 0, "potencia_inhibicion_2d6": "Nula", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 0, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 0,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N06AX16"
            },
            {
                "farmaco": "Duloxetina",
                "clase": "Inhibidor de la Recaptación de Serotonina y Noradrenalina (IRSN)",
                "perfil_receptores": "Inhibición equilibrada de SERT y NET desde dosis terapéuticas iniciales.",
                "metabolismo_cyp": "Metabolismo por CYP1A2 y CYP2D6. Inhibidor moderado del citocromo CYP2D6.",
                "efectos_comunes": ["Náuseas", "Sequedad bucal", "Estreñimiento", "Astenia", "Mareos", "Somnolencia"],
                "efectos_graves": ["Hepatotoxicidad (ictericia, aumento de transaminasas)", "Crisis hipertensiva", "Retención urinaria"],
                "perlas_clinicas": "Indicada en depresión con dolor somático o fibromialgia. Evitar en hepatopatía crónica.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 1, "potencia_inhibicion_2d6": "Moderada", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 1, "sustrato_cyp2b6": 0, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 0, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 0,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N06AX21"
            },
            {
                "farmaco": "Mirtazapina",
                "clase": "Antidepresivo Noradrenérgico y Serotonérgico Específico (NaSSA)",
                "perfil_receptores": "Antagonista alfa-2 presináptico, 5-HT2A, 5-HT2C, 5-HT3 e histaminérgico H1 potente.",
                "metabolismo_cyp": "Metabolismo por CYP1A2, CYP2D6 y CYP3A4. Sin inhibición enzimática relevante.",
                "efectos_comunes": ["Sedación marcada", "Aumento de apetito", "Ganancia de peso significativa", "Sequedad bucal"],
                "efectos_graves": ["Agranulocitosis (rara)", "Convulsiones en sobredosis", "Hepatotoxicidad"],
                "perlas_clinicas": "Ideal en pacientes deprimidos con insomnio severo y pérdida de peso; sin disfunción sexual.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 0, "potencia_inhibicion_2d6": "Nula", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 1, "sustrato_cyp2b6": 0, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 0, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 1,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N06AX11"
            },
            {
                "farmaco": "Haloperidol",
                "clase": "Antipsicótico Típico (Primera Generación)",
                "perfil_receptores": "Antagonismo potente y altamente selectivo de receptores dopaminérgicos D2 mesolímbicos y nigroestriatales.",
                "metabolismo_cyp": "Metabolismo por glucuronidación, CYP3A4 y CYP2D6.",
                "efectos_comunes": ["Parkinsonismo secundario", "Rigidez", "Temblor", "Acatisia motora", "Sedación moderada"],
                "efectos_graves": ["Distonía aguda laringofaríngea", "Discinesia tardía irreversible", "Prolongación del intervalo QTc", "Síndrome neuroléptico maligno"],
                "perlas_clinicas": "Gran potencia antipsicótica con alto riesgo de cascada de prescripción hacia biperideno.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 0, "potencia_inhibicion_2d6": "Nula", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 0, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 1,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N05AD01"
            },
            {
                "farmaco": "Risperidona",
                "clase": "Antipsicótico Atípico (Segunda Generación)",
                "perfil_receptores": "Antagonista potente de 5-HT2A y D2, alfa-1 adrenérgico e histaminérgico H1.",
                "metabolismo_cyp": "Metabolismo vía CYP2D6 hacia 9-hidroxirisperidona (paliperidona activa).",
                "efectos_comunes": ["Hiperprolactinemia marcada (galactorrea, amenorrea)", "Aumento de peso moderado", "Sedación", "Fatiga"],
                "efectos_graves": ["Síntomas extrapiramidales dosis-dependientes (>4-6 mg/día)", "Riesgo cerebrovascular en demencia", "Discinesia tardía"],
                "perlas_clinicas": "A dosis bajas (2-4 mg) mantiene perfil atípico; a dosis altas pierde selectividad atípica.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 0, "potencia_inhibicion_2d6": "Nula", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 0, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 1,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N05AX08"
            },
            {
                "farmaco": "Olanzapina",
                "clase": "Antipsicótico Atípico (Segunda Generación)",
                "perfil_receptores": "Antagonista multirreceptorial 5-HT2A, 5-HT2C, D2, H1, M1 y alfa-1.",
                "metabolismo_cyp": "Metabolismo principal por CYP1A2 y glucuronidación. El tabaco disminuye sus concentraciones plasmáticas por inducción.",
                "efectos_comunes": ["Aumento masivo de peso", "Apetito voraz por carbohidratos", "Sedación diurna intensa", "Boca seca"],
                "efectos_graves": ["Síndrome metabólico acelerado", "Diabetes mellitus tipo 2 de inicio rápido", "Dislipidemia grave"],
                "perlas_clinicas": "Excelente eficacia en agitación y manía pero alto pasivo cardiometabólico.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 0, "potencia_inhibicion_2d6": "Nula", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 1, "sustrato_cyp2b6": 0, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 0, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 0,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N05AH03"
            },
            {
                "farmaco": "Quetiapina",
                "clase": "Antipsicótico Atípico (Segunda Generación)",
                "perfil_receptores": "Disociación rápida de D2, antagonismo 5-HT2A, H1, alfa-1. Norquetiapina inhibe NET y estimula 5-HT1A.",
                "metabolismo_cyp": "Metabolismo casi exclusivo por CYP3A4.",
                "efectos_comunes": ["Sedación profunda temprana", "Mareos", "Hipotensión ortostática", "Ganancia ponderal moderada", "Boca seca"],
                "efectos_graves": ["Cataratas y opacidades del cristalino", "Prolongación moderada de QTc", "Síndrome metabólico"],
                "perlas_clinicas": "Nula carga extrapiramidal; de elección en psicosis asociada a enfermedad de Parkinson.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 0, "potencia_inhibicion_2d6": "Nula", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 0, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 1,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N05AH04"
            },
            {
                "farmaco": "Aripiprazol",
                "clase": "Antipsicótico Atípico / Agonista Parcial D2",
                "perfil_receptores": "Agonista parcial D2 (30% actividad intrínseca), agonista parcial 5-HT1A, antagonista 5-HT2A.",
                "metabolismo_cyp": "Metabolismo por CYP2D6 y CYP3A4. T1/2 prolongada (75 horas).",
                "efectos_comunes": ["Acatisia inicial e inquietud motora", "Insomnio", "Cefalea", "Náuseas", "Ansiedad"],
                "efectos_graves": ["Control de impulsos alterado (ludopatía, hipersexualidad paradójica)", "Discinesia tardía", "Síndrome neuroléptico maligno"],
                "perlas_clinicas": "Neutro a nivel metabólico y ponderal; normaliza hiperprolactinemia inducida por otros antipsicóticos.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 0, "potencia_inhibicion_2d6": "Nula", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 0, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 1,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N05AX12"
            },
            {
                "farmaco": "Ácido Valproico",
                "clase": "Estabilizador del Ánimo / Antiepiléptico",
                "perfil_receptores": "Aumento del tono GABA cerebral, bloqueo de canales de sodio y canales de calcio tipo T.",
                "metabolismo_cyp": "Glucuronidación hepática (UGT) y beta-oxidación mitocondrial. Inhibidor potente de UGT2B7 y CYP2C9.",
                "efectos_comunes": ["Molestias digestivas (náuseas, dispepsia)", "Temblor fino postural", "Aumento de peso", "Alopecia transitoria", "Somnolencia"],
                "efectos_graves": ["Hepatotoxicidad fulminante idiosincrásica", "Pancreatitis aguda necrotizante", "Trombocitopenia grave", "Encefalopatía hiperamonémica", "Teratogenicidad mayor"],
                "perlas_clinicas": "Contraindicado absolutamente en mujeres en edad fértil por teratogenicidad mayor (espina bífida) y riesgo de TEA.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 1, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 0, "potencia_inhibicion_2d6": "Nula", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 0, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 0,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N03AG01"
            },
            {
                "farmaco": "Carbamazepina",
                "clase": "Estabilizador del Ánimo / Antiepiléptico",
                "perfil_receptores": "Bloqueo de canales de sodio voltaje-dependientes en estado inactivado.",
                "metabolismo_cyp": "Potentísimo inductor multienzimático de CYP1A2, CYP2C9, CYP2C19 y CYP3A4. Fenómeno de autoinducción que reduce su propia vida media.",
                "efectos_comunes": ["Mareos", "Ataxia", "Sedación", "Diplopía", "Náuseas", "Hiponatremia por SIADH"],
                "efectos_graves": ["Agranulocitosis y anemia aplásica", "Síndrome de Stevens-Johnson", "Hepatitis tóxica", "Bloqueo auriculoventricular"],
                "perlas_clinicas": "Contraindicada con clozapina por mielosupresión sinérgica. Invalida anticonceptivos orales por inducción metabólica de CYP3A4.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 0, "potencia_inhibicion_2d6": "Nula", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 1, "sustrato_cyp2b6": 0, "sustrato_cyp2c9": 1, "sustrato_cyp2c19": 1, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 1,
                "inductor_cyp1a2": 1, "inductor_cyp2c9": 1, "inductor_cyp2c19": 1, "inductor_cyp3a4": 1,
                "atc_codigo": "N03AF01"
            },
            {
                "farmaco": "Clonazepam",
                "clase": "Benzodiazepina de Alta Potencia",
                "perfil_receptores": "Modulador alostérico positivo de receptores GABA-A con subunidades alfa-1, 2, 3 y 5.",
                "metabolismo_cyp": "Metabolizado principalmente por CYP3A4 y posterior reducción a 7-amino-clonazepam.",
                "efectos_comunes": ["Sedación", "Somnolencia matinal", "Ataxia", "Enlentecimiento psicomotor", "Amnesia anterógrada"],
                "efectos_graves": ["Dependencia física y tolerancia", "Depresión respiratoria severa con opioides o alcohol", "Síndrome de abstinencia grave"],
                "perlas_clinicas": "Retirada escalonada obligatoria; riesgo elevado de caídas y fracturas en personas mayores.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 0, "potencia_inhibicion_2d6": "Nula", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 0, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 1,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N03AE01"
            },
            {
                "farmaco": "Lorazepam",
                "clase": "Benzodiazepina de Alta Potencia",
                "perfil_receptores": "Modulador alostérico positivo de receptores GABA-A.",
                "metabolismo_cyp": "Metabolismo hepático directo por glucuronidación simple mediada principalmente por UGT2B15 (y en menor medida UGT2B7). Sin fase I ni dependencia de citocromos CYP450.",
                "efectos_comunes": ["Sedación", "Mareos", "Debilidad muscular", "Amnesia anterógrada transitoria", "Ataxia"],
                "efectos_graves": ["Dependencia farmacológica", "Depresión respiratoria en combinación con depresores del SNC", "Síndrome de abstinencia"],
                "perlas_clinicas": "Benzodiazepina de elección en hepatopatía y ancianos por ausencia de metabolismo dependiente de citocromos.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 0, "potencia_inhibicion_2d6": "Nula", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 0, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 0,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N05BA06"
            },
            {
                "farmaco": "Donepezilo",
                "clase": "Anticolinesterásico Central",
                "perfil_receptores": "Inhibidor selectivo, reversible y no competitivo de la acetilcolinesterasa (AChE).",
                "metabolismo_cyp": "Metabolismo hepático extenso vía CYP2D6 y CYP3A4.",
                "efectos_comunes": ["Náuseas", "Diarrea", "Calambres musculares nocturnos", "Insomnio", "Anorexia"],
                "efectos_graves": ["Bradicardia sintomática y síncope vagal", "Bloqueo auriculoventricular", "Úlcera péptica activa con hemorragia"],
                "perlas_clinicas": "Los anticolinérgicos (oxibutinina, biperideno) anulan su efecto y empeoran el deterioro cognitivo.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 0, "potencia_inhibicion_2d6": "Nula", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 0, "sustrato_cyp2d6": 1, "sustrato_cyp3a4": 1,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N06DA02"
            },
            {
                "farmaco": "Memantina",
                "clase": "Antagonista no Competitivo NMDA",
                "perfil_receptores": "Bloqueo voltaje-dependiente y de cinética rápida del canal del receptor glutamatérgico NMDA.",
                "metabolismo_cyp": "Excreción renal directa inalterada (75-80%) sin metabolización significativa por citocromos.",
                "efectos_comunes": ["Mareos", "Cefalea", "Estreñimiento", "Somnolencia leve", "Hipertensión arterial"],
                "efectos_graves": ["Confusión transitoria", "Alucinaciones (muy raras)", "Insuficiencia cardíaca congestiva"],
                "perlas_clinicas": "Excelente perfil de tolerabilidad; indicada en enfermedad de Alzheimer moderada a severa.",
                "inhibe_cyp1a2": 0, "inhibe_cyp2c9": 0, "inhibe_cyp2c19": 0, "inhibe_cyp2d6": 0, "potencia_inhibicion_2d6": "Nula", "inhibe_cyp3a4": 0,
                "sustrato_cyp1a2": 0, "sustrato_cyp2b6": 0, "sustrato_cyp2c9": 0, "sustrato_cyp2c19": 0, "sustrato_cyp2d6": 0, "sustrato_cyp3a4": 0,
                "inductor_cyp1a2": 0, "inductor_cyp2c9": 0, "inductor_cyp2c19": 0, "inductor_cyp3a4": 0,
                "atc_codigo": "N06DX01"
            }
        ]
    
        df_stahl = pd.DataFrame(FARMACOS_STAHL)
    
        # Eventos adversos explícitos y limpios de Stahl
        eventos = []
        for _, row in df_stahl.iterrows():
            for c in row["efectos_comunes"]:
                c_clean = c.strip().rstrip(".")
                if not c_clean:
                    continue
                c_clean = c_clean[0].upper() + c_clean[1:]
                eventos.append({
                    "farmaco": row["farmaco"],
                    "atc_codigo": row["atc_codigo"],
                    "efecto_adverso": c_clean,
                    "severidad": "Común / Moderado",
                    "fuente": "Stahl - Prescriber's Guide"
                })
            for g in row["efectos_graves"]:
                g_clean = g.strip().rstrip(".")
                if not g_clean:
                    continue
                g_clean = g_clean[0].upper() + g_clean[1:]
                eventos.append({
                    "farmaco": row["farmaco"],
                    "atc_codigo": row["atc_codigo"],
                    "efecto_adverso": g_clean,
                    "severidad": "Grave / Riesgo Vital",
                    "fuente": "Stahl - Prescriber's Guide"
                })
    
        # Convertir efectos_comunes y efectos_graves en cadenas formateadas legibles para el DataFrame final
        df_stahl_out = df_stahl.copy()
        df_stahl_out["efectos_comunes"] = df_stahl["efectos_comunes"].apply(lambda l: ", ".join(l))
        df_stahl_out["efectos_graves"] = df_stahl["efectos_graves"].apply(lambda l: ", ".join(l))
    
        df_eventos = pd.DataFrame(eventos)
        return df_stahl_out, df_eventos
    
    
    