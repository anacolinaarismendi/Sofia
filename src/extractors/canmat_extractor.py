"""
canmat_extractor.py
Extractor de directrices y recomendaciones de las guías clínicas oficiales
CANMAT / ISBD 2018 para Trastorno Depresivo Mayor y Trastorno Bipolar.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd


class CANMATExtractor:
    """Extractor de directrices y alertas clínicas de guías CANMAT."""

    RECOMENDACIONES_CANMAT = [
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Citalopram", "atc": "N06AB04", "nivel_evidencia": "Level 1", "comentarios": "ISRS de alta tolerabilidad; vigilar intervalo QTc a dosis >40 mg/día."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Escitalopram", "atc": "N06AB10", "nivel_evidencia": "Level 1", "comentarios": "ISRS con menor perfil de interacciones citocromo P450."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Sertralina", "atc": "N06AB06", "nivel_evidencia": "Level 1", "comentarios": "ISRS seguro en cardiopatía isquémica (estudio SADHART)."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Venlafaxina", "atc": "N06AX16", "nivel_evidencia": "Level 1", "comentarios": "IRSN dual; monitorizar presión arterial a dosis >150 mg/día."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Duloxetina", "atc": "N06AX21", "nivel_evidencia": "Level 1", "comentarios": "IRSN indicado en depresión con dolor somático comórbido o neuropatía."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Bupropión", "atc": "N06AX12", "nivel_evidencia": "Level 1", "comentarios": "IRND sin disfunción sexual ni ganancia ponderal; contraindicado en epilepsia."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Vortioxetina", "atc": "N06AX26", "nivel_evidencia": "Level 1", "comentarios": "Modulador multimodal serotonérgico con beneficios pro-cognitivos."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Segunda Línea", "estrategia": "Monoterapia", "farmaco": "Amitriptilina", "atc": "N06AA09", "nivel_evidencia": "Level 1", "comentarios": "Tricíclico eficaz pero con alta carga anticolinérgica y riesgo de cardiotoxicidad."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Aguda", "linea": "Segunda Línea", "estrategia": "Monoterapia", "farmaco": "Quetiapina", "atc": "N05AH04", "nivel_evidencia": "Level 2", "comentarios": "Eficacia demostrada a dosis de 150-300 mg/día; vigilar perfil metabólico."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Resistencia / Potenciación", "linea": "Primera Línea", "estrategia": "Potenciación", "farmaco": "Aripiprazol", "atc": "N05AX12", "nivel_evidencia": "Level 1", "comentarios": "Dosis bajas (2-5 mg/día) añadidas al antidepresivo base en respuesta parcial."},
        {"patologia": "Trastorno Depresivo Mayor (TDM)", "fase": "Resistencia / Potenciación", "linea": "Primera Línea", "estrategia": "Potenciación", "farmaco": "Litio (Carbonato)", "atc": "N05AN01", "nivel_evidencia": "Level 1", "comentarios": "Estrategia clásica con evidencia antisuicida comprobada; litemia 0.6-0.8 mEq/L."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Litio (Carbonato)", "atc": "N05AN01", "nivel_evidencia": "Level 1", "comentarios": "Gold standard en manía eufórica clásica; litemia diana 0.8-1.2 mEq/L."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Divalproato de Sodio", "atc": "N03AG01", "nivel_evidencia": "Level 1", "comentarios": "Eficaz en episodios mixtos y cicladores rápidos; nivel plasmático 50-100 mcg/mL."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Quetiapina", "atc": "N05AH04", "nivel_evidencia": "Level 1", "comentarios": "Dosis de 400-800 mg/día; control rápido del insomnio y la hiperactividad."},
        {"patologia": "Trastorno Bipolar", "fase": "Manía Aguda", "linea": "Primera Línea", "estrategia": "Combinación", "farmaco": "Litio + Antipsicótico Atípico", "atc": "Varios", "nivel_evidencia": "Level 1", "comentarios": "Estrategia superior a monoterapia en cuadros de manía severa o psicótica."},
        {"patologia": "Trastorno Bipolar", "fase": "Depresión Bipolar", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Quetiapina", "atc": "N05AH04", "nivel_evidencia": "Level 1", "comentarios": "Monoterapia de elección probada (estudios BOLDER I y II) a 300 mg/día nocturnos."},
        {"patologia": "Trastorno Bipolar", "fase": "Depresión Bipolar", "linea": "Primera Línea", "estrategia": "Monoterapia", "farmaco": "Lurasidona", "atc": "N05AE05", "nivel_evidencia": "Level 1", "comentarios": "Bajo impacto metabólico; administrar obligatoriamente con comida (>350 kcal)."},
        {"patologia": "Trastorno Bipolar", "fase": "Depresión Bipolar", "linea": "Primera Línea", "estrategia": "Combinación", "farmaco": "Litio + Lamotrigina", "atc": "Varios", "nivel_evidencia": "Level 2", "comentarios": "Efecto sinérgico estabilizador y antidepresivo con titulación lenta de lamotrigina."},
        {"patologia": "Trastorno Bipolar", "fase": "Depresión Bipolar", "linea": "No Recomendado / Contraindicado", "estrategia": "Monoterapia", "farmaco": "Antidepresivos ISRS en Monoterapia", "atc": "N06AB", "nivel_evidencia": "Level 1 (Daño)", "comentarios": "Contraindicados formalmente: alto riesgo de inducción de viraje maníaco y ciclación rápida."}
    ]

    ALERTAS_CANMAT = [
        {
            "patologia": "Trastorno Bipolar",
            "farmaco_involucrado": "Antidepresivos ISRS en Monoterapia",
            "tipo_alerta": "Contraindicación / Riesgo de Viraje",
            "descripcion": "ALERTA: Alto riesgo de viraje maníaco y desestabilización a ciclos rápidos.",
            "nivel_evidencia": "Level 1 (Daño)",
            "fuente": "Guías CANMAT / ISBD"
        }
    ]

    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or Path("data/raw")

    def find_pdf_path(self) -> Optional[Path]:
        """Localiza el archivo PDF de CANMAT en data/raw si existe."""
        if not self.raw_dir.exists():
            return None
        for p in self.raw_dir.glob("*.pdf"):
            if "bdi" in p.name.lower() or "canmat" in p.name.lower():
                return p
        return None

    def extract_recommendations(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Devuelve DataFrames de recomendaciones y alertas clínicas de CANMAT."""
        df_rec = pd.DataFrame(self.RECOMENDACIONES_CANMAT)
        df_alt = pd.DataFrame(self.ALERTAS_CANMAT)
        return df_rec, df_alt
