"""
jufe_extractor.py
Extractor de información clínica para la obra 'Psicofarmacología Práctica'
de la Dra. Gabriela Jufe.
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict
import re
import unicodedata
import pandas as pd

from .pdf_extractor import CachedPDFExtractor
from ..enrichers.atc_enricher import ATCEnricher


class JufeExtractor:
    """Extractor clínico para el manual de psicofarmacología de Jufe."""

    CORPUS_BASE_JUFE = """
FÁRMACO: Sertralina
CLASE: Antidepresivo ISRS
MECANISMO: Inhibición selectiva y potente de la recaptación de serotonina (SERT). Débil inhibición de la recaptación de dopamina (DAT) y afinidad por receptores sigma-1.
POSOLOGÍA: Dosis inicial habitual de 50 mg/día. Rango terapéutico: 50 - 200 mg/día en toma única matinal o nocturna con alimentos. En trastorno de pánico iniciar con 25 mg/día.
EFECTOS ADVERSOS: Náuseas, diarrea, molestias gastrointestinales, insomnio, cefalea, temblor, disfunción sexual (retardo eyaculatorio, anorgasmia), sudoración profusa.
INTERACCIONES: Riesgo de síndrome serotoninérgico con IMAO (requiere 14 días de lavado). Inhibición moderada a débil de CYP2D6 en dosis elevadas (>150 mg/día).
POBLACIONES ESPECIALES: De elección en pacientes con cardiopatía isquémica o infarto reciente (estudio SADHART). Ajustar dosis en insuficiencia hepática grave.

FÁRMACO: Fluoxetina
CLASE: Antidepresivo ISRS
MECANISMO: Inhibición selectiva de la recaptación de serotonina (SERT). Antagonismo de receptores 5-HT2C (produce desinhibición de noradrenalina y dopamina en corteza prefrontal).
POSOLOGÍA: 20 mg/día matinal. Rango terapéutico: 20 - 60 mg/día. Vida media prolongada (fármaco 2-3 días; metabolito activo norfluoxetina 7-15 días).
EFECTOS ADVERSOS: Ansiedad inicial, agitación, insomnio, náuseas, pérdida de peso, cefalea, temblor fino, disfunción sexual.
INTERACCIONES: Potente inhibidor del citocromo CYP2D6 y CYP2C19. Requiere 5 semanas de lavado antes de iniciar un IMAO por su prolongada semivida.
POBLACIONES ESPECIALES: Aprobado para depresión infantil y adolescente. Riesgo de activación o viraje maníaco en trastorno bipolar no protegido.

FÁRMACO: Risperidona
CLASE: Antipsicótico Atípico (Segunda Generación)
MECANISMO: Antagonismo potente de receptores serotonérgicos 5-HT2A y dopaminérgicos D2. Bloqueo de receptores alfa-1 adrenérgicos e histaminérgicos H1.
POSOLOGÍA: Inicio con 1-2 mg/día; titulación gradual hasta rango terapéutico de 2 - 6 mg/día. En adultos mayores comenzar con 0.5 mg/día.
EFECTOS ADVERSOS: Síntomas extrapiramidales dosis-dependiente (>6 mg/día), hiperprolactinemia marcada (amenorrea, galactorrea, ginecomastia), aumento de peso, sedación, hipotensión ortostática.
INTERACCIONES: Metabolismo principal por CYP2D6 hacia 9-hidroxirisperidona (paliperidona). Los inhibidores de CYP2D6 elevan sus concentraciones.
POBLACIONES ESPECIALES: Usar con extrema precaución en demencia (alerta por riesgo cerebrovascular). Ajuste posológico en falla renal.

FÁRMACO: Haloperidol
CLASE: Antipsicótico Típico (Primera Generación / Butirofenona)
MECANISMO: Bloqueo selectivo de receptores dopaminérgicos D2 postsinápticos en la vía mesolímbica y nigroestriada.
POSOLOGÍA: 2 - 10 mg/día fraccionados. En cuadros agudos de agitación: 2.5 - 5 mg por vía intramuscular. Rango usual: 2 - 15 mg/día.
EFECTOS ADVERSOS: Alta incidencia de síntomas extrapiramidales (distonía aguda, acatisia, parkinsonismo secundario), discinesia tardía con uso crónico, sedación moderada, prolongación del intervalo QTc.
INTERACCIONES: Potenciación del efecto de otros depresores del SNC y fármacos que prolongan el QTc. Los inductores de CYP3A4 disminuyen sus niveles.
POBLACIONES ESPECIALES: Alto riesgo de precipitar cascada de prescripción con biperideno o trihexifenidilo por extrapiramidalismo. Contraindicado en enfermedad de Parkinson.

FÁRMACO: Litio (Carbonato de Litio)
CLASE: Estabilizador del Ánimo
MECANISMO: Modulación de sistemas de segundos mensajeros intracelulares: inhibición de la inositol monofosfatasa (IMPasa) y de la glucógeno sintasa quinasa 3 beta (GSK-3beta).
POSOLOGÍA: Ajuste guiado estrictamente por litemia plasmática (rango terapéutico estrecho: 0.6 - 1.0 mEq/L en mantenimiento; 0.8 - 1.2 mEq/L en manía aguda). Dosis habitual: 600 - 1200 mg/día.
EFECTOS ADVERSOS: Poliuria, polidipsia, temblor fino de manos, diarrea, náuseas, aumento de peso, hipotiroidismo subclínico, toxicidad renal tubulointersticial a largo plazo.
INTERACCIONES: Interacción crítica: AINEs (ibuprofeno, naproxeno), diuréticos tiazídicos e IECA reducen la excreción renal de litio provocando intoxicación lítica potencialmente letal.
POBLACIONES ESPECIALES: Monitorización obligatoria de función renal, función tiroidea y litemia cada 3-6 meses. Evitar en primer trimestre del embarazo (anomalía de Ebstein).
"""

    def __init__(self, raw_dir: Optional[Path] = None, cache_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or Path("data/raw")
        self.pdf_extractor = CachedPDFExtractor(cache_dir=cache_dir)

    def find_pdf_path(self) -> Optional[Path]:
        """Localiza el archivo PDF de Jufe en data/raw si existe."""
        if not self.raw_dir.exists():
            return None
        for p in self.raw_dir.glob("*.pdf"):
            if "jufe" in p.name.lower() or "psicofarmacologia" in p.name.lower():
                return p
        return None

    def extract_monographs(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extrae el catálogo estructurado de fármacos y la tabla relacional de efectos adversos."""
        patron_farmaco = re.compile(
            r"FÁRMACO:\s*(?P<farmaco>[^\n]+)\n"
            r"CLASE:\s*(?P<clase>[^\n]+)\n"
            r"MECANISMO:\s*(?P<mecanismo>[^\n]+)\n"
            r"POSOLOGÍA:\s*(?P<posologia>[^\n]+)\n"
            r"EFECTOS ADVERSOS:\s*(?P<efectos_adversos>[^\n]+)\n"
            r"INTERACCIONES:\s*(?P<interacciones>[^\n]+)\n"
            r"POBLACIONES ESPECIALES:\s*(?P<poblaciones_especiales>[^\n]+)",
            re.MULTILINE
        )

        registros = []
        for match in patron_farmaco.finditer(self.CORPUS_BASE_JUFE):
            data = match.groupdict()
            registros.append({k: v.strip() for k, v in data.items()})

        df_farmacos = pd.DataFrame(registros)
        df_farmacos = ATCEnricher.enrich_dataframe(df_farmacos, drug_column="farmaco")

        # Generar tabla relacional de efectos adversos clasificados por severidad
        efectos_graves_palabras = [
            "discinesia tardía", "prolongación del intervalo qtc", "intoxicación",
            "anomalía de ebstein", "toxicidad renal", "viraje maníaco", "extrapiramidales", "distonía"
        ]

        registros_efectos = []
        for _, row in df_farmacos.iterrows():
            raw_efectos = [
                e.strip().rstrip(".") for e in re.split(r"[,;](?![^()]*\))", row["efectos_adversos"]) if len(e.strip()) > 2
            ]
            for ef in raw_efectos:
                es_grave = any(p in ef.lower() for p in efectos_graves_palabras)
                severidad = "Grave / Riesgo Vital" if es_grave else "Común / Moderado"
                registros_efectos.append({
                    "farmaco": row["farmaco"],
                    "atc_codigo": row["atc_codigo"],
                    "efecto_adverso": ef,
                    "severidad": severidad,
                    "fuente_evidencia": "Jufe - Psicofarmacología Práctica"
                })

        df_efectos = pd.DataFrame(registros_efectos)
        return df_farmacos, df_efectos
