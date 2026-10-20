"""
atc_enricher.py
Enriquecimiento y validación de principios activos con el sistema oficial
de Clasificación Anatómica, Terapéutica y Química (ATC) de la OMS.
"""

from typing import Dict, Any, Optional
import unicodedata
import pandas as pd


def normalizar_clave(texto: str) -> str:
    """Normaliza un nombre de fármaco eliminando acentos y espacios innecesarios."""
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto.strip().lower()


class ATCEnricher:
    """Catálogo y enriquecedor oficial de códigos ATC y marcas comerciales."""

    CATALOGO_ATC: Dict[str, Dict[str, Any]] = {
        "sertralina": {
            "nombre_oficial": "Sertralina",
            "atc_codigo": "N06AB06",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Antidepresivos",
            "atc_nivel4": "Inhibidores selectivos de la recaptación de serotonina (ISRS)",
            "marcas_comerciales": ["Zoloft", "Serlift", "Aremis", "Besitran", "Altruline"]
        },
        "fluoxetina": {
            "nombre_oficial": "Fluoxetina",
            "atc_codigo": "N06AB03",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Antidepresivos",
            "atc_nivel4": "Inhibidores selectivos de la recaptación de serotonina (ISRS)",
            "marcas_comerciales": ["Prozac", "Foxetin", "Sarafem", "Adofen"]
        },
        "paroxetina": {
            "nombre_oficial": "Paroxetina",
            "atc_codigo": "N06AB05",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Antidepresivos",
            "atc_nivel4": "Inhibidores selectivos de la recaptación de serotonina (ISRS)",
            "marcas_comerciales": ["Paxil", "Aropax", "Seroxat", "Motivan"]
        },
        "bupropion": {
            "nombre_oficial": "Bupropión",
            "atc_codigo": "N06AX12",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Otros antidepresivos",
            "atc_nivel4": "Inhibidor de recaptación de noradrenalina y dopamina (IRND)",
            "marcas_comerciales": ["Wellbutrin", "Zyban", "Elontril", "Odranal"]
        },
        "litio": {
            "nombre_oficial": "Carbonato de Litio",
            "atc_codigo": "N05AN01",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos / Estabilizadores",
            "atc_nivel4": "Compuestos de litio",
            "marcas_comerciales": ["Carbolit", "Plenur", "Eskalith", "Lithobid"]
        },
        "carbonato de litio": {
            "nombre_oficial": "Carbonato de Litio",
            "atc_codigo": "N05AN01",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos / Estabilizadores",
            "atc_nivel4": "Compuestos de litio",
            "marcas_comerciales": ["Carbolit", "Plenur", "Eskalith", "Lithobid"]
        },
        "clozapina": {
            "nombre_oficial": "Clozapina",
            "atc_codigo": "N05AH02",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Diazepinas, oxazepinas y tiazepinas (Atípico)",
            "marcas_comerciales": ["Leponex", "Clozaril", "Froidir", "Luverina"]
        },
        "risperidona": {
            "nombre_oficial": "Risperidona",
            "atc_codigo": "N05AX08",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Otros antipsicóticos (Atípico)",
            "marcas_comerciales": ["Risperdal", "Ridal", "Sequinan"]
        },
        "haloperidol": {
            "nombre_oficial": "Haloperidol",
            "atc_codigo": "N05AD01",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Derivados de butirofenona (Típico / Primera generación)",
            "marcas_comerciales": ["Haldol", "Halopidol", "Haldol Decanoato"]
        },
        "lamotrigina": {
            "nombre_oficial": "Lamotrigina",
            "atc_codigo": "N03AX09",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Antiepilépticos (N03)",
            "atc_nivel3": "Otros antiepilépticos",
            "atc_nivel4": "Estabilizador del ánimo",
            "marcas_comerciales": ["Lamictal", "Labileno", "Lamotrig"]
        },
        "valproato": {
            "nombre_oficial": "Ácido Valproico / Valproato",
            "atc_codigo": "N03AG01",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Antiepilépticos (N03)",
            "atc_nivel3": "Antiepilépticos",
            "atc_nivel4": "Derivados de ácidos grasos",
            "marcas_comerciales": ["Depakene", "Depakote", "Valcote"]
        },
        "acido valproico": {
            "nombre_oficial": "Ácido Valproico / Valproato",
            "atc_codigo": "N03AG01",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Antiepilépticos (N03)",
            "atc_nivel3": "Antiepilépticos",
            "atc_nivel4": "Derivados de ácidos grasos",
            "marcas_comerciales": ["Depakene", "Depakote", "Valcote"]
        },
        "quetiapina": {
            "nombre_oficial": "Quetiapina",
            "atc_codigo": "N05AH04",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Diazepinas, oxazepinas y tiazepinas (Atípico)",
            "marcas_comerciales": ["Seroquel", "Quetiazic", "Rocoz"]
        },
        "olanzapina": {
            "nombre_oficial": "Olanzapina",
            "atc_codigo": "N05AH03",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Diazepinas (Atípico)",
            "marcas_comerciales": ["Zyprexa", "Midax", "Arenbil"]
        },
        "aripiprazol": {
            "nombre_oficial": "Aripiprazol",
            "atc_codigo": "N05AX12",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Otros antipsicóticos (Agonista parcial D2)",
            "marcas_comerciales": ["Abilify", "Ilimit", "Arizol"]
        }
    }

    @classmethod
    def get_info(cls, nombre_farmaco: str) -> Optional[Dict[str, Any]]:
        """Busca información ATC para un fármaco dado su nombre común."""
        clave = normalizar_clave(nombre_farmaco)
        for k, v in cls.CATALOGO_ATC.items():
            if k in clave or clave in k:
                return v
        return None

    @classmethod
    def enrich_dataframe(cls, df: pd.DataFrame, drug_column: str = "farmaco") -> pd.DataFrame:
        """Enriquece un DataFrame añadiendo columnas ATC (atc_codigo y atc_grupo)."""
        df_out = df.copy()

        def _obtener_codigo(val):
            info = cls.get_info(str(val))
            return info["atc_codigo"] if info else "N/D"

        def _obtener_grupo(val):
            info = cls.get_info(str(val))
            return info["atc_nivel3"] if info else "N/D"

        if "atc_codigo" not in df_out.columns or df_out["atc_codigo"].isnull().any():
            df_out["atc_codigo"] = df_out[drug_column].apply(_obtener_codigo)
        if "atc_grupo" not in df_out.columns or df_out["atc_grupo"].isnull().any():
            df_out["atc_grupo"] = df_out[drug_column].apply(_obtener_grupo)

        return df_out
