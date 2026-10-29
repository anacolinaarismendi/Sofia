"""
atc_enricher.py
Enriquecimiento y validación de principios activos con el sistema oficial
de Clasificación Anatómica, Terapéutica y Química (ATC) de la OMS.
Incluye catálogo exhaustivo de psicofármacos y marcas comerciales registradas.
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
    """Catálogo y enriquecedor oficial de códigos ATC y marcas comerciales para psicofarmacología."""

    MIN_LEN_BUSQUEDA = 4

    CATALOGO_ATC: Dict[str, Dict[str, Any]] = {
        # --- Antidepresivos ISRS ---
        "sertralina": {
            "nombre_oficial": "Sertralina",
            "atc_codigo": "N06AB06",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Antidepresivos",
            "atc_nivel4": "Inhibidores selectivos de la recaptación de serotonina (ISRS)",
            "marcas_comerciales": ["Zoloft", "Serlift", "Aremis", "Besitran", "Altruline", "Aserin"]
        },
        "fluoxetina": {
            "nombre_oficial": "Fluoxetina",
            "atc_codigo": "N06AB03",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Antidepresivos",
            "atc_nivel4": "Inhibidores selectivos de la recaptación de serotonina (ISRS)",
            "marcas_comerciales": ["Prozac", "Foxetin", "Sarafem", "Adofen", "Ansilan"]
        },
        "paroxetina": {
            "nombre_oficial": "Paroxetina",
            "atc_codigo": "N06AB05",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Antidepresivos",
            "atc_nivel4": "Inhibidores selectivos de la recaptación de serotonina (ISRS)",
            "marcas_comerciales": ["Paxil", "Aropax", "Seroxat", "Motivan", "Psicoasten", "Tiarix"]
        },
        "citalopram": {
            "nombre_oficial": "Citalopram",
            "atc_codigo": "N06AB04",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Antidepresivos",
            "atc_nivel4": "Inhibidores selectivos de la recaptación de serotonina (ISRS)",
            "marcas_comerciales": ["Celexa", "Seropram", "Prisdal", "Psiconor", "Zentius"]
        },
        "escitalopram": {
            "nombre_oficial": "Escitalopram",
            "atc_codigo": "N06AB10",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Antidepresivos",
            "atc_nivel4": "Inhibidores selectivos de la recaptación de serotonina (ISRS)",
            "marcas_comerciales": ["Lexapro", "Cipralex", "Esertia", "Meridian", "Humorap"]
        },
        "fluvoxamina": {
            "nombre_oficial": "Fluvoxamina",
            "atc_codigo": "N06AB08",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Antidepresivos",
            "atc_nivel4": "Inhibidores selectivos de la recaptación de serotonina (ISRS)",
            "marcas_comerciales": ["Luvox", "Dumirox", "Favarin"]
        },

        # --- Antidepresivos IRSN / Duales ---
        "venlafaxina": {
            "nombre_oficial": "Venlafaxina",
            "atc_codigo": "N06AX16",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Antidepresivos",
            "atc_nivel4": "Inhibidores de recaptación de serotonina y noradrenalina (IRSN)",
            "marcas_comerciales": ["Effexor", "Efexor XR", "Vandral", "Sesaren", "Elafax"]
        },
        "desvenlafaxina": {
            "nombre_oficial": "Desvenlafaxina",
            "atc_codigo": "N06AX23",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Antidepresivos",
            "atc_nivel4": "Inhibidores de recaptación de serotonina y noradrenalina (IRSN)",
            "marcas_comerciales": ["Pristiq", "Desirel", "Elleva"]
        },
        "duloxetina": {
            "nombre_oficial": "Duloxetina",
            "atc_codigo": "N06AX21",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Antidepresivos",
            "atc_nivel4": "Inhibidores de recaptación de serotonina y noradrenalina (IRSN)",
            "marcas_comerciales": ["Cymbalta", "Xeristar", "Duxetin", "Clobulitt"]
        },
        "milnacipram": {
            "nombre_oficial": "Milnacipram",
            "atc_codigo": "N06AX17",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Antidepresivos",
            "atc_nivel4": "Inhibidores de recaptación de serotonina y noradrenalina (IRSN)",
            "marcas_comerciales": ["Savella", "Ixel", "Dalcipran"]
        },

        # --- Otros Antidepresivos y Multimodales ---
        "bupropion": {
            "nombre_oficial": "Bupropión",
            "atc_codigo": "N06AX12",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Otros antidepresivos",
            "atc_nivel4": "Inhibidor de recaptación de noradrenalina y dopamina (IRND)",
            "marcas_comerciales": ["Wellbutrin", "Zyban", "Elontril", "Odranal"]
        },
        "bupropión": {
            "nombre_oficial": "Bupropión",
            "atc_codigo": "N06AX12",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Otros antidepresivos",
            "atc_nivel4": "Inhibidor de recaptación de noradrenalina y dopamina (IRND)",
            "marcas_comerciales": ["Wellbutrin", "Zyban", "Elontril", "Odranal"]
        },
        "mirtazapina": {
            "nombre_oficial": "Mirtazapina",
            "atc_codigo": "N06AX11",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Otros antidepresivos",
            "atc_nivel4": "Antagonista alfa-2 adrenérgico y 5-HT2/3 (NaSSA)",
            "marcas_comerciales": ["Remeron", "Rexer", "Comenter", "Calixta"]
        },
        "mianserina": {
            "nombre_oficial": "Mianserina",
            "atc_codigo": "N06AX03",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Otros antidepresivos",
            "atc_nivel4": "Antidepresivo tetracíclico",
            "marcas_comerciales": ["Tolvon", "Lantanon"]
        },
        "agomelatina": {
            "nombre_oficial": "Agomelatina",
            "atc_codigo": "N06AX22",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Otros antidepresivos",
            "atc_nivel4": "Agonista melatoninérgico MT1/MT2 y antagonista 5-HT2C (MASSA)",
            "marcas_comerciales": ["Valdoxan", "Thymanax"]
        },
        "vortioxetina": {
            "nombre_oficial": "Vortioxetina",
            "atc_codigo": "N06AX26",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Otros antidepresivos",
            "atc_nivel4": "Modulador y estimulador multimodal de serotonina",
            "marcas_comerciales": ["Brintellix", "Trintellix"]
        },
        "vilazodona": {
            "nombre_oficial": "Vilazodona",
            "atc_codigo": "N06AX27",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Otros antidepresivos",
            "atc_nivel4": "Inhibidor de recaptación de serotonina y agonista parcial 5-HT1A (SPARI)",
            "marcas_comerciales": ["Viibryd"]
        },
        "trazodona": {
            "nombre_oficial": "Trazodona",
            "atc_codigo": "N06AX05",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Otros antidepresivos",
            "atc_nivel4": "Antagonista 5-HT2A e inhibidor de recaptación (SARI)",
            "marcas_comerciales": ["Desyrel", "Deprax", "Taxagon"]
        },
        "tianeptina": {
            "nombre_oficial": "Tianeptina",
            "atc_codigo": "N06AX14",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Otros antidepresivos",
            "atc_nivel4": "Modulador glutamatérgico y facilitador de recaptación 5-HT",
            "marcas_comerciales": ["Stablon", "Coaxil"]
        },
        "reboxetina": {
            "nombre_oficial": "Reboxetina",
            "atc_codigo": "N06AX18",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Otros antidepresivos",
            "atc_nivel4": "Inhibidor selectivo de recaptación de noradrenalina (ISRNA)",
            "marcas_comerciales": ["Edronax", "Prolift"]
        },
        "atomoxetina": {
            "nombre_oficial": "Atomoxetina",
            "atc_codigo": "N06BA09",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Psicoestimulantes y fármacos para TDAH",
            "atc_nivel4": "Inhibidor selectivo de recaptación de noradrenalina",
            "marcas_comerciales": ["Strattera", "Deaten"]
        },

        # --- Antidepresivos Tricíclicos (ATC) ---
        "amitriptilina": {
            "nombre_oficial": "Amitriptilina",
            "atc_codigo": "N06AA09",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Antidepresivos",
            "atc_nivel4": "Inhibidores no selectivos de recaptación de monoaminas (Tricíclicos)",
            "marcas_comerciales": ["Elavil", "Tryptanol", "Triptizol", "Uxen"]
        },
        "clomipramina": {
            "nombre_oficial": "Clomipramina",
            "atc_codigo": "N06AA04",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Antidepresivos",
            "atc_nivel4": "Inhibidores no selectivos de recaptación de monoaminas (Tricíclicos)",
            "marcas_comerciales": ["Anafranil"]
        },
        "imipramina": {
            "nombre_oficial": "Imipramina",
            "atc_codigo": "N06AA02",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Antidepresivos",
            "atc_nivel4": "Inhibidores no selectivos de recaptación de monoaminas (Tricíclicos)",
            "marcas_comerciales": ["Tofranil"]
        },
        "nortriptilina": {
            "nombre_oficial": "Nortriptilina",
            "atc_codigo": "N06AA10",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Antidepresivos",
            "atc_nivel4": "Inhibidores no selectivos de recaptación de monoaminas (Tricíclicos)",
            "marcas_comerciales": ["Pamelor", "Aventyl", "Karile"]
        },

        # --- Antipsicóticos Atípicos (Segunda Generación) ---
        "clozapina": {
            "nombre_oficial": "Clozapina",
            "atc_codigo": "N05AH02",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Diazepinas, oxazepinas y tiazepinas (Atípico)",
            "marcas_comerciales": ["Leponex", "Clozaril", "Froidir", "Luverina", "Lapenax"]
        },
        "risperidona": {
            "nombre_oficial": "Risperidona",
            "atc_codigo": "N05AX08",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Otros antipsicóticos (Atípico)",
            "marcas_comerciales": ["Risperdal", "Ridal", "Sequinan", "Rispen"]
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
        "quetiapina": {
            "nombre_oficial": "Quetiapina",
            "atc_codigo": "N05AH04",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Diazepinas, oxazepinas y tiazepinas (Atípico)",
            "marcas_comerciales": ["Seroquel", "Quetiazic", "Rocoz", "Ketiap"]
        },
        "aripiprazol": {
            "nombre_oficial": "Aripiprazol",
            "atc_codigo": "N05AX12",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Otros antipsicóticos (Agonista parcial D2)",
            "marcas_comerciales": ["Abilify", "Ilimit", "Arizol"]
        },
        "ziprasidona": {
            "nombre_oficial": "Ziprasidona",
            "atc_codigo": "N05AE04",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Derivados de indol (Atípico)",
            "marcas_comerciales": ["Geodon", "Zeldox"]
        },
        "paliperidona": {
            "nombre_oficial": "Paliperidona",
            "atc_codigo": "N05AX13",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Otros antipsicóticos (Atípico / 9-OH-risperidona)",
            "marcas_comerciales": ["Invega", "Invega Sustenna", "Trevicta"]
        },
        "asenapina": {
            "nombre_oficial": "Asenapina",
            "atc_codigo": "N05AH05",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Diazepinas y oxazepinas (Atípico sublingual)",
            "marcas_comerciales": ["Saphris", "Sycrest"]
        },
        "lurasidona": {
            "nombre_oficial": "Lurasidona",
            "atc_codigo": "N05AE05",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Derivados de indol (Atípico)",
            "marcas_comerciales": ["Latuda"]
        },
        "cariprazina": {
            "nombre_oficial": "Cariprazina",
            "atc_codigo": "N05AX15",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Otros antipsicóticos (Agonista parcial D3/D2)",
            "marcas_comerciales": ["Vraylar", "Reagila"]
        },
        "brexpiprazol": {
            "nombre_oficial": "Brexpiprazol",
            "atc_codigo": "N05AX16",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Modulador de actividad serotonina-dopamina (SDAM)",
            "marcas_comerciales": ["Rexulti"]
        },

        # --- Antipsicóticos Típicos (Primera Generación) ---
        "haloperidol": {
            "nombre_oficial": "Haloperidol",
            "atc_codigo": "N05AD01",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Derivados de butirofenona (Típico / Primera generación)",
            "marcas_comerciales": ["Haldol", "Halopidol", "Haldol Decanoato"]
        },
        "clorpromazina": {
            "nombre_oficial": "Clorpromazina",
            "atc_codigo": "N05AA01",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Fenotiazinas con cadena alifática",
            "marcas_comerciales": ["Thorazine", "Largactil", "Ampliactil"]
        },
        "levomepromazina": {
            "nombre_oficial": "Levomepromazina",
            "atc_codigo": "N05AA02",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Fenotiazinas sedativas",
            "marcas_comerciales": ["Nozinan", "Sinogan"]
        },
        "trifluoperazina": {
            "nombre_oficial": "Trifluoperazina",
            "atc_codigo": "N05AB06",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Fenotiazinas con estructura piperazina",
            "marcas_comerciales": ["Stelazine", "Modalina"]
        },
        "pimozida": {
            "nombre_oficial": "Pimozida",
            "atc_codigo": "N05AG02",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Derivados de difenilbutilpiperidina",
            "marcas_comerciales": ["Orap"]
        },
        "zuclopentixol": {
            "nombre_oficial": "Zuclopentixol",
            "atc_codigo": "N05AF05",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos",
            "atc_nivel4": "Derivados de tioxanteno",
            "marcas_comerciales": ["Clopixol", "Cisordinol"]
        },

        # --- Estabilizadores del Ánimo y Antiepilépticos ---
        "litio": {
            "nombre_oficial": "Carbonato de Litio",
            "atc_codigo": "N05AN01",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos / Estabilizadores",
            "atc_nivel4": "Compuestos de litio",
            "marcas_comerciales": ["Carbolit", "Plenur", "Eskalith", "Lithobid", "Ceglution"]
        },
        "carbonato de litio": {
            "nombre_oficial": "Carbonato de Litio",
            "atc_codigo": "N05AN01",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Antipsicóticos / Estabilizadores",
            "atc_nivel4": "Compuestos de litio",
            "marcas_comerciales": ["Carbolit", "Plenur", "Eskalith", "Lithobid", "Ceglution"]
        },
        "valproato": {
            "nombre_oficial": "Ácido Valproico / Divalproato",
            "atc_codigo": "N03AG01",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Antiepilépticos (N03)",
            "atc_nivel3": "Antiepilépticos",
            "atc_nivel4": "Derivados de ácidos grasos",
            "marcas_comerciales": ["Depakene", "Depakote", "Valcote", "Depakote ER"]
        },
        "acido valproico": {
            "nombre_oficial": "Ácido Valproico / Divalproato",
            "atc_codigo": "N03AG01",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Antiepilépticos (N03)",
            "atc_nivel3": "Antiepilépticos",
            "atc_nivel4": "Derivados de ácidos grasos",
            "marcas_comerciales": ["Depakene", "Depakote", "Valcote", "Depakote ER"]
        },
        "divalproato": {
            "nombre_oficial": "Divalproato de Sodio",
            "atc_codigo": "N03AG01",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Antiepilépticos (N03)",
            "atc_nivel3": "Antiepilépticos",
            "atc_nivel4": "Derivados de ácidos grasos",
            "marcas_comerciales": ["Depakote", "Valcote", "Depakote ER"]
        },
        "lamotrigina": {
            "nombre_oficial": "Lamotrigina",
            "atc_codigo": "N03AX09",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Antiepilépticos (N03)",
            "atc_nivel3": "Otros antiepilépticos",
            "atc_nivel4": "Estabilizador del ánimo",
            "marcas_comerciales": ["Lamictal", "Labileno", "Lamotrig", "Crisomet"]
        },
        "carbamazepina": {
            "nombre_oficial": "Carbamazepina",
            "atc_codigo": "N03AF01",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Antiepilépticos (N03)",
            "atc_nivel3": "Antiepilépticos",
            "atc_nivel4": "Derivados de carboxamida",
            "marcas_comerciales": ["Tegretol", "Actinerval", "Carbatrol"]
        },
        "oxcarbazepina": {
            "nombre_oficial": "Oxcarbazepina",
            "atc_codigo": "N03AF02",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Antiepilépticos (N03)",
            "atc_nivel3": "Antiepilépticos",
            "atc_nivel4": "Derivados de carboxamida",
            "marcas_comerciales": ["Trileptal", "Oxcap"]
        },
        "topiramato": {
            "nombre_oficial": "Topiramato",
            "atc_codigo": "N03AX11",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Antiepilépticos (N03)",
            "atc_nivel3": "Otros antiepilépticos",
            "atc_nivel4": "Derivados de monosacáridos sulfamato",
            "marcas_comerciales": ["Topamax", "Topictal"]
        },
        "gabapentina": {
            "nombre_oficial": "Gabapentina",
            "atc_codigo": "N03AX12",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Antiepilépticos (N03)",
            "atc_nivel3": "Otros antiepilépticos",
            "atc_nivel4": "Ligando alfa-2-delta",
            "marcas_comerciales": ["Neurontin", "Labil"]
        },
        "pregabalina": {
            "nombre_oficial": "Pregabalina",
            "atc_codigo": "N03AX16",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Antiepilépticos (N03)",
            "atc_nivel3": "Otros antiepilépticos",
            "atc_nivel4": "Ligando alfa-2-delta / Ansiolítico",
            "marcas_comerciales": ["Lyrica", "Plenica", "Martimil"]
        },

        # --- Ansiolíticos / Benzodiazepinas ---
        "clonazepam": {
            "nombre_oficial": "Clonazepam",
            "atc_codigo": "N03AE01",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Antiepilépticos (N03)",
            "atc_nivel3": "Antiepilépticos / Ansiolíticos",
            "atc_nivel4": "Derivados de benzodiazepina",
            "marcas_comerciales": ["Rivotril", "Klonopin", "Neuryl", "Clonagin", "Diocam"]
        },
        "lorazepam": {
            "nombre_oficial": "Lorazepam",
            "atc_codigo": "N05BA06",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Ansiolíticos",
            "atc_nivel4": "Derivados de benzodiazepina",
            "marcas_comerciales": ["Ativan", "Trapax", "Orfidal", "Donix"]
        },
        "diazepam": {
            "nombre_oficial": "Diazepam",
            "atc_codigo": "N05BA01",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Ansiolíticos",
            "atc_nivel4": "Derivados de benzodiazepina",
            "marcas_comerciales": ["Valium", "Plidan", "Saromet"]
        },
        "alprazolam": {
            "nombre_oficial": "Alprazolam",
            "atc_codigo": "N05BA12",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Ansiolíticos",
            "atc_nivel4": "Derivados de benzodiazepina",
            "marcas_comerciales": ["Xanax", "Alplax", "Trankimazin", "Tafil"]
        },
        "bromazepam": {
            "nombre_oficial": "Bromazepam",
            "atc_codigo": "N05BA08",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Ansiolíticos",
            "atc_nivel4": "Derivados de benzodiazepina",
            "marcas_comerciales": ["Lexotanil", "Lexatin", "Octanyl"]
        },
        "clorazepato": {
            "nombre_oficial": "Clorazepato Dipotásico",
            "atc_codigo": "N05BA05",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Ansiolíticos",
            "atc_nivel4": "Derivados de benzodiazepina",
            "marcas_comerciales": ["Tranxilium", "Justum"]
        },
        "midazolam": {
            "nombre_oficial": "Midazolam",
            "atc_codigo": "N05CD08",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Hipnóticos y sedantes",
            "atc_nivel4": "Derivados de benzodiazepina",
            "marcas_comerciales": ["Dormicum", "Versed"]
        },
        "clobazam": {
            "nombre_oficial": "Clobazam",
            "atc_codigo": "N05BA09",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Ansiolíticos",
            "atc_nivel4": "Derivados de 1,5-benzodiazepina",
            "marcas_comerciales": ["Urbadan", "Frisium"]
        },

        # --- Hipnóticos y Otros Ansiolíticos ---
        "buspirona": {
            "nombre_oficial": "Buspirona",
            "atc_codigo": "N05BE01",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Ansiolíticos",
            "atc_nivel4": "Derivados de azaspirodecanediona (Agonista parcial 5-HT1A)",
            "marcas_comerciales": ["Buspar", "Bespar", "Parnate"]
        },
        "zolpidem": {
            "nombre_oficial": "Zolpidem",
            "atc_codigo": "N05CF02",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Hipnóticos y sedantes",
            "atc_nivel4": "Fármacos relacionados con benzodiazepinas (Imidazopiridinas)",
            "marcas_comerciales": ["Ambien", "Stilnox", "Somnil", "Noctem"]
        },
        "zopiclona": {
            "nombre_oficial": "Zopiclona",
            "atc_codigo": "N05CF01",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicolépticos (N05)",
            "atc_nivel3": "Hipnóticos y sedantes",
            "atc_nivel4": "Ciclopirrolonas",
            "marcas_comerciales": ["Imovane", "Datolan", "Folcodal"]
        },

        # --- Fármacos para Demencias y Deterioro Cognitivo ---
        "donepezilo": {
            "nombre_oficial": "Donepezilo",
            "atc_codigo": "N06DA02",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Fármacos contra la demencia",
            "atc_nivel4": "Anticolinesterásicos centrales",
            "marcas_comerciales": ["Aricept", "Donecept", "Eranz"]
        },
        "rivastigmina": {
            "nombre_oficial": "Rivastigmina",
            "atc_codigo": "N06DA03",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Fármacos contra la demencia",
            "atc_nivel4": "Inhibidor de acetilcolinesterasa y butirilcolinesterasa",
            "marcas_comerciales": ["Exelon", "Prometax"]
        },
        "galantamina": {
            "nombre_oficial": "Galantamina",
            "atc_codigo": "N06DA04",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Fármacos contra la demencia",
            "atc_nivel4": "Anticolinesterásico y modulador alostérico nicotínico",
            "marcas_comerciales": ["Reminyl", "Razadyne"]
        },
        "memantina": {
            "nombre_oficial": "Memantina",
            "atc_codigo": "N06DX01",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Psicoanalépticos (N06)",
            "atc_nivel3": "Fármacos contra la demencia",
            "atc_nivel4": "Antagonista no competitivo de receptores NMDA",
            "marcas_comerciales": ["Namenda", "Ebixa", "Akatinol"]
        },

        # --- Fármacos Coadyuvantes y Anticolinérgicos ---
        "biperideno": {
            "nombre_oficial": "Biperideno",
            "atc_codigo": "N04AA02",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Antiparkinsonianos (N04)",
            "atc_nivel3": "Agentes anticolinérgicos",
            "atc_nivel4": "Anticolinérgico terciario",
            "marcas_comerciales": ["Akineton"]
        },
        "trihexifenidilo": {
            "nombre_oficial": "Trihexifenidilo",
            "atc_codigo": "N04AA01",
            "atc_nivel1": "Sistema Nervioso (N)",
            "atc_nivel2": "Antiparkinsonianos (N04)",
            "atc_nivel3": "Agentes anticolinérgicos",
            "atc_nivel4": "Anticolinérgico terciario",
            "marcas_comerciales": ["Artane", "Parkison"]
        }
    }

    @classmethod
    def get_info(cls, nombre_farmaco: str) -> Optional[Dict[str, Any]]:
        """Busca información ATC para un fármaco dado su nombre común, oficial o marca comercial."""
        if not nombre_farmaco:
            return None
        clave = normalizar_clave(nombre_farmaco)
        if not clave:
            return None

        # 1. Coincidencia exacta con clave o nombre oficial
        for k, v in cls.CATALOGO_ATC.items():
            if clave == k or clave == normalizar_clave(v.get("nombre_oficial", "")):
                return v

        # 2. Coincidencia exacta con marca comercial
        for v in cls.CATALOGO_ATC.values():
            if any(clave == normalizar_clave(m) for m in v.get("marcas_comerciales", [])):
                return v

        # 3. Coincidencia parcial (solo con consultas de al menos MIN_LEN_BUSQUEDA caracteres)
        if len(clave) < cls.MIN_LEN_BUSQUEDA:
            return None
        for k, v in cls.CATALOGO_ATC.items():
            if clave in k or k in clave or clave in normalizar_clave(v.get("nombre_oficial", "")):
                return v
        for v in cls.CATALOGO_ATC.values():
            for marca in v.get("marcas_comerciales", []):
                norm_marca = normalizar_clave(marca)
                if norm_marca and (clave in norm_marca or norm_marca in clave):
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

        df_out["atc_codigo"] = df_out[drug_column].apply(_obtener_codigo)
        df_out["atc_grupo"] = df_out[drug_column].apply(_obtener_grupo)

        return df_out
