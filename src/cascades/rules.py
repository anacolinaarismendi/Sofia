"""Catálogo estructurado y validado de Cascadas de Prescripción en Salud Mental y Psicogeriatría.
Basado en:
- Dra. Gabriela Jufe: Psicofarmacología Práctica (4.ª Edición).
- Dr. Stephen Stahl: Prescriber's Guide (6.ª Edición).
- Criterios STOPP/START v3 (2023) para la prescripción potencialmente inapropiada en personas mayores.
- Consenso de la Asociación Americana de Diabetes (ADA) y Asociación Psiquiátrica Americana (APA).
- Anticholinergic Cognitive Burden (ACB) Scale (Boustani et al. / Campbell et al.).
"""

from typing import Dict, List, Any

# =============================================================================
# 1. ESCALA DE CARGA ANTICOLINÉRGICA (ANTICHOLINERGIC COGNITIVE BURDEN - ACB)
# =============================================================================
# Puntaje validado internacionalmente:
# Score 3: Anticolinérgico potente / definitivo (alto riesgo cognitivo y de delirio).
# Score 2: Anticolinérgico moderado.
# Score 1: Anticolinérgico leve / posible (actividad muscarínica in vitro).
# Score 0: Actividad anticolinérgica nula o no clínicamente significativa.
TABLA_CARGA_ANTICOLINERGICA_ACB: Dict[str, Dict[str, Any]] = {
    # Score 3: Potentes / Definitivos
    "biperideno": {"score": 3, "grupo": "Antiparkinsoniano anticolinérgico", "riesgo": "Deterioro neurocognitivo severo, confusión, íleo paralítico, retención urinaria"},
    "akineton": {"score": 3, "grupo": "Antiparkinsoniano anticolinérgico (marca)", "riesgo": "Deterioro neurocognitivo severo, confusión, íleo paralítico"},
    "trihexifenidilo": {"score": 3, "grupo": "Antiparkinsoniano anticolinérgico", "riesgo": "Toxicidad cognitiva, euforia/abuso, delirio"},
    "artane": {"score": 3, "grupo": "Antiparkinsoniano anticolinérgico (marca)", "riesgo": "Toxicidad cognitiva, euforia/abuso, delirio"},
    "oxibutinina": {"score": 3, "grupo": "Anticolinérgico urológico", "riesgo": "Antagonismo central frontal con inhibidores de colinesterasa, demencia acelerada"},
    "ditropan": {"score": 3, "grupo": "Anticolinérgico urológico (marca)", "riesgo": "Antagonismo central frontal con inhibidores de colinesterasa"},
    "tolterodina": {"score": 3, "grupo": "Anticolinérgico urológico", "riesgo": "Deterioro cognitivo, retención urinaria"},
    "solifenacina": {"score": 3, "grupo": "Anticolinérgico urológico", "riesgo": "Sequedad extrema, confusión en ancianos"},
    "clozapina": {"score": 3, "grupo": "Antipsicótico atípico", "riesgo": "Íleo paralítico obstructivo fatal, sedación masiva, retención urinaria"},
    "leponex": {"score": 3, "grupo": "Antipsicótico atípico (marca)", "riesgo": "Íleo paralítico obstructivo fatal"},
    "amitriptilina": {"score": 3, "grupo": "Antidepresivo tricíclico", "riesgo": "Bloqueo muscarínico severo, confusión, caídas, arritmias cardíacas"},
    "clomipramina": {"score": 3, "grupo": "Antidepresivo tricíclico", "riesgo": "Bloqueo muscarínico intenso, convulsiones, retención urinaria"},
    "nortriptilina": {"score": 3, "grupo": "Antidepresivo tricíclico", "riesgo": "Sequedad de boca, visión borrosa, constipación grave"},
    "imipramina": {"score": 3, "grupo": "Antidepresivo tricíclico", "riesgo": "Hipotensión ortostática, boca seca, delirio en ancianos"},
    "clorpromazina": {"score": 3, "grupo": "Antipsicótico típico de baja potencia", "riesgo": "Sedación marcada, retención urinaria, deterioro cognitivo"},
    "levomepromazina": {"score": 3, "grupo": "Antipsicótico típico sedante", "riesgo": "Hipotensión ortostática, íleo paralítico, deterioro cognitivo"},
    "hidroxizina": {"score": 3, "grupo": "Antihistamínico H1 de primera generación", "riesgo": "Sedación residual diurna, caídas en ancianos, deterioro de memoria"},
    "atarax": {"score": 3, "grupo": "Antihistamínico H1 (marca)", "riesgo": "Sedación residual diurna, caídas en ancianos"},
    "difenhidramina": {"score": 3, "grupo": "Antihistamínico H1 de primera generación", "riesgo": "Confusión nocturna, caídas, retención urinaria"},
    "doxilamina": {"score": 3, "grupo": "Antihistamínico hipnótico", "riesgo": "Sedación matutina prolongada, efecto anticolinérgico acumulativo"},

    # Score 2: Moderados
    "carbamazepina": {"score": 2, "grupo": "Antiepiléptico / Estabilizador", "riesgo": "Visión borrosa, sequedad, mareo, hiponatremia"},
    "tegretol": {"score": 2, "grupo": "Antiepiléptico (marca)", "riesgo": "Visión borrosa, sequedad, mareo"},
    "pimozida": {"score": 2, "grupo": "Antipsicótico", "riesgo": "Prolongación QTc, sequedad de mucosas"},

    # Score 1: Leves / Posibles
    "paroxetina": {"score": 1, "grupo": "Antidepresivo ISRS", "riesgo": "Ligero efecto anticolinérgico M1 (el ISRS más sedante y anticolinérgico)"},
    "olanzapina": {"score": 1, "grupo": "Antipsicótico atípico", "riesgo": "Afinidad M1 moderada, sequedad, constipación"},
    "zyprexa": {"score": 1, "grupo": "Antipsicótico atípico (marca)", "riesgo": "Afinidad M1 moderada, sequedad, constipación"},
    "quetiapina": {"score": 1, "grupo": "Antipsicótico atípico", "riesgo": "Sedación por H1 y afinidad M1 débil"},
    "seroquel": {"score": 1, "grupo": "Antipsicótico atípico (marca)", "riesgo": "Sedación por H1 y afinidad M1 débil"},
    "risperidona": {"score": 1, "grupo": "Antipsicótico atípico", "riesgo": "Baja afinidad muscarínica, pero contribuye a la carga total"},
    "haloperidol": {"score": 1, "grupo": "Antipsicótico típico de alta potencia", "riesgo": "Mínima acción anticolinérgica pero precipita uso de Biperideno"},
    "venlafaxina": {"score": 1, "grupo": "Antidepresivo IRSN", "riesgo": "Sequedad de boca y diaforesis autonómica"},
    "fluvoxamina": {"score": 1, "grupo": "Antidepresivo ISRS", "riesgo": "Náuseas y sequedad mucosa leve"},
    "diazepam": {"score": 1, "grupo": "Benzodiacepina", "riesgo": "Sedación acumulativa, ataxia, riesgo de caídas"},
    "lorazepam": {"score": 1, "grupo": "Benzodiacepina", "riesgo": "Sedación, relajación muscular, riesgo de caídas en ancianos"},
    "alprazolam": {"score": 1, "grupo": "Benzodiacepina", "riesgo": "Tolerancia, caídas, deterioro psicomotor"},
    "clonazepam": {"score": 1, "grupo": "Benzodiacepina", "riesgo": "Sedación diurna, ataxia, dependencia"},

    # Score 0: Nulo / Seguro a nivel colinérgico
    "sertralina": {"score": 0, "grupo": "Antidepresivo ISRS", "riesgo": "Sin carga anticolinérgica relevante"},
    "escitalopram": {"score": 0, "grupo": "Antidepresivo ISRS", "riesgo": "El ISRS más selectivo por SERT, nulo efecto anticolinérgico"},
    "citalopram": {"score": 0, "grupo": "Antidepresivo ISRS", "riesgo": "Nula actividad anticolinérgica"},
    "bupropion": {"score": 0, "grupo": "Antidepresivo IRND", "riesgo": "Nula afinidad muscarínica (ahorrador cognitivo y sexual)"},
    "mirtazapina": {"score": 0, "grupo": "Antidepresivo NaSSA", "riesgo": "Antagonista H1 pero nula afinidad muscarínica"},
    "vortioxetina": {"score": 0, "grupo": "Antidepresivo multimodal", "riesgo": "Procognitivo, nula actividad anticolinérgica"},
    "aripiprazol": {"score": 0, "grupo": "Antipsicótico atípico (agonista parcial D2)", "riesgo": "Nula afinidad muscarínica, ahorrador cognitivo"},
    "litio": {"score": 0, "grupo": "Estabilizador del ánimo", "riesgo": "Sin afinidad por receptores muscarínicos"},
    "lamotrigina": {"score": 0, "grupo": "Estabilizador / Antiepiléptico", "riesgo": "Sin actividad anticolinérgica"},
    "valproato": {"score": 0, "grupo": "Estabilizador / Antiepiléptico", "riesgo": "Sin afinidad muscarínica"},
    "donepezilo": {"score": 0, "grupo": "Inhibidor acetilcolinesterasa (procognitivo)", "riesgo": "Procolinérgico central (lo opuesto a anticolinérgico)"},
    "rivastigmina": {"score": 0, "grupo": "Inhibidor acetilcolinesterasa", "riesgo": "Procolinérgico central"}
}


# =============================================================================
# 2. CATÁLOGO COMPLETO DE CASCADAS CLÍNICAS (ESQUEMA COMPATIBLE STREAMLIT)
# =============================================================================
CASCADAS_CLINICAS_EXPANDIDAS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Haloperidol (Antipsicótico Típico)": {
        "Rigidez muscular, lentitud motora o temblor de reposo (Parkinsonismo secundario)": {
            "nivel": "CRÍTICO",
            "titulo": "🚨 ALERTA ROJA: CASCADA DE PRESCRIPCIÓN CLÁSICA (PARKINSONISMO → BIPERIDENO)",
            "farmaco_causal": "Haloperidol",
            "farmaco_cascada": "Biperideno / Trihexifenidilo",
            "doc_id": "CASC-HAL-BIP-001",
            "mecanismo": "Bloqueo masivo (>80%) de receptores dopaminérgicos D2 en la vía nigroestriatal, provocando una disfunción extrapiramidal idéntica a la enfermedad de Parkinson idiopática.",
            "error_tipico": "Prescribir Biperideno (Akineton) o Trihexifenidilo de forma crónica para enmascarar el temblor o la rigidez en lugar de optimizar la dosis del antipsicótico.",
            "riesgo_iatrogenico": "Elevada carga anticolinérgica agregada (ACB +3): deterioro neurocognitivo (memoria episódica y funciones ejecutivas), boca seca, estreñimiento grave / íleo paralítico, retención urinaria y riesgo elevado de inducir o desenmascarar discinesia tardía irreversible.",
            "protocolo": [
                "<b>Reducir la dosis</b> de Haloperidol a la menor dosis antipsicótica efectiva (habitualmente ≤ 2-4 mg/día).",
                "Si persisten los síntomas extrapiramidales o existe riesgo de recaída psicótica, <b>rotar</b> hacia un antipsicótico atípico con menor perfil extrapiramidal (ej. Quetiapina, Aripiprazol u Olanzapina).",
                "Si ya se inició Biperideno, realizar <b>desescalada y retirada gradual</b> (reduciendo 1-2 mg/semana a lo largo de 2-4 semanas) para evitar rebote colinérgico agudo y taquifilaxia."
            ],
            "fuente": "Jufe - Psicofarmacología Práctica, Stahl - Prescriber's Guide, Criterios STOPP/START v3"
        },
        "Inquietud motora imperiosa e incapacidad para permanecer sentado (Acatisia)": {
            "nivel": "ADVERTENCIA",
            "titulo": "⚠️ SOSPECHA DE ACATISIA FARMACOLÓGICA VS. AGITACIÓN PSICÓTICA",
            "farmaco_causal": "Haloperidol",
            "farmaco_cascada": "Aumento de antipsicótico o Benzodiacepinas crónicas",
            "doc_id": "CASC-PROT-003",
            "mecanismo": "Hipoactividad dopaminérgica en circuitos mesocorticales y estriatales inducida por el potente bloqueo D2.",
            "error_tipico": "Confundir la inquietud motora con empeoramiento de la psicosis o ansiedad y aumentar la dosis de Haloperidol, o instaurar benzodiacepinas a dosis altas a largo plazo.",
            "riesgo_iatrogenico": "Agravamiento paradójico de la angustia subjetiva e impulsividad motora, elevando el riesgo de conducta violenta o ideación autolítica.",
            "protocolo": [
                "<b>Disminuir la dosis</b> de Haloperidol de forma inmediata.",
                "Considerar un beta-bloqueante lipofílico a dosis bajas (<b>Propranolol 10-40 mg/día</b>) como coadyuvante sintomático transitorio si no hay contraindicación respiratoria o bradicardia.",
                "Rotar a un antipsicótico con menor potencial acatisiante si no cede tras el ajuste de dosis."
            ],
            "fuente": "Jufe - Psicofarmacología Práctica, Stahl"
        },
        "Contracciones espasmódicas agudas en cuello, mandíbula u ojos (Distonía aguda)": {
            "nivel": "CRÍTICO",
            "titulo": "🛑 EMERGENCIA EXTRAPIRAMIDAL AGUDA (DISTONÍA AGUDA O CRISIS OCULÓGIRA)",
            "farmaco_causal": "Haloperidol",
            "farmaco_cascada": "Biperideno crónico",
            "doc_id": "CASC-HAL-BIP-001",
            "mecanismo": "Desbalance agudo hipercolinérgico estriatal por bloqueo D2 intenso en fases tempranas o tras aumentos bruscos de posología.",
            "error_tipico": "Manejar la urgencia con anticolinérgicos parenterales pero dejar luego el Biperideno prescrito de forma crónica e indefinida durante meses.",
            "riesgo_iatrogenico": "Cronicidad inadvertida de la terapia anticolinérgica con toxicidad cognitiva y autonómica.",
            "protocolo": [
                "<b>Tratamiento agudo de rescate:</b> Biperideno 5 mg IM/IV lenta o benzodiacepina parenteral de acción rápida.",
                "<b>Reevaluar la posología:</b> Reducir dosis o ralentizar la titulación del antipsicótico.",
                "<b>Retirada planificada:</b> Suspender el anticolinérgico oral en pocas semanas y considerar la sustitución por un antipsicótico atípico."
            ],
            "fuente": "Jufe - Psicofarmacología Práctica, Stahl"
        }
    },
    "Risperidona (Antipsicótico Atípico)": {
        "Rigidez, lentitud motora o temblor dosis-dependiente": {
            "nivel": "CRÍTICO",
            "titulo": "🚨 CASCADA EXTRAPIRAMIDAL POR PÉRDIDA DE ATIPICIDAD (DOSIS-DEPENDIENTE)",
            "farmaco_causal": "Risperidona",
            "farmaco_cascada": "Biperideno / Trihexifenidilo",
            "doc_id": "CASC-HAL-BIP-001",
            "mecanismo": "A dosis superiores a 4-6 mg/día, la Risperidona sobrepasa el 80% de ocupación de receptores D2 en el estriado, perdiendo la ventaja protectora de su antagonismo 5-HT2A y comportándose clínicamente como un neuroléptico típico.",
            "error_tipico": "Añadir Biperideno u otros anticolinérgicos en lugar de retornar la Risperidona a su rango de atipicidad terapéutica.",
            "riesgo_iatrogenico": "Deterioro cognitivo, síndrome confusional en pacientes vulnerables y riesgo incrementado de discinesia tardía.",
            "protocolo": [
                "<b>Reducir la dosis</b> a un intervalo óptimo de atipicidad (habitualmente 2-4 mg/día).",
                "Si se requiere mayor potencia antipsicótica sin efectos motores, valorar la rotación a otro antipsicótico atípico (ej. Aripiprazol o Quetiapina).",
                "Evitar la prescripción crónica profiláctica de anticolinérgicos en combinación con antipsicóticos de segunda generación."
            ],
            "fuente": "Jufe - Psicofarmacología Práctica, Stahl"
        },
        "Galactorrea, amenorrea o ginecomastia (Hiperprolactinemia marcada)": {
            "nivel": "ADVERTENCIA",
            "titulo": "⚠️ CASCADA ENDOCRINA: HIPERPROLACTINEMIA Y RIESGO DE AGONISTAS DOPAMINÉRGICOS",
            "farmaco_causal": "Risperidona",
            "farmaco_cascada": "Cabergolina / Bromocriptina",
            "doc_id": "MONO-RISPERIDONA-007",
            "mecanismo": "Bloqueo D2 en las células lactotrópicas de la adenohipófisis (vía tuberoinfundibular), zona carente de barrera hematoencefálica estricta.",
            "error_tipico": "Derivar a endocrinología y prescribir agonistas dopaminérgicos (Cabergolina, Bromocriptina), lo que compite frontalmente con la acción antipsicótica y reactiva los síntomas psicóticos.",
            "riesgo_iatrogenico": "Reagudización psicótica por competición D2 mesolímbica; desmineralización ósea acelerada por hipogonadismo crónico.",
            "protocolo": [
                "Medir prolactina sérica basal y descartar prolactinoma orgánico si los niveles son extremos (>150-200 ng/mL).",
                "<b>Estrategia de elección:</b> Coadyuvar con una dosis baja de <b>Aripiprazol (5-10 mg/día)</b>; su agonismo parcial D2 hipofisario rescata el tono inhibitorio y normaliza la prolactina sin descompensar la psicosis.",
                "Si persiste, rotar hacia antipsicóticos ahorradores de prolactina (Olanzapina, Quetiapina o Aripiprazol en monoterapia)."
            ],
            "fuente": "Consensos ADA/APA y Guías de Endocrinología, Stahl"
        },
        "Aumento de peso progresivo y resistencia a la insulina": {
            "nivel": "ADVERTENCIA",
            "titulo": "⚠️ ALTERACIÓN METABÓLICA Y GANANCIA PONDERAL",
            "farmaco_causal": "Risperidona",
            "farmaco_cascada": "Hipolipemiantes / Antidiabéticos automáticos",
            "doc_id": "CASC-METAB-002",
            "mecanismo": "Antagonismo moderado de receptores 5-HT2C e histaminérgicos H1 que estimula el centro hipotalámico del apetito.",
            "error_tipico": "Iniciar fármacos hipolipemiantes o antidiabéticos sin intervenir precozmente en la dieta o valorar alternativas metabólicamente neutras.",
            "riesgo_iatrogenico": "Desarrollo progresivo de síndrome metabólico y aumento del riesgo cardiovascular global.",
            "protocolo": [
                "Registrar peso, perímetro abdominal, glucemia y perfil lipídico basal y a las 4-8 semanas.",
                "Si la ganancia de peso supera el 5% respecto al basal en el primer mes, plantear rotación a antipsicóticos de bajo impacto metabólico (Aripiprazol, Lurasidona o Ziprasidona).",
                "Acompañar siempre de asesoramiento nutricional y actividad física."
            ],
            "fuente": "Consenso ADA/APA, Jufe"
        }
    },
    "Olanzapina (Antipsicótico Atípico)": {
        "Aumento de peso masivo (>5% basal), hiperfagia y dislipidemia": {
            "nivel": "CRÍTICO",
            "titulo": "🚨 CASCADA METABÓLICA: DE OLANZAPINA A POLIFARMACIA CARDIOMETABÓLICA",
            "farmaco_causal": "Olanzapina",
            "farmaco_cascada": "Metformina / Estatinas / Fibratos",
            "doc_id": "CASC-METAB-002",
            "mecanismo": "Potente bloqueo histaminérgico H1 que activa la AMPK hipotalámica y antagonismo 5-HT2C que anula las señales de saciedad mediadas por leptina y POMC, induciendo apetencia compulsiva por hidratos de carbono.",
            "error_tipico": "Prescribir estatinas y fibratos de forma automática para la dislipidemia resultante sin replantear el tratamiento antipsicótico causante.",
            "riesgo_iatrogenico": "Obesidad grado II-III, esteatosis hepática metabólica, cardiopatía isquémica prematura y polifarmacia cardiovascular.",
            "protocolo": [
                "<b>Rotación estratégica:</b> Si la estabilidad psicopatológica lo consiente, rotar a antipsicóticos metabólicamente favorables (Aripiprazol, Lurasidona, Cariprazina o Ziprasidona).",
                "<b>Coadyuvancia con Metformina:</b> Si Olanzapina es insustituible por eficacia clínica exclusiva, iniciar precozmente Metformina (500 a 850 mg c/12h) para mitigar la lipogénesis y mejorar la sensibilidad a la insulina.",
                "Controles analíticos estrictos: glucemia y lípidos basales, al mes, a los 3 meses y posteriormente de forma anual."
            ],
            "fuente": "Consenso ADA/APA sobre Antipsicóticos y Diabetes, Jufe, Stahl"
        },
        "Hiperglucemia en ayunas o elevación de HbA1c (Resistencia a la insulina)": {
            "nivel": "CRÍTICO",
            "titulo": "🚨 DISMETABOLISMO GRAVE: RESISTENCIA A LA INSULINA POR BLOQUEO M3 PANCREÁTICO",
            "farmaco_causal": "Olanzapina",
            "farmaco_cascada": "Antidiabéticos orales / Insulina",
            "doc_id": "CASC-METAB-002",
            "mecanismo": "El antagonismo de receptores muscarínicos M3 en las células beta de los islotes de Langerhans reduce directamente la secreción aguda de insulina inducida por glucosa, independientemente de la masa adiposa.",
            "error_tipico": "Etiquetar al paciente con diabetes mellitus tipo 2 idiopática e iniciar antidiabéticos orales o insulina sin cuestionar la Olanzapina.",
            "riesgo_iatrogenico": "Diabetes inducida por fármacos de difícil control, episodios de cetoacidosis atípica y daño micro/macrovascular.",
            "protocolo": [
                "Considerar de forma prioritaria la desescalada o rotación a un antipsicótico con nula afinidad M3 (ej. Aripiprazol).",
                "Si la sustitución no es viable, coordinar con endocrinología el inicio de agentes sensibilizantes a la insulina (Metformina) o análogos GLP-1.",
                "Evitar fármacos que promuevan mayor ganancia ponderal (como sulfonilureas) siempre que sea posible."
            ],
            "fuente": "Jufe - Psicofarmacología Práctica, Consenso ADA/APA"
        },
        "Sedación diurna profunda y somnolencia matutina": {
            "nivel": "ADVERTENCIA",
            "titulo": "⚠️ SEDACIÓN CENTRAL POR BLOQUEO HISTAMINÉRGICO H1",
            "farmaco_causal": "Olanzapina",
            "farmaco_cascada": "Psicoestimulantes / Cafeína abusiva",
            "doc_id": "CASC-PROT-003",
            "mecanismo": "Afinidad nanomolar por receptores histaminérgicos H1 y muscarínicos M1 en el sistema nervioso central.",
            "error_tipico": "Prescribir psicoestimulantes (Metilfenidato, Modafinilo) o consumo abusivo de cafeína/bebidas energéticas.",
            "riesgo_iatrogenico": "Riesgo de descompensación psicótica o arritmias por interacción entre estimulantes y antipsicóticos.",
            "protocolo": [
                "Administrar la dosis total diaria en una toma única nocturna, 1 a 2 horas antes de dormir.",
                "Reducir ligeramente la dosis si el estado clínico lo permite.",
                "Si la interferencia sociolaboral es severa, rotar hacia un atípico activador."
            ],
            "fuente": "Stahl - Prescriber's Guide, Jufe"
        }
    },
    "Clozapina (Antipsicótico Atípico)": {
        "Sialorrea nocturna profusa (hipersalivación y babeo en almohada)": {
            "nivel": "CRÍTICO",
            "titulo": "⚠️ SIALORREA PARADÓJICA POR AGONISMO MUSCARÍNICO M4",
            "farmaco_causal": "Clozapina",
            "farmaco_cascada": "Biperideno / Trihexifenidilo oral",
            "doc_id": "MONO-CLOZAPINA-006",
            "mecanismo": "Agonismo selectivo sobre receptores muscarínicos colinérgicos M4 periféricos y centrales, asociado a inhibición del reflejo deglutorio durante las fases profundas del sueño.",
            "error_tipico": "Prescribir anticolinérgicos sistémicos de alta potencia por vía oral (Biperideno o Trihexifenidilo) para frenar la salivación.",
            "riesgo_iatrogenico": "¡ALTO PELIGRO DE ÍLEO PARALÍTICO! La adición de anticolinérgicos sistémicos a la Clozapina paraliza la motilidad cólica provocando constipación severa, obstrucción intestinal potencialmente letal y agravamiento cognitivo.",
            "protocolo": [
                "<b>NUNCA prescribir anticolinérgicos orales sistémicos de forma concomitante con Clozapina.</b>",
                "<b>Tratamiento local de elección:</b> Ipratropio tópico en spray sublingual (1-2 pulverizaciones antes de dormir) o colirio de atropina al 1% sublingual (1-2 gotas); actúan localmente sin toxicidad intestinal sistémica.",
                "Medidas conductuales: colocar una toalla protectora en la almohada y elevar la cabecera de la cama 30 grados para favorecer la deglución espontánea."
            ],
            "fuente": "Jufe - Psicofarmacología Práctica, Stahl"
        },
        "Estreñimiento severo / hipomotilidad colónica (Riesgo de íleo paralítico)": {
            "nivel": "CRÍTICO",
            "titulo": "🛑 ALERTA MÁXIMA DE FARMACOVIGILANCIA: HIPOMOTILIDAD GASTROINTESTINAL GRAVE",
            "farmaco_causal": "Clozapina",
            "farmaco_cascada": "Laxantes no supervisados / Anticolinérgicos añadidos",
            "doc_id": "MONO-CLOZAPINA-006",
            "mecanismo": "Potente antagonismo muscarínico colinérgico y antiserotoninérgico entérico que reduce drásticamente el peristaltismo gastrointestinal.",
            "error_tipico": "Subestimar el estreñimiento, considerarlo banal o tratarlo únicamente con aumento de fibra dietética sin laxantes osmóticos activos.",
            "riesgo_iatrogenico": "Megacolon tóxico, fecaloma obstructivo, perforación intestinal, peritonitis estercorácea y shock séptico con mortalidad superior al 30% (mortalidad que supera a la de la agranulocitosis).",
            "protocolo": [
                "<b>Monitorización obligatoria:</b> Interrogar el hábito defecatorio en cada consulta médica (si transcurren >48 horas sin deposición, intervenir activamente).",
                "Instaurar de inmediato laxantes osmóticos (Macrogol / Polietilenglicol) y pauta de hidratación oral abundante.",
                "Evitar terminantemente la adición de cualquier otro fármaco con propiedades anticolinérgicas."
            ],
            "fuente": "Alertas Sanitarias de Farmacovigilancia, Stahl, Jufe"
        },
        "Aumento ponderal severo y resistencia insulínica en esquizofrenia refractaria": {
            "nivel": "CRÍTICO",
            "titulo": "🚨 RIESGO CARDIOMETABÓLICO EN PACIENTE REFRACTARIO",
            "farmaco_causal": "Clozapina",
            "farmaco_cascada": "Suspensión brusca vs. polifarmacia no guiada",
            "doc_id": "CASC-METAB-002",
            "mecanismo": "Estimulación central intensa del apetito por bloqueo H1 y alteración directa de la secreción pancreática de insulina.",
            "error_tipico": "Suspender bruscamente la Clozapina (provocando tormenta colinérgica y recaída psicótica masiva) o descuidar la monitorización analítica.",
            "riesgo_iatrogenico": "Recaída psicótica intratable por retirada vs. morbilidad cardiovascular grave por abandono metabólico.",
            "protocolo": [
                "Como Clozapina es el único fármaco con indicación aprobada en resistencia, la rotación a menudo es inviable.",
                "Estrategia de referencia: Coadyuvancia precoz con <b>Metformina (850-1000 mg c/12h)</b> o añadir <b>Aripiprazol a dosis bajas (5-10 mg/día)</b> para frenar la ganancia de peso.",
                "Revisión trimestral de lípidos, glucosa basal y HbA1c."
            ],
            "fuente": "Consenso ADA/APA, Jufe, Stahl"
        }
    },
    "Litio (Carbonato de Litio)": {
        "Poliuria marcada (>3 L/día) y polidipsia intensa (Diabetes insípida nefrogénica)": {
            "nivel": "CRÍTICO",
            "titulo": "🛑 ALERTA ROJA: DIABETES INSÍPIDA NEFROGÉNICA Y RIESGO LETAL DE TIAZIDAS",
            "farmaco_causal": "Litio (Carbonato de Litio)",
            "farmaco_cascada": "Diuréticos Tiazídicos (Hidroclorotiazida / Clortalidona)",
            "doc_id": "INTER-LIT-RENAL-004",
            "mecanismo": "El litio penetra a las células principales del túbulo colector renal a través de los canales epiteliales de sodio (ENaC) e inhibe la adenilil ciclasa y la GSK-3beta, bloqueando la translocación de acuaporina-2 mediada por vasopresina (ADH).",
            "error_tipico": "¡Prescribir diuréticos tiazídicos (Hidroclorotiazida, Clortalidona) para la hipertensión o retención hidrosalina sin ajustar el Litio!",
            "riesgo_iatrogenico": "Las tiazidas inducen natriuresis distal compensada con hiperreabsorción de sodio y LITIO en el túbulo proximal. Esto eleva la litemia un 40-70% provocando INTOXICACIÓN GRAVE POR LITIO POTENCIALMENTE MORTAL (ataxia, convulsiones, arritmias y fracaso renal agudo).",
            "protocolo": [
                "Medir de inmediato litemia en ayunas, creatinina, ionograma plasmático y volumen urinario de 24 horas.",
                "Si la poliuria es severa y el litio es imprescindible, el fármaco de elección es <b>Amilorida (5-10 mg/día)</b>: bloquea los canales ENaC impidiendo la acumulación de litio en el túbulo renal sin elevar la litemia.",
                "Si un diurético tiazídico fuera imprescindible por causa cardiovascular mayor, la dosis de litio DEBE REDUCIRSE al 50% con determinaciones de litemia cada 48-72 horas hasta estabilizar niveles."
            ],
            "fuente": "Jufe - Psicofarmacología Práctica, Stahl - Prescriber's Guide"
        },
        "Temblor fino postural de manos (persistente o agravado)": {
            "nivel": "ADVERTENCIA",
            "titulo": "⚠️ CASCADA: TEMBLOR POR LITIO → PROPRANOLOL SIN MEDIR LITEMIA",
            "farmaco_causal": "Litio (Carbonato de Litio)",
            "farmaco_cascada": "Propranolol sin litemia",
            "doc_id": "MONO-LITIO-005",
            "mecanismo": "Efecto electrofisiológico directo del catión litio sobre la excitabilidad y sincronización neuromuscular periférica.",
            "error_tipico": "Prescribir Propranolol u otros betabloqueantes de forma automática sin verificar si el temblor es el primer síntoma de sobredosis o litemia tóxica en ascenso.",
            "riesgo_iatrogenico": "Enmascaramiento de toxicidad renal/sistémica por litio; agravamiento de broncoespasmo en pacientes con hiperreactividad bronquial o bradicardia sintomática.",
            "protocolo": [
                "<b>Paso 1 obligado:</b> Determinar litemia sérica a las 12 horas de la última dosis nocturna.",
                "Si la litemia supera 0.8-1.0 mEq/L, ajustar la dosis a la baja hasta el rango terapéutico óptimo (0.6-0.8 mEq/L).",
                "Moderar el consumo de cafeína (las fluctuaciones agudas alteran la excreción renal de litio).",
                "Solo si la litemia es normal y el temblor causa interferencia funcional grave, añadir Propranolol a dosis bajas (10-20 mg c/8-12h) tras verificar ausencia de contraindicaciones respiratorias o cardíacas."
            ],
            "fuente": "Jufe - Psicofarmacología Práctica, Stahl"
        },
        "Astenia, aumento de peso y friolera (Hipotiroidismo farmacológico)": {
            "nivel": "SEGURO",
            "titulo": "ℹ️ MANEJO RACIONAL: HIPOTIROIDISMO INDUCIDO POR LITIO (NO SUSPENDER EL LITIO)",
            "farmaco_causal": "Litio (Carbonato de Litio)",
            "farmaco_cascada": "Suspensión inapropiada de litio",
            "doc_id": "MONO-LITIO-005",
            "mecanismo": "El litio interfiere en la captación tiroidea de yoduro, bloquea el acoplamiento de tirosinas y frena la liberación periférica de T3 y T4.",
            "error_tipico": "Suspender apresuradamente el carbonato de litio ante el hallazgo de TSH elevada, perdiendo la estabilidad afectiva en un paciente bipolar previamente compensado.",
            "riesgo_iatrogenico": "Recaída grave en manía o depresión bipolar con riesgo elevado de conducta suicida.",
            "protocolo": [
                "Solicitar analítica completa con TSH y T4 libre.",
                "<b>NO suspender el carbonato de litio:</b> el hipotiroidismo secundario se maneja de forma segura iniciando terapia sustitutiva con Levotiroxina oral a dosis estándar (25-50 mcg/día inicialmente).",
                "Titular la dosis de levotiroxina según niveles de TSH cada 6-8 semanas manteniendo el litio en su rango terapéutico."
            ],
            "fuente": "Guías CANMAT para Trastorno Bipolar, Jufe"
        }
    },
    "Paroxetina (ISRS)": {
        "Disfunción sexual (anorgasmia, retardo eyaculatorio o caída de la libido)": {
            "nivel": "ADVERTENCIA",
            "titulo": "⚠️ CASCADA: DISFUNCIÓN SEXUAL POR ISRS → SILDENAFILO / TADALAFILO",
            "farmaco_causal": "Paroxetina",
            "farmaco_cascada": "Sildenafilo / Tadalafilo",
            "doc_id": "MONO-PAROXETINA-003",
            "mecanismo": "La hiperactivación de receptores 5-HT2A y 5-HT2C inducida por el bloqueo de recaptación de serotonina inhibe la liberación de dopamina en el área mesolímbica y suprime la vía del óxido nítrico genital y espinal.",
            "error_tipico": "Recetar de forma rutinaria inhibidores de la fosfodiesterasa-5 (Sildenafilo/Tadalafilo) sin investigar alternativas de optimización del antidepresivo.",
            "riesgo_iatrogenico": "Gasto económico elevado y falta de eficacia sobre la anorgasmia o la pérdida de deseo (el sildenafilo únicamente actúa sobre la erección hemodinámica, no sobre los receptores centrales de la libido).",
            "protocolo": [
                "Evaluar la reducción de la dosis de Paroxetina a la dosis mínima eficaz.",
                "<b>Rotación estratégica:</b> Rotar hacia antidepresivos con mecanismo ahorrador de función sexual: <b>Bupropión</b> (inhibidor de recaptación de dopamina/noradrenalina), <b>Vortioxetina</b> (modulador multimodal) o <b>Mirtazapina</b>.",
                "<b>Coadyuvancia farmacológica:</b> Si el paciente no puede cambiar de antidepresivo, añadir <b>Bupropión (150 mg/día)</b> como modulador dopaminérgico para revertir la disfunción sexual inducida por el ISRS."
            ],
            "fuente": "Jufe - Psicofarmacología Práctica, Stahl - Prescriber's Guide"
        },
        "Mareos, descargas eléctricas cefálicas ('brain zaps') y náuseas por olvido de dosis": {
            "nivel": "ADVERTENCIA",
            "titulo": "⚠️ SÍNDROME DE DISCONTINUACIÓN INTENSO POR VIDA MEDIA CORTA",
            "farmaco_causal": "Paroxetina",
            "farmaco_cascada": "Escalada de dosis o Benzodiacepinas",
            "doc_id": "MONO-PAROXETINA-003",
            "mecanismo": "La Paroxetina carece de metabolitos activos y posee una vida media corta (aprox. 21-24h) con alta afinidad por SERT, originando una brusca desensibilización serotoninérgica y un rebote colinérgico al retrasar una sola dosis.",
            "error_tipico": "Interpretar erróneamente los síntomas de abstinencia/discontinuación como una 'recaída fulminante del cuadro depresivo o ansioso' y subir la dosis o añadir ansiolíticos.",
            "riesgo_iatrogenico": "Polifarmacia con benzodiacepinas, retraso diagnóstico y dependencia psicológica severa al fármaco.",
            "protocolo": [
                "Reinstaurar la dosis habitual de Paroxetina para confirmar el cese inmediato del malestar (prueba diagnóstica).",
                "Si el objetivo clínico es retirar el antidepresivo, realizar una desescalada extraordinariamente lenta (ej. reducciones de 2.5-5 mg cada 2-4 semanas) o sustituir transitoriamente por Fluoxetina (por su prolongada vida media con norfluoxetina de 7 a 15 días)."
            ],
            "fuente": "Jufe - Psicofarmacología Práctica, Stahl"
        }
    },
    "Fluoxetina (ISRS)": {
        "Inquietud interna, ansiedad de inicio, insomnio y temblor en las primeras 2 semanas": {
            "nivel": "ADVERTENCIA",
            "titulo": "⚠️ CASCADA DE ANSIEDAD INICIAL → BENZODIACEPINAS CRÓNICAS",
            "farmaco_causal": "Fluoxetina",
            "farmaco_cascada": "Benzodiacepinas a largo plazo",
            "doc_id": "MONO-FLUOXETINA-002",
            "mecanismo": "El bloqueo 5-HT2C característico de la fluoxetina y el aumento agudo de serotonina en la amígdala provocan una estimulación noradrenérgica y dopaminérgica transitoria durante los primeros días de tratamiento.",
            "error_tipico": "Pautar benzodiacepinas potentes a dosis altas sin plan de retirada, cronificando el ansiolítico durante meses o años.",
            "riesgo_iatrogenico": "Dependencia física a benzodiacepinas, sedación residual, deterioro cognitivo y riesgo de caídas con fractura en adultos mayores.",
            "protocolo": [
                "<b>Información preventiva previa:</b> Advertir al paciente que la activación inicial es benigna y transitoria (habitualmente cede entre los 7 y 14 días).",
                "Comenzar con dosis reducidas (10 mg/día durante los primeros 4 a 7 días antes de subir a 20 mg/día).",
                "Si la ansiedad o el insomnio son intensos, utilizar benzodiacepinas a dosis mínimas durante un máximo estricto de 7 a 14 días como puente terapéutico, con fecha pactada de suspensión."
            ],
            "fuente": "Jufe - Psicofarmacología Práctica, Criterios STOPP/START"
        },
        "Pérdida marcada de apetito y náuseas tempranas": {
            "nivel": "ADVERTENCIA",
            "titulo": "⚠️ ACTIVACIÓN SEROTONINÉRGICA GASTROINTESTINAL TEMPRANA",
            "farmaco_causal": "Fluoxetina",
            "farmaco_cascada": "Procinéticos / Inhibidores de bomba de protones innecesarios",
            "doc_id": "MONO-FLUOXETINA-002",
            "mecanismo": "Estimulación de receptores 5-HT3 y 5-HT4 en el plexo mientérico digestivo que acelera el vaciamiento gástrico y genera náuseas agudas.",
            "error_tipico": "Iniciar inhibidores de la bomba de protones (Omeprazol) o procinéticos de forma prolongada sin relacionarlo con el inicio del antidepresivo.",
            "riesgo_iatrogenico": "Polifarmacia innecesaria e interacción farmacocinética de omeprazol (inhibidor CYP2C19).",
            "protocolo": [
                "Indicar la toma de Fluoxetina siempre acompañada de alimentos sólidos.",
                "Explicar que las náuseas se atenúan de forma espontánea tras las primeras 2 semanas por desensibilización de receptores 5-HT3.",
                "Evitar añadir fármacos gastroprotectores si no existen factores de riesgo de úlcera péptica."
            ],
            "fuente": "Jufe - Psicofarmacología Práctica"
        }
    },
    "Donepezilo (Inhibidor de Acetilcolinesterasa)": {
        "Incontinencia urinaria de urgencia y polaquiuria": {
            "nivel": "CRÍTICO",
            "titulo": "🚨 CASCADA FARMACOLÓGICA NEFRO-UROLÓGICA: DONEPEZILO → OXIBUTININA",
            "farmaco_causal": "Donepezilo",
            "farmaco_cascada": "Oxibutinina / Tolterodina / Solifenacina",
            "doc_id": "CASC-PROT-003",
            "mecanismo": "El incremento periférico de acetilcolina estimula los receptores muscarínicos colinérgicos M2 y M3 en el músculo detrusor de la vejiga, provocando contracciones espasmódicas involuntarias e incontinencia de urgencia.",
            "error_tipico": "Derivar a urología y prescribir anticolinérgicos urológicos como Oxibutinina, Tolterodina o Solifenacina para tratar la incontinencia.",
            "riesgo_iatrogenico": "¡ANTAGONISMO FARMACOLÓGICO FRONTAL! Los anticolinérgicos urológicos atraviesan la barrera hematoencefálica y BLOQUEAN los receptores colinérgicos cerebrales, aniquilando el beneficio cognitivo del Donepezilo y acelerando el deterioro de la demencia y el delirio confusional (ACB +3).",
            "protocolo": [
                "<b>Desprescribir de inmediato</b> cualquier fármaco anticolinérgico urológico (Oxibutinina, Tolterodina).",
                "Reconocer la incontinencia como un efecto adverso colinérgico del inhibidor de acetilcolinesterasa.",
                "Reducir la dosis de Donepezilo (de 10 mg a 5 mg/día) o evaluar cambio a parches transdérmicos de Rivastigmina (menor pico plasmático periférico).",
                "Si el tratamiento urológico es estrictamente ineludible, utilizar un agonista de receptores beta-3 adrenérgicos (<b>Mirabegrón</b>), el cual relaja el detrusor vesical sin ninguna actividad anticolinérgica en el sistema nervioso central."
            ],
            "fuente": "Criterios STOPP/START para Pacientes Ancianos, Stahl, Jufe"
        },
        "Bradicardia sintomática, mareos ortostáticos o síncope": {
            "nivel": "CRÍTICO",
            "titulo": "🛑 RIESGO CARDIOLÓGICO: BRADICARDIA COLINÉRGICA Y SÍNCOPE VAGAL",
            "farmaco_causal": "Donepezilo",
            "farmaco_cascada": "Antiagregantes / Implante de marcapasos sin retirar fármaco",
            "doc_id": "CASC-PROT-003",
            "mecanismo": "El exceso colinérgico estimula el tono vagal sobre los receptores muscarínicos M2 del nódulo sinoauricular y auriculoventricular cardíaco.",
            "error_tipico": "Interpretar el síncope como un 'accidente cerebrovascular isquémico transitorio' o una 'caída accidental del anciano' e iniciar antiagregantes o implante de marcapasos sin retirar el fármaco.",
            "riesgo_iatrogenico": "Caídas graves, fracturas de cadera, traumatismo craneoencefálico e intervenciones invasivas innecesarias.",
            "protocolo": [
                "Realizar electrocardiograma basal para evaluar frecuencia cardíaca y descartar bloqueo auriculoventricular o intervalo PR prolongado.",
                "Suspender o reducir la dosis de Donepezilo si la frecuencia cardíaca desciende de 50 lpm o existen síntomas presincopales.",
                "Evitar la combinación con otros bradicardizantes como betabloqueantes o calcioantagonistas no dihidropiridínicos (Verapamilo, Diltiazem)."
            ],
            "fuente": "Jufe, Stahl, Criterios STOPP/START"
        }
    },
    "Venlafaxina (IRSN)": {
        "Elevación de la presión arterial (hipertensión arterial diastólica dosis-dependiente)": {
            "nivel": "ADVERTENCIA",
            "titulo": "⚠️ CASCADA CARDIOVASCULAR: HIPERTENSIÓN POR IRSN → ANTIHIPERTENSIVOS",
            "farmaco_causal": "Venlafaxina",
            "farmaco_cascada": "Antihipertensivos automáticos (Enalapril / Amlodipino)",
            "doc_id": "CASC-PROT-003",
            "mecanismo": "A dosis medias y altas (>150-225 mg/día), la inhibición del transportador de noradrenalina (NET) incrementa el tono simpático vascular periférico y la resistencia vascular.",
            "error_tipico": "Iniciar nuevos fármacos antihipertensivos o aumentar su dosis sin relacionar el empeoramiento tensional con la escalada de Venlafaxina.",
            "riesgo_iatrogenico": "Polifarmacia cardiovascular, hipotensión ortostática por fármacos combinados e hipertrofia ventricular inadvertida.",
            "protocolo": [
                "Monitorizar la presión arterial antes y durante el tratamiento con Venlafaxina en cada escalón posológico.",
                "Si la PA diastólica aumenta >10 mmHg o supera 90 mmHg, <b>reducir la dosis a ≤150 mg/día</b>.",
                "Si persiste la hipertensión descontrolada, <b>rotar hacia un antidepresivo sin efecto noradrenérgico periférico</b> (Sertralina, Escitalopram o Vortioxetina)."
            ],
            "fuente": "Jufe - Psicofarmacología Práctica, Stahl - Prescriber's Guide"
        }
    },
    "Quetiapina (Antipsicótico Atípico)": {
        "Hipotensión ortostática, mareos posturales y taquicardia refleja": {
            "nivel": "ADVERTENCIA",
            "titulo": "⚠️ CASCADA HEMODINÁMICA: HIPOTENSIÓN ALFA-1 ADRENÉRGICA",
            "farmaco_causal": "Quetiapina",
            "farmaco_cascada": "Midodrina / Fludrocortisona / Descartar síncope",
            "doc_id": "CASC-PROT-003",
            "mecanismo": "Potente antagonismo de receptores adrenérgicos alfa-1 que impide la vasoconstricción refleja compensadora al ponerse de pie.",
            "error_tipico": "Prescribir fármacos vasopresores o someter al paciente a pruebas cardiológicas extensivas sin verificar la titulación brusca de Quetiapina.",
            "riesgo_iatrogenico": "Caídas graves, traumatismos craneales y fracturas en personas mayores.",
            "protocolo": [
                "Ralentizar la velocidad de titulación (aumentar de 25-50 mg en lugar de saltos bruscos).",
                "Fraccionar la dosis o administrar la formulación de liberación prolongada (XR) por la noche.",
                "Educar al paciente sobre incorporación lenta en dos tiempos desde la posición supina."
            ],
            "fuente": "Stahl, Jufe"
        }
    },
    "Ácido Valproico (Estabilizador del Ánimo)": {
        "Temblor postural fino de manos (frecuente y dosis-dependiente)": {
            "nivel": "ADVERTENCIA",
            "titulo": "⚠️ TEMBLOR POR VALPROATO → BIPERIDENO (ERROR GRAVE)",
            "farmaco_causal": "Ácido Valproico / Divalproato",
            "farmaco_cascada": "Biperideno (ineficaz para temblor de valproato)",
            "doc_id": "CASC-PROT-003",
            "mecanismo": "Efecto electrofisiológico directo sobre circuitos cerebelo-talámico-corticales; no involucra la vía nigroestriatal colinérgica.",
            "error_tipico": "Prescribir Biperideno creyendo que es un temblor extrapiramidal por antipsicóticos.",
            "riesgo_iatrogenico": "Cero eficacia sobre el temblor de valproato agregando toda la toxicidad anticolinérgica innecesaria (ACB +3).",
            "protocolo": [
                "Medir niveles plasmáticos de valproemia (rango óptimo: 50-100 mcg/mL).",
                "Reducir ligeramente la dosis de valproato si se encuentra en el rango alto.",
                "Si el temblor es incapacitante y el valproato es imprescindible, el fármaco de elección es <b>Propranolol (20-40 mg/día)</b>, NUNCA anticolinérgicos como Biperideno."
            ],
            "fuente": "Jufe - Psicofarmacología Práctica, Stahl"
        }
    }
}


# =============================================================================
# 3. MATRIZ DE PAREJAS DE FÁRMACOS EN POLIFARMACIA (DETECCIÓN CRUZADA ACTIVA)
# =============================================================================
PAREJAS_CASCADA_POLIFARMACIA: List[Dict[str, Any]] = [
    {
        "id": "CASC-PAIR-01",
        "causal_alias": ["haloperidol", "haldol", "flufenazina", "zuclopentixol", "clorpromazina"],
        "cascada_alias": ["biperideno", "akineton", "trihexifenidilo", "artane"],
        "farmaco_causal_nombre": "Antipsicótico Típico (Haloperidol / Similares)",
        "farmaco_cascada_nombre": "Anticolinérgico (Biperideno / Trihexifenidilo)",
        "nivel": "CRÍTICO",
        "titulo": "🚨 CASCADA ACTIVA DETECTADA: PARKINSONISMO TRATADO CON ANTICOLINÉRGICO CRÓNICO",
        "mecanismo": "El antipsicótico típico bloquea masivamente los receptores D2 estriatales; el anticolinérgico enmascara el temblor pero añade toxicidad cognitiva severa.",
        "riesgo_iatrogenico": "Carga anticolinérgica crítica (ACB +3), deterioro de memoria, constipación grave / íleo paralítico y riesgo de discinesia tardía irreversible.",
        "protocolo_desprescripcion": [
            "1. Reducir la dosis del antipsicótico típico a la mínima eficaz.",
            "2. Valorar rotación a un antipsicótico atípico con bajo perfil motor (Quetiapina, Aripiprazol, Olanzapina).",
            "3. Desescalar y retirar gradualmente el Biperideno (1-2 mg/semana en 2 a 4 semanas) para evitar rebote colinérgico."
        ],
        "doc_id": "CASC-HAL-BIP-001"
    },
    {
        "id": "CASC-PAIR-02",
        "causal_alias": ["donepezilo", "aricept", "rivastigmina", "exelon", "prometax", "galantamina", "reminyl"],
        "cascada_alias": ["oxibutinina", "ditropan", "tolterodina", "urispas", "solifenacina", "vesicare", "fesoterodina", "toviaz"],
        "farmaco_causal_nombre": "Inhibidor de Colinesterasa (Donepezilo / Rivastigmina)",
        "farmaco_cascada_nombre": "Anticolinérgico Urológico (Oxibutinina / Tolterodina / Solifenacina)",
        "nivel": "CRÍTICO",
        "titulo": "🛑 CASCADA ACTIVA DE ANTAGONISMO FRONTAL: INHIBIDOR DE COLINESTERASA + ANTICOLINÉRGICO UROLÓGICO",
        "mecanismo": "Donepezilo eleva la acetilcolina cerebral para frenar la demencia; Oxibutinina bloquea la acetilcolina en todo el organismo, anulando el beneficio neurológico y precipitando delirio.",
        "riesgo_iatrogenico": "Aceleración catastrófica del deterioro cognitivo, delirio, riesgo de caídas y fracturas.",
        "protocolo_desprescripcion": [
            "1. Desprescribir inmediatamente el anticolinérgico urológico (Oxibutinina/Tolterodina).",
            "2. Reconocer la incontinencia como efecto secundario colinérgico; reducir la dosis de Donepezilo o rotar a parches de Rivastigmina.",
            "3. Si se precisa fármaco urológico, sustituir por Mirabegrón (agonista beta-3 sin actividad anticolinérgica central)."
        ],
        "doc_id": "CASC-PROT-003"
    },
    {
        "id": "CASC-PAIR-03",
        "causal_alias": ["clozapina", "leponex", "nemea"],
        "cascada_alias": ["biperideno", "akineton", "trihexifenidilo", "artane"],
        "farmaco_causal_nombre": "Clozapina",
        "farmaco_cascada_nombre": "Anticolinérgico Sistémico (Biperideno)",
        "nivel": "CRÍTICO",
        "titulo": "🛑 CASCADA DE MÁXIMO RIESGO: CLOZAPINA + ANTICOLINÉRGICO ORAL (RIESGO DE ÍLEO PARALÍTICO FATAL)",
        "mecanismo": "La clozapina ralentiza severamente el tránsito gastrointestinal. La adición de anticolinérgicos orales paraliza por completo la motilidad colónica.",
        "riesgo_iatrogenico": "Megacolon tóxico, fecaloma obstructivo, perforación intestinal, sepsis y muerte (mortalidad >30%).",
        "protocolo_desprescripcion": [
            "1. Suspender inmediatamente el anticolinérgico oral sistémico.",
            "2. Para la sialorrea por clozapina: utilizar exclusivamente Ipratropio spray sublingual o colirio de atropina 1% sublingual (acción local sin impacto sistémico).",
            "3. Monitorizar el hábito defecatorio con laxantes osmóticos (Macrogol) si pasan >48h sin deposición."
        ],
        "doc_id": "MONO-CLOZAPINA-006"
    },
    {
        "id": "CASC-PAIR-04",
        "causal_alias": ["litio", "carbonato de litio", "plenur"],
        "cascada_alias": ["hidroclorotiazida", "clortalidona", "indapamida", "esidrex", "higrotona"],
        "farmaco_causal_nombre": "Carbonato de Litio",
        "farmaco_cascada_nombre": "Diurético Tiazídico (Hidroclorotiazida / Clortalidona)",
        "nivel": "CRÍTICO",
        "titulo": "🛑 INTERACCIÓN FARMACOLÓGICA LETAL: LITIO + TIAZIDA (INTOXICACIÓN AGUDA POR LITEMIA)",
        "mecanismo": "Las tiazidas inducen pérdida distal de sodio; el riñón compensa reabsorbiendo masivamente sodio y LITIO en el túbulo proximal, elevando la litemia un 40-70%.",
        "riesgo_iatrogenico": "Intoxicación aguda por litio (litemia >1.5-2.0 mEq/L): arritmias cardíacas, ataxia, convulsiones, coma y fracaso renal agudo irreversible.",
        "protocolo_desprescripcion": [
            "1. Determinar litemia urgente, creatinina e ionograma.",
            "2. Si la tiazida se pautó por poliuria del litio: SUSPENDER TIAZIDA e iniciar Amilorida (5-10 mg/día), que no altera la litemia.",
            "3. Si la tiazida es imprescindible cardiológicamente: REDUCIR dosis de litio al 50% y monitorizar litemia cada 48-72 horas."
        ],
        "doc_id": "INTER-LIT-RENAL-004"
    },
    {
        "id": "CASC-PAIR-05",
        "causal_alias": ["olanzapina", "zyprexa", "clozapina", "leponex"],
        "cascada_alias": ["metformina", "dianben", "atorvastatina", "simvastatina", "rosuvastatina", "fenofibrato"],
        "farmaco_causal_nombre": "Antipsicótico Atípico Pro-Metabólico (Olanzapina / Clozapina)",
        "farmaco_cascada_nombre": "Metformina / Hipolipemiantes (Coadyuvancia o Cascada)",
        "nivel": "ADVERTENCIA",
        "titulo": "⚠️ CASCADA METABÓLICA O COADYUVANCIA CARDIOMETABÓLICA",
        "mecanismo": "Bloqueo H1 y 5-HT2C con activación de AMPK hipotalámica y bloqueo M3 pancreático que inducen ganancia de peso masiva y resistencia a la insulina.",
        "riesgo_iatrogenico": "Polifarmacia cardiometabólica progresiva; riesgo de atribuir la diabetes o dislipidemia a causas idiopáticas sin optimizar el antipsicótico.",
        "protocolo_desprescripcion": [
            "1. Evaluar si es factible rotar a antipsicóticos metabólicamente neutros (Aripiprazol, Lurasidona, Cariprazina o Ziprasidona).",
            "2. Si Olanzapina o Clozapina son insustituibles por estabilidad clínica, mantener Metformina como coadyuvancia deliberada protectora.",
            "3. Monitorizar perfil lipídico, glucemia basal y HbA1c trimestralmente."
        ],
        "doc_id": "CASC-METAB-002"
    },
    {
        "id": "CASC-PAIR-06",
        "causal_alias": ["paroxetina", "seroxat", "paxil", "sertralina", "zoloft", "fluoxetina", "prozac", "citalopram", "escitalopram", "venlafaxina"],
        "cascada_alias": ["sildenafilo", "viagra", "tadalafilo", "cialis", "vardenafilo"],
        "farmaco_causal_nombre": "Antidepresivo Serotoninérgico (ISRS / IRSN)",
        "farmaco_cascada_nombre": "Inhibidor PDE-5 (Sildenafilo / Tadalafilo)",
        "nivel": "ADVERTENCIA",
        "titulo": "⚠️ CASCADA DE DISFUNCIÓN SEXUAL: ISRS → INHIBIDOR PDE-5",
        "mecanismo": "La hiperactivación 5-HT2A/5-HT2C inhibe la dopamina mesolímbica y el reflejo eyaculatorio/orgásmico. Los inhibidores PDE-5 solo actúan sobre la erección periférica.",
        "riesgo_iatrogenico": "Gasto económico, persistencia de anorgasmia o pérdida de deseo sin abordar la causa farmacológica.",
        "protocolo_desprescripcion": [
            "1. Evaluar reducción de dosis del antidepresivo a la mínima eficaz.",
            "2. Rotar hacia antidepresivos con perfil ahorrador sexual: Bupropión, Vortioxetina o Mirtazapina.",
            "3. Si la rotación no es viable, coadyuvar con Bupropión (150 mg/día) para restaurar el tono dopaminérgico."
        ],
        "doc_id": "MONO-PAROXETINA-003"
    },
    {
        "id": "CASC-PAIR-07",
        "causal_alias": ["litio", "carbonato de litio", "plenur"],
        "cascada_alias": ["propranolol", "sumial"],
        "farmaco_causal_nombre": "Carbonato de Litio",
        "farmaco_cascada_nombre": "Propranolol (Beta-bloqueante)",
        "nivel": "ADVERTENCIA",
        "titulo": "⚠️ CASCADA ACTIVA: TEMBLOR POR LITIO TRATADO CON PROPRANOLOL SIN MEDIR LITEMIA",
        "mecanismo": "El litio produce temblor postural; prescribir betabloqueantes sin medir litemia puede enmascarar una intoxicación progresiva en curso.",
        "riesgo_iatrogenico": "Enmascaramiento de toxicidad sistémica por sobredosis de litio, riesgo de bradicardia y broncoespasmo.",
        "protocolo_desprescripcion": [
            "1. Medir litemia sérica urgente a las 12 horas post-dosis (rango seguro: 0.6-0.8 mEq/L).",
            "2. Si la litemia es elevada, reducir la dosis de litio (el temblor remitirá sin necesidad de betabloqueante).",
            "3. Solo si la litemia es óptima y persiste interferencia funcional, mantener Propranolol a dosis mínima (10-20 mg c/8-12h)."
        ],
        "doc_id": "MONO-LITIO-005"
    },
    {
        "id": "CASC-PAIR-08",
        "causal_alias": ["fluoxetina", "prozac", "sertralina", "zoloft", "paroxetina", "venlafaxina"],
        "cascada_alias": ["lorazepam", "orfidal", "alprazolam", "trankimazin", "diazepam", "valium", "clonazepam", "rivotril", "lormetazepam"],
        "farmaco_causal_nombre": "Antidepresivo de Inicio (ISRS / IRSN)",
        "farmaco_cascada_nombre": "Benzodiacepina Crónica",
        "nivel": "ADVERTENCIA",
        "titulo": "⚠️ CASCADA DE ANSIEDAD INICIAL TRATADA CON BENZODIACEPINA CRÓNICA",
        "mecanismo": "La activación serotoninérgica inicial provoca ansiedad transitoria (7-14 días). Pautar benzodiacepinas sin plan de desescalada genera dependencia crónica inadvertida.",
        "riesgo_iatrogenico": "Dependencia física a BZD, tolerancia, sedación diurna, riesgo de caídas y fracturas en personas mayores.",
        "protocolo_desprescripcion": [
            "1. Planificar la desescalada gradual de la benzodiacepina una vez instaurado el efecto terapéutico del antidepresivo (semanas 4-6).",
            "2. Reducción lenta (10-25% de la dosis cada 1-2 semanas) para evitar síndrome de abstinencia.",
            "3. Mantener el antidepresivo como tratamiento de base a largo plazo."
        ],
        "doc_id": "MONO-FLUOXETINA-002"
    },
    {
        "id": "CASC-PAIR-09",
        "causal_alias": ["donepezilo", "aricept", "rivastigmina", "exelon"],
        "cascada_alias": ["propranolol", "atenolol", "bisoprolol", "metoprolol", "verapamilo", "diltiazem"],
        "farmaco_causal_nombre": "Inhibidor de Colinesterasa (Donepezilo / Rivastigmina)",
        "farmaco_cascada_nombre": "Bradicardizante Cardíaco (Beta-bloqueante / Calcioantagonista)",
        "nivel": "CRÍTICO",
        "titulo": "🛑 RIESGO CARDIOVASCULAR SEVERO: SINERGIA BRADICARDIZANTE Y SÍNCOPE VAGAL",
        "mecanismo": "La estimulación colinérgica vagal de Donepezilo combinada con el bloqueo beta-adrenérgico o de canales de calcio frena sinérgicamente el nódulo sinoauricular y AV.",
        "riesgo_iatrogenico": "Bradicardia extrema (<45 lpm), bloqueo AV completo, asistolia, síncope y traumatismos craneoencefálicos graves por caída.",
        "protocolo_desprescripcion": [
            "1. Realizar electrocardiograma (ECG) para medir frecuencia cardíaca e intervalo PR.",
            "2. Evaluar reducción de dosis de Donepezilo o del betabloqueante según indicación prioritaria.",
            "3. Evitar duplicidad de fármacos cronotrópicos negativos en pacientes con antecedentes de presíncope."
        ],
        "doc_id": "CASC-PROT-003"
    }
]
