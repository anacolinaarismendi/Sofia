#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_notebooks.py
Aplica y ejecuta deterministamente las correcciones en los 5 notebooks de Sofia:
- 02_preprocesamiento_stahl.ipynb
- 00_limpieza_y_eda.ipynb
- 01_preprocesamiento_jufe.ipynb
- 03_preprocesamiento_canmat.ipynb
- 04_rag_psicofarmacologia.ipynb
"""

import json
import re
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def sanitize_outputs(nb):
    """Elimina rutas absolutas locales con nombres de usuario de las salidas de celdas."""
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            if "text" in out:
                if isinstance(out["text"], list):
                    out["text"] = [
                        re.sub(r"/Users/[^/\s'\"]+", "[WORKSPACE_ROOT]", line)
                        for line in out["text"]
                    ]
                elif isinstance(out["text"], str):
                    out["text"] = re.sub(r"/Users/[^/\s'\"]+", "[WORKSPACE_ROOT]", out["text"])


def ejecutar_notebook_en_python(nb, nb_name):
    """Ejecuta las celdas de código de un notebook secuencialmente en Python."""
    ctx = {}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            # Omitir líneas con comandos mágicos IPython (%pip, %matplotlib, etc.)
            lineas_python = [
                line for line in src.splitlines()
                if not line.strip().startswith("%") and not line.strip().startswith("!")
            ]
            src_limpio = "\n".join(lineas_python)
            if src_limpio.strip():
                try:
                    exec(src_limpio, ctx)
                except Exception as e:
                    print(f"Error en {nb_name} celda {i}: {e}")
                    raise e


# ==============================================================================
# 1. NOTEBOOK 02 (STAHL)
# ==============================================================================
def update_notebook_02():
    print("\n--- Actualizando 02_preprocesamiento_stahl.ipynb ---")
    nb_path = WORKSPACE_ROOT / "notebooks" / "02_preprocesamiento_stahl.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Celda 1: _encontrar_raiz portátil
    cell_1_code = '''import os
import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np

# Detectar la ruta raíz del proyecto de forma dinámica y robusta
def _encontrar_raiz():
    cwd = Path.cwd().resolve()
    for p in [cwd] + list(cwd.parents):
        if (p / 'src').is_dir() and (p / 'data').is_dir():
            return p
        if (p / 'Sofia' / 'src').is_dir() and (p / 'Sofia' / 'data').is_dir():
            return p / 'Sofia'
    raise RuntimeError("No se pudo determinar la raíz del proyecto Sofia.")

project_root = _encontrar_raiz()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

DATA_RAW = project_root / 'data' / 'raw'
DATA_PROCESSED = project_root / 'data' / 'processed'
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

print('Entorno de Stahl configurado. Directorio de datos: data/processed')
'''
    nb["cells"][1]["source"] = [l + "\n" for l in cell_1_code.splitlines()]

    # Celda 3: Litio 0.8-1.1 y Bupropión off-label TDAH
    c3 = "".join(nb["cells"][3]["source"])
    c3 = c3.replace("hasta 1.0-1.2 mEq/L en manía aguda", "hasta 0.8-1.1 mEq/L en manía aguda")
    c3 = c3.replace(
        "Útil en cesación tabáquica y TDAH comórbido.",
        "Útil en cesación tabáquica y TDAH comórbido (uso fuera de indicación / off-label)."
    )
    nb["cells"][3]["source"] = [l + "\n" for l in c3.splitlines()]

    # Celda 7: analizar_perfil_cyp con sustrato_cyp2b6, oraciones, negaciones y aserciones
    cell_7_code = '''def analizar_perfil_cyp(texto_cyp):
    """Extrae interacción enzimática analizando cada oración de forma independiente."""
    oraciones = [s.strip() for s in texto_cyp.lower().replace('\\n', '.').split('.') if s.strip()]
    res = {
        'inhibe_cyp2d6': 0,
        'inhibe_cyp3a4': 0,
        'sustrato_cyp1a2': 0,
        'sustrato_cyp2b6': 0,
        'sustrato_cyp2d6': 0,
        'sustrato_cyp3a4': 0,
    }
    for s in oraciones:
        if re.search(r'\\b(no|sin|ni)\\b', s):
            continue
        if bool(re.search(r'inhib(?:e|idor).*?cyp2d6', s)):
            res['inhibe_cyp2d6'] = 1
        if bool(re.search(r'inhib(?:e|idor).*?cyp3a4', s)):
            res['inhibe_cyp3a4'] = 1
        if (('sustrato' in s or 'metabolizado por' in s or 'metabolismo por' in s) and 'cyp1a2' in s):
            res['sustrato_cyp1a2'] = 1
        if (('sustrato' in s or 'metabolizado por' in s or 'metabolismo por' in s) and 'cyp2b6' in s):
            res['sustrato_cyp2b6'] = 1
        if (('sustrato' in s or 'metabolizado por' in s or 'metabolismo por' in s) and 'cyp2d6' in s):
            res['sustrato_cyp2d6'] = 1
        if (('sustrato' in s or 'metabolizado por' in s or 'metabolismo por' in s) and 'cyp3a4' in s):
            res['sustrato_cyp3a4'] = 1
    return res

cyp_features = df_stahl_raw['metabolismo_cyp'].apply(analizar_perfil_cyp).apply(pd.Series)
df_stahl = pd.concat([df_stahl_raw, cyp_features], axis=1)

# Asignar código ATC completo incluyendo Carbonato de Litio
ATC_STAHL_MAP = {
    'Clozapina': 'N05AH02',
    'Paroxetina': 'N06AB05',
    'Bupropión': 'N06AX12',
    'Lamotrigina': 'N03AX09',
    'Carbonato de Litio': 'N05AN01'
}
df_stahl['atc_codigo'] = df_stahl['farmaco'].map(ATC_STAHL_MAP)

# Aserciones de validación farmacocinética clínica
assert df_stahl['atc_codigo'].notnull().all(), 'Error: Hay fármacos sin código ATC'
assert df_stahl.loc[df_stahl['farmaco'] == 'Paroxetina', 'inhibe_cyp2d6'].values[0] == 1, 'Error: Paroxetina debe inhibir CYP2D6'
assert df_stahl.loc[df_stahl['farmaco'] == 'Bupropión', 'inhibe_cyp2d6'].values[0] == 1, 'Error: Bupropión debe inhibir CYP2D6'
assert df_stahl.loc[df_stahl['farmaco'] == 'Bupropión', 'sustrato_cyp2b6'].values[0] == 1, 'Error: Bupropión debe ser sustrato de CYP2B6'
assert df_stahl.loc[df_stahl['farmaco'] == 'Clozapina', 'sustrato_cyp1a2'].values[0] == 1, 'Error: Clozapina debe ser sustrato de CYP1A2'

cols_cyp = ['inhibe_cyp2d6', 'inhibe_cyp3a4', 'sustrato_cyp1a2', 'sustrato_cyp2b6', 'sustrato_cyp2d6', 'sustrato_cyp3a4']
assert df_stahl.loc[df_stahl['farmaco'] == 'Lamotrigina', cols_cyp].sum().sum() == 0, 'Error: Lamotrigina no debe tener marcas CYP'
assert df_stahl.loc[df_stahl['farmaco'] == 'Carbonato de Litio', cols_cyp].sum().sum() == 0, 'Error: Litio no debe tener marcas CYP'

df_stahl[['farmaco', 'atc_codigo', 'inhibe_cyp2d6', 'sustrato_cyp1a2', 'sustrato_cyp2b6', 'sustrato_cyp2d6']]
'''
    nb["cells"][7]["source"] = [l + "\n" for l in cell_7_code.splitlines()]

    sanitize_outputs(nb)

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    ejecutar_notebook_en_python(nb, "02_preprocesamiento_stahl.ipynb")
    print("✓ 02_preprocesamiento_stahl.ipynb completado y CSVs generados.")


# ==============================================================================
# 2. NOTEBOOK 00 (LIMPIEZA Y EDA)
# ==============================================================================
def update_notebook_00():
    print("\n--- Actualizando 00_limpieza_y_eda.ipynb ---")
    nb_path = WORKSPACE_ROOT / "notebooks" / "00_limpieza_y_eda.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Limpiar salida de celda 1 (%pip)
    if len(nb["cells"]) > 1 and "%pip" in "".join(nb["cells"][1].get("source", [])):
        nb["cells"][1]["outputs"] = []

    # Celda 2: _encontrar_raiz portátil
    cell_2_code = '''# 1. Configuración de Entorno y Rutas Dinámicas Portables
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Detectar la ruta raíz del proyecto de forma dinámica y robusta
def _encontrar_raiz():
    cwd = Path.cwd().resolve()
    for p in [cwd] + list(cwd.parents):
        if (p / 'src').is_dir() and (p / 'data').is_dir():
            return p
        if (p / 'Sofia' / 'src').is_dir() and (p / 'Sofia' / 'data').is_dir():
            return p / 'Sofia'
    raise RuntimeError("No se pudo determinar la raíz del proyecto Sofia.")

project_root = _encontrar_raiz()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

DATA_PROCESSED = project_root / 'data' / 'processed'

# Aislar configuración de matplotlib dentro del proyecto
mpl_tmp = project_root / '.matplotlib_tmp'
mpl_tmp.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(mpl_tmp)

import matplotlib.pyplot as plt
import seaborn as sns

# Configuración estética de gráficos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['figure.dpi'] = 120

print('Entorno de EDA configurado correctamente.')
print(f'- Directorio de datos procesados: {DATA_PROCESSED.relative_to(project_root)}')
'''
    nb["cells"][2]["source"] = [l + "\n" for l in cell_2_code.splitlines()]

    # Celda 4: validador ATC informativo por dataset
    cell_4_code = '''import re

# Carga de datasets limpios
df_jufe = pd.read_csv(DATA_PROCESSED / 'jufe_farmacos_clean.csv')
df_jufe_efectos = pd.read_csv(DATA_PROCESSED / 'jufe_efectos_adversos_clean.csv')
df_stahl = pd.read_csv(DATA_PROCESSED / 'stahl_farmacos_clean.csv')
df_stahl_eventos = pd.read_csv(DATA_PROCESSED / 'stahl_eventos_adversos_clean.csv')
df_canmat = pd.read_csv(DATA_PROCESSED / 'canmat_recomendaciones_clean.csv')
df_canmat_alertas = pd.read_csv(DATA_PROCESSED / 'canmat_alertas_clinicas_clean.csv')

atc_regex = re.compile(r'^[A-Z]\d{2}[A-Z]{2}\d{2}$')

def validar_atc(series, fuente="Desconocida"):
    val = series.dropna()
    if len(val) == 0:
        return 'N/A'
    invalidos = val[~val.apply(lambda x: bool(atc_regex.match(str(x))))].tolist()
    if invalidos:
        print(f"⚠️ ATC fuera de formato nivel 5 en {fuente}: {len(invalidos)}/{len(val)} códigos no conformes -> {invalidos}")
        return f"✗ Inválido ({len(invalidos)})"
    return '✓ Válido'

auditoria = [
    {'Dataset': 'Jufe - Fármacos', 'Filas': len(df_jufe), 'Columnas': len(df_jufe.columns), 'Nulos': df_jufe.isnull().sum().sum(), 'Duplicados': df_jufe.duplicated().sum(), 'ATC': validar_atc(df_jufe['atc_codigo'], fuente='Jufe - Fármacos')},
    {'Dataset': 'Jufe - Efectos Adversos', 'Filas': len(df_jufe_efectos), 'Columnas': len(df_jufe_efectos.columns), 'Nulos': df_jufe_efectos.isnull().sum().sum(), 'Duplicados': df_jufe_efectos.duplicated().sum(), 'ATC': 'N/A'},
    {'Dataset': 'Stahl - Fármacos y CYP', 'Filas': len(df_stahl), 'Columnas': len(df_stahl.columns), 'Nulos': df_stahl.isnull().sum().sum(), 'Duplicados': df_stahl.duplicated().sum(), 'ATC': validar_atc(df_stahl['atc_codigo'], fuente='Stahl - Fármacos y CYP')},
    {'Dataset': 'Stahl - Eventos Adversos', 'Filas': len(df_stahl_eventos), 'Columnas': len(df_stahl_eventos.columns), 'Nulos': df_stahl_eventos.isnull().sum().sum(), 'Duplicados': df_stahl_eventos.duplicated().sum(), 'ATC': 'N/A'},
    {'Dataset': 'CANMAT - Recomendaciones', 'Filas': len(df_canmat), 'Columnas': len(df_canmat.columns), 'Nulos': df_canmat.isnull().sum().sum(), 'Duplicados': df_canmat.duplicated().sum(), 'ATC': validar_atc(df_canmat['atc'], fuente='CANMAT - Recomendaciones')},
    {'Dataset': 'CANMAT - Alertas Críticas', 'Filas': len(df_canmat_alertas), 'Columnas': len(df_canmat_alertas.columns), 'Nulos': df_canmat_alertas.isnull().sum().sum(), 'Duplicados': df_canmat_alertas.duplicated().sum(), 'ATC': 'N/A'},
]

df_auditoria = pd.DataFrame(auditoria)
print('=== TABLA DE CONTROL DE CALIDAD Y AUDITORÍA DE DATOS ===')
df_auditoria
'''
    nb["cells"][4]["source"] = [l + "\n" for l in cell_4_code.splitlines()]

    # Celda 6: Catálogo de fármacos agrupado por ATC (9 fármacos)
    cell_6_code = '''# Consolidar catálogo único agrupando por código ATC para unificar Litio y Carbonato de Litio
df_combinado = pd.concat([
    df_jufe[['farmaco', 'atc_codigo', 'clase']].assign(fuente='Jufe'),
    df_stahl[['farmaco', 'atc_codigo', 'clase']].assign(fuente='Stahl')
])

catalogo_farmacos = df_combinado.groupby('atc_codigo', as_index=False).agg({
    'farmaco': 'first',
    'clase': 'first',
    'fuente': lambda s: ', '.join(sorted(set(s)))
}).rename(columns={'fuente': 'fuentes'})

print(f'Total de fármacos psiquiátricos indexados en el catálogo base (por código ATC único): {len(catalogo_farmacos)}')
catalogo_farmacos
'''
    nb["cells"][6]["source"] = [l + "\n" for l in cell_6_code.splitlines()]

    # Celda 10: Heatmap CYP con sustrato_cyp2b6 y filtro dinámico
    cell_10_code = '''# Construir matriz binaria de citocromos a partir del dataset de Stahl
cyp_candidatos = [
    ('inhibe_cyp2d6', 'Inhibe CYP2D6'),
    ('inhibe_cyp3a4', 'Inhibe CYP3A4'),
    ('sustrato_cyp1a2', 'Sustrato CYP1A2'),
    ('sustrato_cyp2b6', 'Sustrato CYP2B6'),
    ('sustrato_cyp2d6', 'Sustrato CYP2D6'),
    ('sustrato_cyp3a4', 'Sustrato CYP3A4')
]
cols_existentes = [col for col, _ in cyp_candidatos if col in df_stahl.columns]
nombres_legibles = [nom for col, nom in cyp_candidatos if col in df_stahl.columns]

matriz_cyp = df_stahl.set_index('farmaco')[cols_existentes]
matriz_cyp.columns = nombres_legibles

plt.figure(figsize=(10, 4.5))
sns.heatmap(
    matriz_cyp,
    annot=True,
    cmap='YlOrRd',
    cbar=False,
    linewidths=1.5,
    linecolor='white',
    annot_kws={'size': 12, 'weight': 'bold'}
)
plt.title('Matriz de Interacciones CYP450 por Psicofármaco (Stahl - Prescriber\\'s Guide)', fontsize=13, fontweight='bold')
plt.xlabel('Enzimas Citocromo P450', fontsize=11)
plt.ylabel('Principio Activo', fontsize=11)
plt.tight_layout()
plt.show()
'''
    nb["cells"][10]["source"] = [l + "\n" for l in cell_10_code.splitlines()]

    # Celda 13: Métricas CANMAT limpias y aclaración de 22 filas curadas
    cell_13_code = '''# Cálculo dinámico de métricas clínicas para el informe
total_rec = len(df_canmat)
conteo_ev = df_canmat['nivel_evidencia'].value_counts()
total_ev_level1 = conteo_ev.get('Level 1', 0) + conteo_ev.get('Level 1 (Daño)', 0)

print(f'=== SÍNTESIS DE EVIDENCIA CANMAT (Total: {total_rec} recomendaciones curadas a mano) ===')
print('ℹ Nota: Conjunto curado representativo de 22 directrices clave extraídas de las guías CANMAT.')
print(f'• Total con evidencia metodológica Level 1 (a favor y en contra/daño): {total_ev_level1}/{total_rec} ({total_ev_level1/total_rec*100:.1f}%)')
for nivel, count in conteo_ev.items():
    print(f'  - {nivel}: {count}/{total_rec} ({count/total_rec*100:.1f}%)')
'''
    nb["cells"][13]["source"] = [l + "\n" for l in cell_13_code.splitlines()]

    # Celda 14: Conclusiones 1 y 3 reformuladas
    cell_14_markdown = '''## 4. Conclusiones y Repercusiones Clínicas para el Motor RAG Sofia

1. **Jerarquización Terapéutica Racional en CANMAT:** Sofia debe ordenar las opciones clínicas atendiendo conjuntamente a la **línea de tratamiento** (Primera Línea > Segunda Línea > Tercera Línea) y al **nivel de evidencia** (Level 1 a 4). Aquellas opciones marcadas como **Level 1 (Daño)** deben tratarse estrictamente como recomendaciones negativas o en contra (contraindicación o desaconsejo clínico formal), nunca como tratamientos a priorizar.
2. **Detección Determinista de Bloqueos CYP2D6 y Vía CYP2B6:** Fármacos de uso masivo como Paroxetina y Bupropión son potentes inhibidores de CYP2D6, lo que incrementa marcadamente las concentraciones plasmáticas de sustratos como Risperidona o betabloqueantes. Asimismo, Bupropión requiere activación/metabolización primordial por CYP2B6. Estas interacciones deben disparar alertas específicas de seguridad.
3. **Alerta Roja de Seguridad y Modelado Propio:** Los antidepresivos ISRS en monoterapia para Depresión Bipolar están clasificados con la etiqueta **Level 1 (Daño)** —convención interna del proyecto Sofia para modelar advertencias críticas de riesgo en base a CANMAT— debido al riesgo severo de viraje maníaco y ciclación rápida.
4. **Monitoreo de Toxicidad Grave:** Los estabilizadores (Litio y Lamotrigina) y antipsicóticos específicos (Clozapina) presentan eventos adversos potencialmente letales (intoxicación por litemia, síndrome de Stevens-Johnson, agranulocitosis) que requieren protocolos obligatorios de farmacovigilancia.
'''
    nb["cells"][14]["source"] = [l + "\n" for l in cell_14_markdown.splitlines()]

    sanitize_outputs(nb)

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    ejecutar_notebook_en_python(nb, "00_limpieza_y_eda.ipynb")
    print("✓ 00_limpieza_y_eda.ipynb completado y validado.")


# ==============================================================================
# 3. NOTEBOOK 01 (JUFE)
# ==============================================================================
def update_notebook_01():
    print("\n--- Actualizando 01_preprocesamiento_jufe.ipynb ---")
    nb_path = WORKSPACE_ROOT / "notebooks" / "01_preprocesamiento_jufe.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Celda 2: _encontrar_raiz portátil
    cell_2_code = '''import os
import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np

# Detectar la ruta raíz del proyecto de forma dinámica y robusta
def _encontrar_raiz():
    cwd = Path.cwd().resolve()
    for p in [cwd] + list(cwd.parents):
        if (p / 'src').is_dir() and (p / 'data').is_dir():
            return p
        if (p / 'Sofia' / 'src').is_dir() and (p / 'Sofia' / 'data').is_dir():
            return p / 'Sofia'
    raise RuntimeError("No se pudo determinar la raíz del proyecto Sofia.")

project_root = _encontrar_raiz()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

DATA_RAW = project_root / 'data' / 'raw'
DATA_PROCESSED = project_root / 'data' / 'processed'
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

from src.extractors.jufe_extractor import JufeExtractor
from src.extractors.pdf_extractor import CachedPDFExtractor
from src.enrichers.atc_enricher import ATCEnricher

print('Entorno configurado correctamente. Directorio de datos: data/processed')
'''
    nb["cells"][2]["source"] = [l + "\n" for l in cell_2_code.splitlines()]

    # Celda 4: Constante MAX_PAGINAS_PDF y aviso
    cell_4_code = '''# Instanciar el extractor modular con sistema de caché en disco
MAX_PAGINAS_PDF = 150
jufe_extractor = JufeExtractor(raw_dir=DATA_RAW)
ruta_pdf = jufe_extractor.find_pdf_path()

if ruta_pdf:
    print(f'Archivo PDF localizado con éxito: {ruta_pdf.name}')
    pdf_pages = jufe_extractor.pdf_extractor.load_or_extract_all_pages(ruta_pdf, max_pages=MAX_PAGINAS_PDF)
    print(f'Total de páginas disponibles en caché: {len(pdf_pages)}')
    if len(pdf_pages) >= MAX_PAGINAS_PDF:
        print(f'⚠️ Aviso: Se alcanzó el límite máximo de {MAX_PAGINAS_PDF} páginas. El documento podría estar truncado.')
else:
    print('No se detectó el PDF en data/raw. Se utiliza el corpus representativo predefinido en CORPUS_BASE_JUFE.')

print('ℹ Nota metodológica: El extractor localiza el PDF si está disponible, pero la extracción downstream '
      'emplea el corpus curado representativo CORPUS_BASE_JUFE (5 fármacos) para garantizar reproducibilidad determinista.')

# Extracción de monografías completas y eventos adversos
df_jufe, df_jufe_efectos = jufe_extractor.extract_monographs()
print(f'Total de fármacos psiquiátricos procesados: {len(df_jufe)}')
print(f'Total de asociaciones fármaco-efecto adverso extraídas: {len(df_jufe_efectos)}')
df_jufe.head(6)
'''
    nb["cells"][4]["source"] = [l + "\n" for l in cell_4_code.splitlines()]

    # Celda 12: Validaciones exhaustivas (fármacos únicos, ATC regex, severidades, integridad referencial)
    cell_12_code = '''# Validaciones de integridad clínica y relacional
assert df_jufe['farmaco'].isnull().sum() == 0, 'Error: Hay fármacos sin nombre'
assert df_jufe['farmaco'].duplicated().sum() == 0, 'Error: Fármacos duplicados en catálogo de Jufe'
assert df_jufe['atc_codigo'].notnull().all(), 'Error: Hay códigos ATC nulos'
assert df_jufe['atc_codigo'].apply(lambda x: bool(re.match(r'^[A-Z]\\d{2}[A-Z]{2}\\d{2}$', str(x)))).all(), 'Error: Formato ATC nivel 5 inválido'

VOCAB_SEVERIDADES = {'Común / Moderado', 'Grave / Riesgo Vital'}
sev_encontradas = set(df_jufe_efectos['severidad'].dropna())
assert sev_encontradas.issubset(VOCAB_SEVERIDADES), f'Error: Severidades fuera de vocabulario: {sev_encontradas - VOCAB_SEVERIDADES}'

assert set(df_jufe_efectos['farmaco']).issubset(set(df_jufe['farmaco'])), 'Error: Efectos adversos con fármacos no catalogados'
assert len(df_jufe) >= 5, 'Error: Catálogo de fármacos insuficiente'
assert len(df_jufe_efectos) >= 20, 'Error: Catálogo de efectos adversos insuficiente'

# Rutas de salida normalizadas relativas
path_farmacos = DATA_PROCESSED / 'jufe_farmacos_clean.csv'
path_efectos = DATA_PROCESSED / 'jufe_efectos_adversos_clean.csv'

df_jufe.to_csv(path_farmacos, index=False, encoding='utf-8')
df_jufe_efectos.to_csv(path_efectos, index=False, encoding='utf-8')

rel_farmacos = path_farmacos.relative_to(project_root)
rel_efectos = path_efectos.relative_to(project_root)
print(f'Dataset principal de Jufe exportado a: {rel_farmacos} ({len(df_jufe)} fármacos)')
print(f'Dataset de eventos adversos de Jufe exportado a: {rel_efectos} ({len(df_jufe_efectos)} pares)')
'''
    nb["cells"][12]["source"] = [l + "\n" for l in cell_12_code.splitlines()]

    sanitize_outputs(nb)

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    ejecutar_notebook_en_python(nb, "01_preprocesamiento_jufe.ipynb")
    print("✓ 01_preprocesamiento_jufe.ipynb completado y validado.")


# ==============================================================================
# 4. NOTEBOOK 03 (CANMAT)
# ==============================================================================
def update_notebook_03():
    print("\n--- Actualizando 03_preprocesamiento_canmat.ipynb ---")
    nb_path = WORKSPACE_ROOT / "notebooks" / "03_preprocesamiento_canmat.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Celda 0: Markdown Objetivo 2 aclarando Level 1 (Daño)
    c0 = "".join(nb["cells"][0]["source"])
    c0 = c0.replace(
        "Level 1, Level 2, Evidencia de Daño",
        "Level 1, Level 2, y la etiqueta propia 'Level 1 (Daño)' para recomendaciones en contra"
    )
    nb["cells"][0]["source"] = [l + "\n" for l in c0.splitlines()]

    # Celda 1: _encontrar_raiz portátil
    cell_1_code = '''import sys
from pathlib import Path
import pandas as pd
import re

# Detectar la ruta raíz del proyecto de forma dinámica y robusta
def _encontrar_raiz():
    cwd = Path.cwd().resolve()
    for p in [cwd] + list(cwd.parents):
        if (p / 'src').is_dir() and (p / 'data').is_dir():
            return p
        if (p / 'Sofia' / 'src').is_dir() and (p / 'Sofia' / 'data').is_dir():
            return p / 'Sofia'
    raise RuntimeError("No se pudo determinar la raíz del proyecto Sofia.")

project_root = _encontrar_raiz()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.extractors.canmat_extractor import CANMATExtractor

DATA_RAW = project_root / 'data' / 'raw'
DATA_PROCESSED = project_root / 'data' / 'processed'
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

print('Entorno de CANMAT configurado. Directorio de datos: data/processed')
'''
    nb["cells"][1]["source"] = [l + "\n" for l in cell_1_code.splitlines()]

    # Celda 5: Limpieza de espacios en ambas tablas, auditoría de nulos, duplicados, ATC y valores únicos
    cell_5_code = '''# Limpieza de espacios en ambas tablas (df_canmat y df_alertas_canmat)
for df in [df_canmat, df_alertas_canmat]:
    for c in [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]:
        df[c] = df[c].str.strip()

# Comprobación de nulos y duplicados en columnas clave
cols_clave = ['patologia', 'linea', 'estrategia', 'farmaco', 'nivel_evidencia']
nulos_clave = df_canmat[cols_clave].isnull().sum().to_dict()
print(f"Control de calidad en columnas clave: {nulos_clave}")
duplicados = df_canmat.duplicated().sum()
print(f"Duplicados en df_canmat: {duplicados}")
assert duplicados == 0, f"Error: Hay {duplicados} filas duplicadas en df_canmat"

# Comprobación de formato ATC nivel 5
atc_val = df_canmat['atc'].dropna()
atc_regex = re.compile(r'^[A-Z]\\d{2}[A-Z]{2}\\d{2}$')
atc_invalidos = atc_val[~atc_val.apply(lambda x: bool(atc_regex.match(str(x))))].unique().tolist()
if atc_invalidos:
    print(f"ℹ Aviso: Códigos ATC no estándar (agrupaciones o combinaciones clínicas): {atc_invalidos}")

# Valores únicos de línea y nivel_evidencia
print(f"Valores únicos de Línea: {df_canmat['linea'].unique().tolist()}")
print(f"Valores únicos de Nivel de Evidencia: {df_canmat['nivel_evidencia'].unique().tolist()}")

# Tabla cruzada de contingencia original
print("\\nDistribución por Patología y Nivel de Evidencia:")
df_canmat.groupby(['patologia', 'nivel_evidencia']).size().unstack(fill_value=0)
'''
    nb["cells"][5]["source"] = [l + "\n" for l in cell_5_code.splitlines()]

    sanitize_outputs(nb)

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    ejecutar_notebook_en_python(nb, "03_preprocesamiento_canmat.ipynb")
    print("✓ 03_preprocesamiento_canmat.ipynb completado y validado.")


# ==============================================================================
# 5. NOTEBOOK 04 (RAG)
# ==============================================================================
def update_notebook_04():
    print("\n--- Actualizando 04_rag_psicofarmacologia.ipynb ---")
    nb_path = WORKSPACE_ROOT / "notebooks" / "04_rag_psicofarmacologia.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Celda 1: pip completo
    nb["cells"][1]["source"] = ["%pip install requests scikit-learn pandas numpy\n"]
    nb["cells"][1]["outputs"] = []

    # Celda 2: _encontrar_raiz portátil
    cell_2_code = '''import os
import sys
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Detección dinámica y robusta de rutas
def _encontrar_raiz():
    cwd = Path.cwd().resolve()
    for p in [cwd] + list(cwd.parents):
        if (p / 'src').is_dir() and (p / 'data').is_dir():
            return p
        if (p / 'Sofia' / 'src').is_dir() and (p / 'Sofia' / 'data').is_dir():
            return p / 'Sofia'
    raise RuntimeError("No se pudo determinar la raíz del proyecto Sofia.")

project_root = _encontrar_raiz()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

DATA_PROCESSED = project_root / 'data' / 'processed'
DIR_PRETRAINING = project_root / 'pre_training'

print('Entorno RAG configurado exitosamente.')
print(f'- Ruta base del proyecto: {project_root.name}')
print(f'- Directorio de datos: {DATA_PROCESSED.relative_to(project_root)}')
print(f'- Directorio pre_training: {DIR_PRETRAINING.relative_to(project_root)}')
'''
    nb["cells"][2]["source"] = [l + "\n" for l in cell_2_code.splitlines()]

    # Celda 3: Markdown aclarando uso de Jufe/Stahl para control de cobertura
    c3 = '''## 1. Carga de Datasets Procesados y del Corpus Clínico `pre_training/`

Para maximizar la fiabilidad y trazabilidad médica, integramos:
* **Tablas estructuradas procesadas:** `df_jufe` y `df_stahl` se cargan como control de cobertura y coherencia; `df_canmat` se compila directamente en fichas clínicas.
* **Corpus de conocimiento curado (`pre_training/`):** Documentos en Markdown enriquecidos con metadatos YAML (monografías, interacciones CYP450, cascadas de prescripción, algoritmos CANMAT y alertas de seguridad).
'''
    nb["cells"][3]["source"] = [l + "\n" for l in c3.splitlines()]

    # Celda 4: cargar_corpus_pretraining con utf-8-sig, avisos front matter y Voice Snippet
    cell_4_code = '''# 1. Cargar Datasets Tabulares de Soporte
df_jufe = pd.read_csv(DATA_PROCESSED / 'jufe_farmacos_clean.csv')
df_jufe_efectos = pd.read_csv(DATA_PROCESSED / 'jufe_efectos_adversos_clean.csv')
df_stahl = pd.read_csv(DATA_PROCESSED / 'stahl_farmacos_clean.csv')
df_stahl_eventos = pd.read_csv(DATA_PROCESSED / 'stahl_eventos_adversos_clean.csv')
df_canmat = pd.read_csv(DATA_PROCESSED / 'canmat_recomendaciones_clean.csv')

# 2. Función Lectora del Corpus pre_training/
def cargar_corpus_pretraining(directorio: Path):
    docs = []
    if not directorio.exists():
        print(f'Aviso: Directorio {directorio} no encontrado.')
        return docs

    for f in sorted(directorio.glob('**/*.md')):
        if f.name.startswith('00_'):
            continue
        try:
            with open(f, 'r', encoding='utf-8-sig') as arch:
                texto = arch.read()
        except Exception as e:
            print(f'Aviso: No se pudo leer {f.name}: {e}')
            continue

        parts = texto.split('---', 2)
        if len(parts) < 3:
            print(f'⚠️ Aviso: Archivo omitido por carecer de front matter YAML: {f.name}')
            continue

        raw_yaml = parts[1]
        cuerpo = parts[2].strip()
        meta = {}
        for line in raw_yaml.strip().splitlines():
            if ':' in line:
                k, v = line.split(':', 1)
                meta[k.strip()] = v.strip().strip('"').strip("'")

        voice_snippet = ""
        match_voice = re.search(r'## 1\\. Resumen Ejecutivo Verbalizable.*?\\n>(.*?)(?=\\n##|\\Z)', cuerpo, re.DOTALL)
        if match_voice:
            voice_snippet = match_voice.group(1).replace('\\n>', ' ').replace('"', '').strip()
        else:
            print(f'ℹ Aviso: No se detectó Voice Snippet en: {f.name}')

        docs.append({
            'id': meta.get('id', f.stem),
            'tipo': meta.get('tipo', f.parent.name),
            'titulo': meta.get('titulo', meta.get('farmaco', f.stem.replace('_', ' ').title())),
            'atc': meta.get('atc_codigo', 'N/D'),
            'voice_snippet': voice_snippet,
            'contenido': cuerpo,
            'archivo_origen': str(f.relative_to(project_root))
        })
    return docs

docs_pretraining = cargar_corpus_pretraining(DIR_PRETRAINING)
print(f'Datos cargados exitosamente:')
print(f'- Tablas CSV: Jufe ({len(df_jufe)} fármacos) | Stahl ({len(df_stahl)} fármacos) | CANMAT ({len(df_canmat)} directrices)')
print(f'- Corpus pre_training: {len(docs_pretraining)} documentos Markdown ricos parseados.')
'''
    nb["cells"][4]["source"] = [l + "\n" for l in cell_4_code.splitlines()]

    # Celda 6: CATALOGO_MARCAS con Valproato y sin N/D
    cell_6_code = '''CATALOGO_MARCAS = pd.DataFrame([
    {"farmaco": "Sertralina", "marca": "Zoloft", "laboratorio": "Pfizer", "pais": "Internacional", "presentacion": "Comprimidos 50mg / 100mg"},
    {"farmaco": "Sertralina", "marca": "Serlift", "laboratorio": "Roemmers", "pais": "Argentina/LatAm", "presentacion": "Comprimidos 50mg / 100mg"},
    {"farmaco": "Sertralina", "marca": "Aremis", "laboratorio": "Almirall", "pais": "España", "presentacion": "Comprimidos 50mg / 100mg"},
    {"farmaco": "Fluoxetina", "marca": "Prozac", "laboratorio": "Eli Lilly", "pais": "Internacional", "presentacion": "Cápsulas 20mg"},
    {"farmaco": "Fluoxetina", "marca": "Foxetin", "laboratorio": "Gador", "pais": "Argentina/LatAm", "presentacion": "Comprimidos 20mg"},
    {"farmaco": "Paroxetina", "marca": "Paxil / Seroxat", "laboratorio": "GlaxoSmithKline (GSK)", "pais": "Internacional", "presentacion": "Comprimidos 20mg"},
    {"farmaco": "Paroxetina", "marca": "Aropax", "laboratorio": "GSK", "pais": "LatAm", "presentacion": "Comprimidos 20mg"},
    {"farmaco": "Paroxetina", "marca": "Psicoasten", "laboratorio": "Baliarda", "pais": "Argentina", "presentacion": "Comprimidos 20mg"},
    {"farmaco": "Clozapina", "marca": "Clozaril", "laboratorio": "Novartis", "pais": "Internacional", "presentacion": "Comprimidos 25mg / 100mg"},
    {"farmaco": "Clozapina", "marca": "Lapenax", "laboratorio": "Novartis", "pais": "LatAm", "presentacion": "Comprimidos 25mg / 100mg"},
    {"farmaco": "Bupropión", "marca": "Wellbutrin", "laboratorio": "GSK", "pais": "Internacional", "presentacion": "Comprimidos XL 150mg / 300mg"},
    {"farmaco": "Bupropión", "marca": "Zyban", "laboratorio": "GSK", "pais": "Internacional", "presentacion": "Comprimidos SR 150mg"},
    {"farmaco": "Bupropión", "marca": "Odranal", "laboratorio": "Roemmers", "pais": "Argentina", "presentacion": "Comprimidos 150mg"},
    {"farmaco": "Haloperidol", "marca": "Haldol", "laboratorio": "Janssen-Cilag", "pais": "Internacional", "presentacion": "Gotas / Comprimidos 5mg / Ampollas"},
    {"farmaco": "Haloperidol", "marca": "Halopidol", "laboratorio": "Neuval", "pais": "LatAm", "presentacion": "Gotas y comprimidos"},
    {"farmaco": "Litio (Carbonato de Litio)", "marca": "Plenur", "laboratorio": "Faes Farma", "pais": "España", "presentacion": "Comprimidos liberación prolongada 400mg"},
    {"farmaco": "Litio (Carbonato de Litio)", "marca": "Ceglution", "laboratorio": "Baliarda", "pais": "Argentina", "presentacion": "Comprimidos 300mg"},
    {"farmaco": "Lamotrigina", "marca": "Lamictal", "laboratorio": "GSK", "pais": "Internacional", "presentacion": "Comprimidos dispersables 25mg / 50mg / 100mg"},
    {"farmaco": "Lamotrigina", "marca": "Lamotrax", "laboratorio": "Montpellier", "pais": "Argentina", "presentacion": "Comprimidos 25mg / 50mg / 100mg"},
    {"farmaco": "Risperidona", "marca": "Risperdal", "laboratorio": "Janssen-Cilag", "pais": "Internacional", "presentacion": "Comprimidos 1mg / 2mg / 3mg, solución oral"},
    {"farmaco": "Risperidona", "marca": "Sequinan", "laboratorio": "Roemmers", "pais": "Argentina", "presentacion": "Comprimidos 1mg / 2mg / 3mg"},
    {"farmaco": "Valproato (Ácido Valproico)", "marca": "Depakote", "laboratorio": "Abbott", "pais": "Internacional", "presentacion": "Comprimidos 250mg / 500mg"},
    {"farmaco": "Valproato (Ácido Valproico)", "marca": "Depakine", "laboratorio": "Sanofi", "pais": "Europa/LatAm", "presentacion": "Comprimidos 200mg / 500mg"},
    {"farmaco": "Valproato (Ácido Valproico)", "marca": "Logical", "laboratorio": "Investi", "pais": "Argentina", "presentacion": "Comprimidos 250mg / 500mg"}
])

print(f"Catálogo de marcas registrado: {len(CATALOGO_MARCAS)} marcas comerciales con laboratorios.")
CATALOGO_MARCAS.head()
'''
    nb["cells"][6]["source"] = [l + "\n" for l in cell_6_code.splitlines()]

    # Celda 7: Markdown sin "26" fijo
    c7 = '''## 3. Construcción y Fusión de la Base de Conocimiento RAG

Integramos los documentos curados de `pre_training/`, las recomendaciones de las guías CANMAT (formato tabular CSV) y el catálogo de marcas comerciales.
'''
    nb["cells"][7]["source"] = [l + "\n" for l in c7.splitlines()]

    # Celda 8: Control de IDs duplicados avisando y celda de cobertura
    cell_8_code = '''import unicodedata

documentos_rag = []

# 1. Incorporar los documentos curados de pre_training/
documentos_rag.extend(docs_pretraining)

# 2. Añadir fichas directas de guías CANMAT con IDs únicos (GUIA-CANMAT-CSV-XXX)
for i, (_, row_c) in enumerate(df_canmat.iterrows()):
    doc_id = f'GUIA-CANMAT-CSV-{i+1:03d}'
    atc_val = 'N/D' if pd.isna(row_c['atc']) else str(row_c['atc'])
    contenido_guia = f"""# GUÍA CLÍNICA CANMAT: {row_c['patologia'].upper()} ({row_c['fase'].upper()})
- Patología: {row_c['patologia']}
- Fase Clínica: {row_c['fase']}
- Recomendación: {row_c['linea']}
- Estrategia: {row_c['estrategia']}
- Fármaco / Esquema: {row_c['farmaco']} (ATC: {atc_val})
- Nivel de Evidencia Científica: {row_c['nivel_evidencia']}
- Directriz Clínica: {row_c['comentarios']}
- Fuente: Canadian Network for Mood and Anxiety Treatments (CANMAT)
"""
    documentos_rag.append({
        'id': doc_id,
        'tipo': 'guia_canmat_csv',
        'titulo': f"CANMAT: {row_c['patologia']} - {row_c['farmaco']}",
        'atc': atc_val,
        'voice_snippet': f"CANMAT recomienda {row_c['farmaco']} en {row_c['linea']} para {row_c['patologia']} ({row_c['fase']}) con evidencia {row_c['nivel_evidencia']}.",
        'contenido': contenido_guia.strip(),
        'archivo_origen': 'data/processed/canmat_recomendaciones_clean.csv'
    })

# 3. Añadir catálogo de marcas comerciales
for farmaco, grupo in CATALOGO_MARCAS.groupby('farmaco'):
    norm_f = unicodedata.normalize('NFKD', farmaco.split()[0]).encode('ASCII', 'ignore').decode('utf-8').upper()
    doc_id = f'MARCA-{norm_f}-001'
    marcas_validas = [f"{r['marca']} ({r['laboratorio']}, {r['pais']})" for _, r in grupo.iterrows() if r['marca'] != 'N/D']
    marcas_str = ', '.join(marcas_validas)
    pres_validas = [p for p in grupo['presentacion'].unique() if p != 'N/D']
    contenido_marca = f"""# MARCAS COMERCIALES Y LABORATORIOS: {farmaco.upper()}
- Principio Activo: {farmaco}
- Marcas registradas y Laboratorios: {marcas_str}
- Presentaciones clínicas: {'; '.join(pres_validas)}
"""
    documentos_rag.append({
        'id': doc_id,
        'tipo': 'catalogo_marcas',
        'titulo': f"Marcas Comerciales y Laboratorios: {farmaco}",
        'atc': 'N/D',
        'voice_snippet': f"Las marcas principales de {farmaco} incluyen {marcas_str}.",
        'contenido': contenido_marca.strip(),
        'archivo_origen': 'CATALOGO_MARCAS'
    })

# Control de unicidad de IDs con advertencia
ids_vistos = set()
docs_unicos = []
for d in documentos_rag:
    if d['id'] in ids_vistos:
        print(f"⚠️ Aviso: ID duplicado detectado y omitido: {d['id']} ({d['titulo']})")
    else:
        ids_vistos.add(d['id'])
        docs_unicos.append(d)
documentos_rag = docs_unicos

print(f'Total de documentos clínicos indexables en el RAG (IDs únicos): {len(documentos_rag)}')

# Control de Cobertura Cruzada: Monografías vs Datasets CSV
atcs_csv = set(df_jufe['atc_codigo'].dropna()).union(set(df_stahl['atc_codigo'].dropna()))
mono_docs = [d for d in docs_pretraining if d['tipo'] == 'monografia_farmacologica']
print(f'\\n=== CONTROL DE COBERTURA CRUZADA (Monografías vs CSVs) ===')
for m in mono_docs:
    atc_m = m.get('atc', 'N/D')
    en_csv = atc_m in atcs_csv
    icono = "✓" if en_csv else "ℹ Discrepancia detectada"
    print(f"- {m['titulo']} [{m['id']}] (ATC: {atc_m}): {icono} {'presente en CSV' if en_csv else 'NO figura en los CSVs estructurados (esperado para fármacos añadidos al corpus pre_training)'}")
'''
    nb["cells"][8]["source"] = [l + "\n" for l in cell_8_code.splitlines()]

    # Celda 10: VectorStore con fragmentar_documento, STOPWORDS_ES, límites y top_k=6 con topes
    cell_10_code = '''STOPWORDS_ES = [
    'de', 'la', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no',
    'una', 'su', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque',
    'esta', 'son', 'entre', 'está', 'cuando', 'muy', 'sin', 'sobre', 'ser', 'tiene', 'también', 'me',
    'hasta', 'hay', 'donde', 'han', 'quien', 'están', 'estado', 'desde', 'todo', 'nos', 'durante',
    'estados', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'fueron', 'ese', 'eso', 'había', 'ante',
    'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él',
    'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'sea', 'poco', 'ella',
    'estar', 'haber', 'estas', 'estaba', 'estamos', 'algunas', 'algo', 'nosotros'
]

def fragmentar_documento(doc, max_chars=2000):
    """Divide un documento clínico por secciones ## de hasta max_chars."""
    contenido = doc['contenido']
    secciones = re.split(r'\\n(?=##\\s+)', contenido)
    chunks = []
    for i, sec in enumerate(secciones):
        sec = sec.strip()
        if not sec:
            continue
        primera_linea = sec.split('\\n')[0]
        titulo_sec = primera_linea.replace('##', '').strip() if primera_linea.startswith('##') else 'Introducción'

        if len(sec) <= max_chars:
            chunks.append({
                **doc,
                'chunk_id': f"{doc['id']}#sec-{i}",
                'seccion_relevante': titulo_sec,
                'texto_indexable': f"{doc['titulo']} [{doc['id']}] - {titulo_sec}\\n{sec}",
                'contenido_chunk': sec
            })
        else:
            sub_partes = sec.split('\\n\\n')
            buf = ""
            sub_idx = 0
            for p in sub_partes:
                if len(buf) + len(p) + 2 <= max_chars:
                    buf += ("\\n\\n" + p if buf else p)
                else:
                    if buf:
                        chunks.append({
                            **doc,
                            'chunk_id': f"{doc['id']}#sec-{i}-{sub_idx}",
                            'seccion_relevante': titulo_sec,
                            'texto_indexable': f"{doc['titulo']} [{doc['id']}] - {titulo_sec}\\n{buf}",
                            'contenido_chunk': buf
                        })
                        sub_idx += 1
                    buf = p
            if buf:
                chunks.append({
                    **doc,
                    'chunk_id': f"{doc['id']}#sec-{i}-{sub_idx}",
                    'seccion_relevante': titulo_sec,
                    'texto_indexable': f"{doc['titulo']} [{doc['id']}] - {titulo_sec}\\n{buf}",
                    'contenido_chunk': buf
                })
    return chunks if chunks else [{
        **doc,
        'chunk_id': f"{doc['id']}#0",
        'seccion_relevante': 'Documento Completo',
        'texto_indexable': f"{doc['titulo']} [{doc['id']}]\\n{contenido[:max_chars]}",
        'contenido_chunk': contenido
    }]

class ClinicalVectorStore:
    def __init__(self, ollama_host='http://localhost:11434', embedding_model='embeddinggemma:latest'):
        self.ollama_host = ollama_host
        self.embedding_model = embedding_model
        self.documentos_padre = []
        self.chunks = []
        self.embeddings = None
        self.vectorizer = None
        self.ollama_servidor_activo = False
        self.use_ollama_embeddings = False
        self._verificar_ollama()
        self.modo_indice = 'ollama' if self.use_ollama_embeddings else 'tfidf'

    def _verificar_ollama(self):
        try:
            r = requests.get(f'{self.ollama_host}/api/tags', timeout=2)
            if r.status_code == 200:
                self.ollama_servidor_activo = True
                tags = [m.get('name', '') for m in r.json().get('models', [])]
                self.use_ollama_embeddings = any(self.embedding_model in t or t.startswith(self.embedding_model.split(':')[0]) for t in tags)
                if self.use_ollama_embeddings:
                    print(f'✓ Conectado a Ollama local ({self.ollama_host}) con modelo {self.embedding_model}.')
                else:
                    print(f'ℹ Servidor Ollama activo en {self.ollama_host}, pero modelo {self.embedding_model} no instalado. Se usará TF-IDF léxico para recuperación.')
                return
        except Exception:
            pass
        self.ollama_servidor_activo = False
        self.use_ollama_embeddings = False
        print('ℹ Ollama no está activo. Se usará el vectorizador TF-IDF local con strip_accents=unicode y STOPWORDS_ES.')

    def indexar_documentos(self, docs):
        self.documentos_padre = docs
        self.chunks = []
        for d in docs:
            self.chunks.extend(fragmentar_documento(d, max_chars=2000))

        textos = [c['texto_indexable'] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=12000,
            strip_accents='unicode',
            stop_words=STOPWORDS_ES
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(textos)

        if self.use_ollama_embeddings:
            try:
                print(f'Generando embeddings con Ollama ({self.embedding_model}) para {len(self.chunks)} fragmentos...')
                emb_list = []
                for t in textos:
                    res = requests.post(
                        f'{self.ollama_host}/api/embeddings',
                        json={'model': self.embedding_model, 'prompt': t[:6000]},
                        timeout=30
                    )
                    res.raise_for_status()
                    emb_list.append(res.json()['embedding'])
                self.embeddings = np.array(emb_list)
                self.modo_indice = 'ollama'
                print(f'✓ Embeddings calculados exitosamente con Ollama ({self.embedding_model}).')
                return
            except Exception as e:
                print(f'Error generando embeddings en Ollama: {e}. Conmutando a fallback TF-IDF local.')
                self.use_ollama_embeddings = False
                self.modo_indice = 'tfidf'
                self.embeddings = None
        else:
            self.modo_indice = 'tfidf'

        print(f'✓ Indexados {len(self.documentos_padre)} documentos ({len(self.chunks)} fragmentos enriquecidos) con TF-IDF local.')

    def buscar(self, query, top_k=6):
        indices_ordenados = []
        sims = np.zeros(len(self.chunks))

        if self.modo_indice == 'ollama' and self.use_ollama_embeddings and self.embeddings is not None:
            try:
                res = requests.post(
                    f'{self.ollama_host}/api/embeddings',
                    json={'model': self.embedding_model, 'prompt': query},
                    timeout=5
                )
                if res.status_code == 200:
                    query_vec = np.array(res.json()['embedding']).reshape(1, -1)
                    sims = cosine_similarity(query_vec, self.embeddings)[0]
                    indices_ordenados = np.argsort(sims)[::-1]
            except Exception:
                pass

        if len(indices_ordenados) == 0:
            query_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            indices_ordenados = np.argsort(sims)[::-1]

        # Aplicar topes: máx 3 fragmentos por documento padre y máx 2 fichas de marcas
        recuperados = []
        conteo_por_doc = {}
        conteo_marcas = 0

        for idx in indices_ordenados:
            chunk = self.chunks[idx]
            doc_id = chunk['id']
            tipo = chunk.get('tipo', '')

            if tipo == 'catalogo_marcas' and conteo_marcas >= 2:
                continue
            if conteo_por_doc.get(doc_id, 0) >= 3:
                continue

            score = round(float(sims[idx]), 4)
            recuperados.append({**chunk, 'score_similitud': score})
            conteo_por_doc[doc_id] = conteo_por_doc.get(doc_id, 0) + 1
            if tipo == 'catalogo_marcas':
                conteo_marcas += 1

            if len(recuperados) >= top_k:
                break

        return recuperados

# Inicializar e indexar
vector_store = ClinicalVectorStore()
vector_store.indexar_documentos(documentos_rag)
'''
    nb["cells"][10]["source"] = [l + "\n" for l in cell_10_code.splitlines()]

    # Celda 12: Prompt de Sofia como asistente de apoyo, Regla 5, validador sin_citas, abstención y verificar_caso
    cell_12_code = '''PROMPT_SISTEMA_CLINICO = """Eres Sofia, asistente clínica de apoyo a la decisión médica basada en evidencia en Psicofarmacología y Farmacovigilancia.
Tu función es brindar asistencia diagnóstica y terapéutica orientativa a profesionales de la salud con el máximo rigor científico, basándote EXCLUSIVAMENTE en el CONTEXTO CLÍNICO RECUPERADO.

### DIRECTRICES DE RAZONAMIENTO Y SEGURIDAD CLÍNICA:
1. GROUNDING ESTRICTO: Basa todas tus afirmaciones exclusivamente en los documentos clínicos del contexto recuperado del corpus de Sofia. Si el contexto no contiene datos suficientes para responder con seguridad, declara explícitamente "No consta" sin inventar datos.
2. TRAZABILIDAD Y CITACIÓN POR IDENTIFICADOR ([DOC-ID]): No atribuyas citas a autores personales. Respalda cada afirmación citando entre corchetes el código identificador único del documento clínico correspondiente (ej: [MONO-SERTRALINA-001], [INTER-CYP2D6-001], [CASC-HAL-BIP-001], [GUIA-CANMAT-MANIA-001], [MARCA-SERTRALINA-001]).
3. FARMACOVIGILANCIA Y DETECCIÓN DE CASCADAS: Si la consulta plantea el uso de un fármaco para tratar el efecto adverso de otro (ej. haloperidol con biperideno), resalta de forma prioritaria una 'ALERTA DE CASCADA DE PRESCRIPCIÓN' y orienta hacia la desescalada o rotación a un fármaco con menor perfil adverso.
4. METABOLISMO ENZIMÁTICO Y CITOCROMOS: Especifica las vías metabólicas y los citocromos involucrados (CYP1A2, CYP2D6, CYP3A4, CYP2B6, glucuronidación UGT1A4 o eliminación renal) ÚNICAMENTE si el contexto clínico recuperado los menciona explícitamente.
5. ALCANCE COMO HERRAMIENTA DE APOYO: Sofia es una herramienta computacional de soporte a la decisión; todas las recomendaciones deben ser valoradas en el contexto individual del paciente por el médico prescriptor responsable.

### FORMATO DE RESPUESTA CLÍNICA OBLIGATORIO:
Estructura tu respuesta en los siguientes 4 apartados numerados. Si para cualquiera de ellos no dispones de datos en el contexto clínico recuperado, escribe únicamente "No consta":
- **1. Juicio Clínico Inmediato:** Conclusión directa y aplicable a la consulta en 2-3 líneas (si no hay datos suficientes, escribe "No consta").
- **2. Mecanismo Farmacológico y Cinética:** Explicación fisiopatológica, receptores involucrados y perfil enzimático CYP450 (solo si el contexto los menciona explícitamente; de lo contrario, escribe "No consta").
- **3. Pauta Operativa y Recomendaciones:** Dosis sugeridas, titulación/desescalada y parámetros obligatorios de monitorización (si no constan en el contexto, escribe "No consta").
- **4. Alertas de Seguridad y Trazabilidad Comercial:** Advertencias críticas (caja negra, litemia, agranulocitosis) y nombres comerciales / laboratorios aplicables (si no constan en el contexto, escribe "No consta").
"""

def validar_trazabilidad_doc_ids(respuesta: str, vs: ClinicalVectorStore) -> dict:
    """Valida que todos los identificadores [DOC-ID] citados existan formalmente en la base indexada."""
    citas = re.findall(r"\\[((?:MONO|INTER|CASC|GUIA|ALERTA|MARCA)-[A-Z0-9\\-]+)\\]", respuesta)
    ids_conocidos = {d['id'] for d in vs.documentos_padre}
    validas = [c for c in citas if c in ids_conocidos]
    invalidas = [c for c in citas if c not in ids_conocidos]
    return {
        'total_citas': len(citas),
        'validas': validas,
        'invalidas': invalidas,
        'sin_citas': len(citas) == 0,
        'cobertura_pct': round((len(validas) / max(len(citas), 1)) * 100, 1)
    }

def generar_respuesta_rag(pregunta, vector_store, modelo_ollama="qwen3.5:2b-mlx", top_k=6):
    docs_recuperados = vector_store.buscar(pregunta, top_k=top_k)
    max_score = max([d['score_similitud'] for d in docs_recuperados]) if docs_recuperados else 0.0

    # Guardia de cobertura: si la similitud léxica máxima es casi nula, abstención
    if max_score < 0.05:
        return "No consta información clínica en el corpus de Sofia sobre esta consulta.", []

    contexto_str = "\\n\\n---\\n\\n".join([
        f"--- DOCUMENTO CLÍNICO [{d.get('id', 'DOC-001')}] ---\\nTítulo: {d['titulo']} (Tipo: {d.get('tipo', 'clinico')} | ATC: {d.get('atc', 'N/D')})\\nSección: {d.get('seccion_relevante', 'General')}\\nID de Citación: [{d.get('id', 'DOC-001')}]\\n{d['contenido_chunk']}"
        for d in docs_recuperados
    ])

    prompt_completo = f"""{PROMPT_SISTEMA_CLINICO}

==================================================
CONTEXTO CLÍNICO RECUPERADO:
{contexto_str}
==================================================
PREGUNTA DEL MÉDICO / USUARIO:
{pregunta}
==================================================
RESPUESTA CLÍNICA DE SOFIA:"""

    if vector_store.ollama_servidor_activo:
        try:
            res = requests.post(
                f"{vector_store.ollama_host}/api/generate",
                json={
                    "model": modelo_ollama,
                    "prompt": prompt_completo,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "think": False,
                        "num_ctx": 8192,
                        "num_predict": 1200
                    }
                },
                timeout=180
            )
            if res.status_code == 200:
                texto_gen = res.json().get("response", "").strip()
                if texto_gen:
                    return texto_gen, docs_recuperados
                else:
                    print("Aviso: Respuesta vacía de Ollama. Utilizando modo de contingencia local.")
        except Exception as e:
            print(f"Aviso: Error con Ollama ({e}). Mostrando síntesis clínica local con fragmentos recuperados.")

    # Modo de contingencia sin LLM
    top_doc = docs_recuperados[0] if docs_recuperados else {}
    top_id = top_doc.get("id", "DOC-001")
    sintesis = f"**1. Juicio Clínico Inmediato (Modo Local):** Para '{pregunta}', fragmentos recuperados:\\n"
    for d in docs_recuperados:
        sintesis += f"- Fragmento [{d.get('id', 'DOC-001')}] ({d.get('seccion_relevante', 'Sección')}): {d['titulo']} (Score: {d['score_similitud']})\\n"
        if d.get('voice_snippet'):
            snip = d['voice_snippet']
            resumen_texto = (snip[:160] + '...') if len(snip) > 160 else snip
            sintesis += f"  > Resumen Verbal: \\\"{resumen_texto}\\\"\\n"
    sintesis += f"\\n**4. Alertas de Seguridad y Trazabilidad Comercial:** Respaldo registrado en [{top_id}]."
    return sintesis, docs_recuperados

def verificar_caso(caso_id, pregunta, docs_esperados, respuesta, docs_recuperados, vs):
    """Comprueba que se recuperaron los documentos esperados, que las citas existen y estaban en el contexto."""
    ids_recuperados = {d['id'] for d in docs_recuperados}
    aciertos = [doc for doc in docs_esperados if doc in ids_recuperados]
    trazabilidad = validar_trazabilidad_doc_ids(respuesta, vs)

    citas_en_resp = set(trazabilidad['validas'])
    citas_no_en_contexto = citas_en_resp - ids_recuperados

    print(f"[{caso_id}] Verificación:")
    print(f"• Documentos esperados recuperados: {len(aciertos)}/{len(docs_esperados)} ({aciertos})")
    print(f"• Citas válidas en base de conocimiento: {len(trazabilidad['validas'])}/{trazabilidad['total_citas']}")
    if citas_no_en_contexto:
        print(f"⚠️ Citas no presentes en el contexto recuperado: {citas_no_en_contexto}")
    else:
        print(f"✓ Todas las citas válidas corresponden a documentos recuperados en el contexto.")
'''
    nb["cells"][12]["source"] = [l + "\n" for l in cell_12_code.splitlines()]

    # Celda 14: Caso 1
    cell_14_code = '''pregunta_1 = "¿Qué laboratorios comercializan la Sertralina, cuáles son sus marcas comerciales principales y cuál es su posología recomendada de inicio y mantenimiento?"
resp_1, docs_1 = generar_respuesta_rag(pregunta_1, vector_store)
print(resp_1)
print("\\n" + "="*60)
verificar_caso("CASO 1", pregunta_1, ["MONO-SERTRALINA-001", "MARCA-SERTRALINA-001"], resp_1, docs_1, vector_store)
'''
    nb["cells"][14]["source"] = [l + "\n" for l in cell_14_code.splitlines()]

    # Celda 15: Simulación Caso 1 (redactada a mano con MARCA-SERTRALINA-001 y 25 mg ancianos)
    c15 = '''### 📋 Simulación de Respuesta Clínica con LLM (Ejemplo de Referencia Redactado a Mano - Caso 1)
> **Nota metodológica:** Esta transcripción ilustra la salida clínica estructurada generada con trazabilidad ideal [DOC-ID].

> **Referencias Técnicas:** `[MONO-SERTRALINA-001]`, `[MARCA-SERTRALINA-001]`

**RESPUESTA CLÍNICA:**

**1. Juicio Clínico Inmediato:**
La sertralina `[MONO-SERTRALINA-001]` es un antidepresivo ISRS de primera línea con un amplio perfil de seguridad y mínima propensión a interacciones a dosis terapéuticas estándar.

**2. Mecanismo Farmacológico y Cinética:**
Inhibidor potente y selectivo de la recaptación de serotonina (SERT) con débil acción dopaminérgica `[MONO-SERTRALINA-001]`. A dosis habituales (≤ 100 mg/día) presenta una inhibición leve o mínima sobre el citocromo CYP2D6, lo que la posiciona como fármaco de elección en pacientes ancianos y cardiópatas.

**3. Pauta Operativa y Recomendaciones:**
- Dosis inicial habitual: 50 mg/día en toma matutina o vespertina `[MONO-SERTRALINA-001]`.
- En ancianos o pacientes con comorbilidad de pánico/ansiedad: iniciar con 25 mg/día durante los primeros 4 a 7 días para minimizar la agitación paradójica inicial.
- Dosis de mantenimiento: 50 a 100 mg/día; dosis máxima de 200 mg/día tras titulación progresiva de 50 mg cada 1 a 2 semanas.

**4. Alertas de Seguridad y Trazabilidad Comercial:**
- Marcas comerciales principales y laboratorios `[MARCA-SERTRALINA-001]`: **Zoloft** (Pfizer), **Serlift** (Roemmers) y **Aremis** (Almirall, España).
- Presentaciones clínicas disponibles: comprimidos ranurados de 50 mg y 100 mg.
'''
    nb["cells"][15]["source"] = [l + "\n" for l in c15.splitlines()]

    # Celda 17: Caso 2
    cell_17_code = '''pregunta_2 = "¿Qué interacciones farmacocinéticas y precauciones enzimáticas con el citocromo CYP2D6 se describen para Paroxetina y Bupropión?"
resp_2, docs_2 = generar_respuesta_rag(pregunta_2, vector_store)
print(resp_2)
print("\\n" + "="*60)
verificar_caso("CASO 2", pregunta_2, ["INTER-CYP2D6-001", "MONO-PAROXETINA-003", "MONO-BUPROPION-004"], resp_2, docs_2, vector_store)
'''
    nb["cells"][17]["source"] = [l + "\n" for l in cell_17_code.splitlines()]

    # Celda 18: Simulación Caso 2 (redactada a mano con MONO-PAROXETINA-003 y sin síndrome serotoninérgico espurio)
    c18 = '''### 📋 Simulación de Respuesta Clínica con LLM (Ejemplo de Referencia Redactado a Mano - Caso 2)
> **Nota metodológica:** Transcripción redactada a mano como ejemplo clínico de referencia.

> **Referencias Técnicas:** `[INTER-CYP2D6-001]`, `[MONO-PAROXETINA-003]`, `[MONO-BUPROPION-004]`

**RESPUESTA CLÍNICA:**

**1. Juicio Clínico Inmediato:**
Tanto la Paroxetina `[MONO-PAROXETINA-003]` como el Bupropión `[MONO-BUPROPION-004]` son clasificados formalmente como **inhibidores potentes del citocromo CYP2D6** `[INTER-CYP2D6-001]`. Su coadministración o uso concomitante con sustratos de ventana estrecha genera riesgo crítico de toxicidad farmacológica.

**2. Mecanismo Farmacológico y Cinética:**
- Paroxetina ejerce inactivación dependiente de mecanismo sobre CYP2D6 `[INTER-CYP2D6-001]`, bloqueando el aclaramiento de fármacos que dependen de esta isoforma.
- Bupropión es metabolizado por CYP2B6 a hidroxibupropión `[MONO-BUPROPION-004]`, el cual junto con la molécula madre es un inhibidor potente de CYP2D6 `[INTER-CYP2D6-001]`.
- Al bloquear CYP2D6, se incrementan marcadamente los niveles séricos de antidepresivos tricíclicos, risperidona, aripiprazol y betabloqueantes como metoprolol `[INTER-CYP2D6-001]`. Asimismo, anulan la bioactivación de profármacos como tamoxifeno, codeína y tramadol `[INTER-CYP2D6-001]`.

**3. Pauta Operativa y Recomendaciones:**
- Si se coadministra con aripiprazol, reducir la dosis de aripiprazol al 50% `[INTER-CYP2D6-001]`.
- Si se combina con risperidona, monitorizar estrechamente síntomas extrapiramidales y valorar reducir dosis un 25-50% `[INTER-CYP2D6-001]`.
- Como alternativa antidepresiva más limpia ante sustratos sensibles de 2D6, preferir sertralina (≤ 100 mg/día) o escitalopram `[INTER-CYP2D6-001]`.

**4. Alertas de Seguridad y Trazabilidad Comercial:**
Alerta de caja negra por interacciones farmacocinéticas en pacientes polimedicados `[INTER-CYP2D6-001]`. Marcas: Paxil/Seroxat (GSK) `[MONO-PAROXETINA-003]`, Wellbutrin/Zyban (GSK) `[MONO-BUPROPION-004]`.
'''
    nb["cells"][18]["source"] = [l + "\n" for l in c18.splitlines()]

    # Celda 20: Caso 3
    cell_20_code = '''pregunta_3 = "¿Qué fármacos se recomiendan en primera línea para un episodio de manía aguda y por qué se desaconseja el uso de antidepresivos en monoterapia en depresión bipolar?"
resp_3, docs_3 = generar_respuesta_rag(pregunta_3, vector_store)
print(resp_3)
print("\\n" + "="*60)
verificar_caso("CASO 3", pregunta_3, ["GUIA-CANMAT-MANIA-001", "GUIA-CANMAT-DEP-002", "ALERTA-ISRS-VIRAJE-004"], resp_3, docs_3, vector_store)
'''
    nb["cells"][20]["source"] = [l + "\n" for l in cell_20_code.splitlines()]

    # Celda 21: Simulación Caso 3 (redactada a mano con lista completa y Level 1 Daño explicado)
    c21 = '''### 📋 Simulación de Respuesta Clínica con LLM (Ejemplo de Referencia Redactado a Mano - Caso 3)
> **Nota metodológica:** Transcripción redactada a mano como ejemplo clínico de referencia.

> **Referencias Técnicas:** `[GUIA-CANMAT-MANIA-001]`, `[GUIA-CANMAT-DEP-002]`, `[ALERTA-ISRS-VIRAJE-004]`, `[MONO-LITIO-005]`

**RESPUESTA CLÍNICA:**

**1. Juicio Clínico Inmediato:**
Para la manía bipolar aguda, las guías CANMAT/ISBD 2018 establecen como **Primera Línea con evidencia Level 1** tanto la monoterapia como la terapia combinada `[GUIA-CANMAT-MANIA-001]`. Los antidepresivos en monoterapia están formalmente desaconsejados por riesgo crítico de viraje maníaco con evidencia de **Level 1 (Daño)** `[GUIA-CANMAT-DEP-002]`, `[ALERTA-ISRS-VIRAJE-004]`.

**2. Mecanismo Farmacológico y Cinética:**
- En manía aguda: litio modula cascadas de GSK-3β e inositol `[MONO-LITIO-005]`; los antipsicóticos atípicos bloquean con alta afinidad los receptores D2 estriatales y 5-HT2A.
- En depresión bipolar: los antidepresivos tradicionales en monoterapia desestabilizan el tono monoaminérgico, induciendo ciclación rápida, hiperactividad dopaminérgica y estados mixtos disfóricos con alto riesgo suicida `[ALERTA-ISRS-VIRAJE-004]`.

**3. Pauta Operativa y Recomendaciones:**
- Primera Línea en Monoterapia para Manía `[GUIA-CANMAT-MANIA-001]`: Carbonato de Litio (litemia aguda 0.8-1.1 mEq/L), Divalproato de Sodio, Quetiapina, Aripiprazol, Risperidona o Asenapina.
- Primera Línea en Terapia Combinada `[GUIA-CANMAT-MANIA-001]`: Litio o Valproato asociado a un antipsicótico atípico (indicada en manía severa con síntomas psicóticos o agitación).
- Ante depresión bipolar: suspender inmediatamente cualquier antidepresivo en monoterapia `[ALERTA-ISRS-VIRAJE-004]` y optimizar estabilizadores (quetiapina, lurasidona, litio, lamotrigina) `[GUIA-CANMAT-DEP-002]`.

**4. Alertas de Seguridad y Trazabilidad Comercial:**
Alerta de seguridad crítica sobre viraje maníaco iatrogénico `[ALERTA-ISRS-VIRAJE-004]`. Etiqueta CANMAT: Level 1 (Daño / Recomendación en contra) `[GUIA-CANMAT-DEP-002]`. Litio disponible como Plenur (Faes Farma) `[MONO-LITIO-005]`.
'''
    nb["cells"][21]["source"] = [l + "\n" for l in c21.splitlines()]

    # Celda 23: Caso 4
    cell_23_code = '''pregunta_4 = "Un paciente en tratamiento con Haloperidol comienza con rigidez y temblor y se le prescribe medicación antiparkinsoniana como biperideno. ¿Cuál es el riesgo de cascada de prescripción y qué pauta de desprescripción se aconseja?"
resp_4, docs_4 = generar_respuesta_rag(pregunta_4, vector_store)
print(resp_4)
print("\\n" + "="*60)
verificar_caso("CASO 4", pregunta_4, ["CASC-HAL-BIP-001", "MONO-HALOPERIDOL-008"], resp_4, docs_4, vector_store)
'''
    nb["cells"][23]["source"] = [l + "\n" for l in cell_23_code.splitlines()]

    # Celda 24: Simulación Caso 4 (redactada a mano con matices: agravar/desenmascarar, distonía aguda, 1-2 mg/sem)
    c24 = '''### 📋 Simulación de Respuesta Clínica con LLM (Ejemplo de Referencia Redactado a Mano - Caso 4)
> **Nota metodológica:** Transcripción redactada a mano como ejemplo clínico de referencia.

> **Referencias Técnicas:** `[CASC-HAL-BIP-001]`, `[MONO-HALOPERIDOL-008]`

**RESPUESTA CLÍNICA:**

**1. Juicio Clínico Inmediato:**
La prescripción continuada de biperideno para tratar el parkinsonismo inducido por haloperidol constituye una **ALERTA CRÍTICA DE CASCADA DE PRESCRIPCIÓN IATROGÉNICA** `[CASC-HAL-BIP-001]`. La conducta de elección es la reducción del antipsicótico o su rotación, desaconsejando el tratamiento anticolinérgico crónico.

**2. Mecanismo Farmacológico y Cinética:**
- Haloperidol `[MONO-HALOPERIDOL-008]` a dosis medias-altas bloquea > 80% de los receptores dopaminérgicos D2 en la vía nigroestriatal, desencadenando pseudoparkinsonismo por disbalance colinérgico/dopaminérgico.
- El uso crónico de anticolinérgicos como biperideno añade toxicidad cognitiva, sedación, constipación grave y **puede agravar o desenmascarar discinesia tardía** al hipersensibilizar los receptores D2 `[CASC-HAL-BIP-001]`.
- Excepción: el uso de biperideno (parenteral/oral) es legítimo y de primera línea en la urgencia ante **distonía aguda**, pero debe limitarse a pocos días y no mantenerse como profilaxis crónica.

**3. Pauta Operativa y Algoritmo de Desprescripción:**
- Paso 1: Evaluar reducción progresiva de la dosis de haloperidol `[MONO-HALOPERIDOL-008]`.
- Paso 2: Rotar a un antipsicótico de segunda generación con menor afinidad D2 y bajo perfil extrapiramidal (ej. quetiapina u olanzapina) `[CASC-HAL-BIP-001]`.
- Paso 3: Retirada gradual del anticolinérgico: reducir biperideno a razón de **1 a 2 mg por semana** para prevenir rebote colinérgico (diaforesis, náuseas, agitación) `[CASC-HAL-BIP-001]`.

**4. Alertas de Seguridad y Trazabilidad Comercial:**
Alerta de seguridad por cascada iatrogénica `[CASC-HAL-BIP-001]`. Haloperidol: Haldol (Janssen) `[MONO-HALOPERIDOL-008]`. Biperideno: Akineton.
'''
    nb["cells"][24]["source"] = [l + "\n" for l in c24.splitlines()]

    sanitize_outputs(nb)

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    ejecutar_notebook_en_python(nb, "04_rag_psicofarmacologia.ipynb")
    print("✓ 04_rag_psicofarmacologia.ipynb completado y validado.")


if __name__ == "__main__":
    update_notebook_02()
    update_notebook_00()
    update_notebook_01()
    update_notebook_03()
    update_notebook_04()
    print("\n========================================================")
    print("✓ TODOS LOS NOTEBOOKS (00 AL 04) HAN SIDO ACTUALIZADOS Y EJECUTADOS")
    print("========================================================")
