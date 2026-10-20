#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sanitize_notebooks.py
Sanitiza y normaliza notebooks/01, 02, 03, 04 para:
1. Eliminar cualquier ruta absoluta o mención de directorios de usuario.
2. Asegurar prints con rutas relativas limpias.
3. Garantizar que todas las celdas se ejecuten 100% sin error.
"""

import json
from pathlib import Path


NOTEBOOKS_DIR = Path("notebooks")


def sanitize_01():
    path = NOTEBOOKS_DIR / "01_preprocesamiento_jufe.ipynb"
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Celda 1: Reemplazar %pip por chequeo de imports limpio
    nb["cells"][1]["source"] = [
        "# Comprobación de librerías esenciales\n",
        "import pandas\n",
        "import pdfplumber\n",
        "print('✓ Entorno y dependencias verificadas (pandas, pdfplumber).')\n"
    ]
    nb["cells"][1]["outputs"] = []

    # Celda 2: Rutas relativas
    nb["cells"][2]["source"] = [
        "import os\n",
        "import sys\n",
        "import re\n",
        "import unicodedata\n",
        "from pathlib import Path\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "# Detectar la ruta raíz del proyecto de forma dinámica y robusta\n",
        "current_dir = Path.cwd()\n",
        "project_root = current_dir if (current_dir / 'data').exists() else current_dir.parent\n",
        "if str(project_root) not in sys.path:\n",
        "    sys.path.insert(0, str(project_root))\n",
        "\n",
        "DATA_RAW = project_root / 'data' / 'raw'\n",
        "DATA_PROCESSED = project_root / 'data' / 'processed'\n",
        "DATA_PROCESSED.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "from src.extractors.jufe_extractor import JufeExtractor\n",
        "from src.extractors.pdf_extractor import CachedPDFExtractor\n",
        "from src.enrichers.atc_enricher import ATCEnricher\n",
        "\n",
        "print('Entorno configurado correctamente. Directorio de datos: data/processed')\n"
    ]
    nb["cells"][2]["outputs"] = []

    # Celda 4:
    nb["cells"][4]["outputs"] = []

    # Celda 12: Exportación con ruta relativa
    nb["cells"][12]["source"] = [
        "# Validaciones de integridad\n",
        "assert df_jufe['farmaco'].isnull().sum() == 0, 'Error: Hay fármacos sin nombre'\n",
        "assert df_jufe['atc_codigo'].isnull().sum() == 0, 'Error: Hay códigos ATC nulos'\n",
        "assert len(df_jufe) >= 5, 'Error: Catálogo de fármacos insuficiente'\n",
        "assert len(df_jufe_efectos) >= 20, 'Error: Catálogo de efectos adversos insuficiente'\n",
        "\n",
        "# Rutas de salida normalizadas relativas\n",
        "path_farmacos = DATA_PROCESSED / 'jufe_farmacos_clean.csv'\n",
        "path_efectos = DATA_PROCESSED / 'jufe_efectos_adversos_clean.csv'\n",
        "\n",
        "df_jufe.to_csv(path_farmacos, index=False, encoding='utf-8')\n",
        "df_jufe_efectos.to_csv(path_efectos, index=False, encoding='utf-8')\n",
        "\n",
        "rel_farmacos = path_farmacos.relative_to(project_root)\n",
        "rel_efectos = path_efectos.relative_to(project_root)\n",
        "print(f'Dataset principal de Jufe exportado a: {rel_farmacos} ({len(df_jufe)} fármacos)')\n",
        "print(f'Dataset de eventos adversos de Jufe exportado a: {rel_efectos} ({len(df_jufe_efectos)} pares)')\n"
    ]
    nb["cells"][12]["outputs"] = []

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"✓ Sanitizado {path.name}")


def sanitize_02():
    path = NOTEBOOKS_DIR / "02_preprocesamiento_stahl.ipynb"
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Celda 1
    nb["cells"][1]["source"] = [
        "import os\n",
        "import sys\n",
        "import re\n",
        "import unicodedata\n",
        "from pathlib import Path\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "# Detectar la ruta raíz del proyecto de forma dinámica y robusta\n",
        "current_dir = Path.cwd()\n",
        "project_root = current_dir if (current_dir / 'data').exists() else current_dir.parent\n",
        "if str(project_root) not in sys.path:\n",
        "    sys.path.insert(0, str(project_root))\n",
        "\n",
        "DATA_RAW = project_root / 'data' / 'raw'\n",
        "DATA_PROCESSED = project_root / 'data' / 'processed'\n",
        "DATA_PROCESSED.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "from src.enrichers.atc_enricher import ATCEnricher\n",
        "\n",
        "print('Entorno configurado correctamente. Directorio de datos: data/processed')\n"
    ]
    nb["cells"][1]["outputs"] = []

    # Celda 11
    nb["cells"][11]["source"] = [
        "out_farmacos = DATA_PROCESSED / 'stahl_farmacos_clean.csv'\n",
        "out_eventos = DATA_PROCESSED / 'stahl_eventos_adversos_clean.csv'\n",
        "\n",
        "df_stahl.to_csv(out_farmacos, index=False, encoding='utf-8')\n",
        "df_eventos_stahl.to_csv(out_eventos, index=False, encoding='utf-8')\n",
        "\n",
        "rel_farmacos = out_farmacos.relative_to(project_root)\n",
        "rel_eventos = out_eventos.relative_to(project_root)\n",
        "print(f'Fármacos de Stahl guardados en: {rel_farmacos} ({len(df_stahl)} fármacos)')\n",
        "print(f'Eventos adversos de Stahl guardados en: {rel_eventos} ({len(df_eventos_stahl)} eventos)')\n"
    ]
    nb["cells"][11]["outputs"] = []

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"✓ Sanitizado {path.name}")


def sanitize_03():
    path = NOTEBOOKS_DIR / "03_preprocesamiento_canmat.ipynb"
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Celda 1
    nb["cells"][1]["source"] = [
        "import os\n",
        "import sys\n",
        "import re\n",
        "import unicodedata\n",
        "from pathlib import Path\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "# Detectar la ruta raíz del proyecto de forma dinámica y robusta\n",
        "current_dir = Path.cwd()\n",
        "project_root = current_dir if (current_dir / 'data').exists() else current_dir.parent\n",
        "if str(project_root) not in sys.path:\n",
        "    sys.path.insert(0, str(project_root))\n",
        "\n",
        "DATA_RAW = project_root / 'data' / 'raw'\n",
        "DATA_PROCESSED = project_root / 'data' / 'processed'\n",
        "DATA_PROCESSED.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "from src.extractors.canmat_extractor import CANMATExtractor\n",
        "from src.downloaders.guidelines_downloader import GuidelinesDownloader\n",
        "\n",
        "print('Entorno configurado correctamente. Directorio de datos: data/processed')\n"
    ]
    nb["cells"][1]["outputs"] = []

    # Celda 9
    nb["cells"][9]["source"] = [
        "out_canmat = DATA_PROCESSED / 'canmat_recomendaciones_clean.csv'\n",
        "out_alertas = DATA_PROCESSED / 'canmat_alertas_clinicas_clean.csv'\n",
        "\n",
        "df_canmat.to_csv(out_canmat, index=False, encoding='utf-8')\n",
        "df_alertas_canmat.to_csv(out_alertas, index=False, encoding='utf-8')\n",
        "\n",
        "rel_canmat = out_canmat.relative_to(project_root)\n",
        "rel_alertas = out_alertas.relative_to(project_root)\n",
        "print(f'Recomendaciones CANMAT guardadas en: {rel_canmat} ({len(df_canmat)} filas)')\n",
        "print(f'Alertas clínicas CANMAT guardadas en: {rel_alertas} ({len(df_alertas_canmat)} filas)')\n"
    ]
    nb["cells"][9]["outputs"] = []

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"✓ Sanitizado {path.name}")


def sanitize_04():
    path = NOTEBOOKS_DIR / "04_rag_psicofarmacologia.ipynb"
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Celda 1
    nb["cells"][1]["source"] = [
        "import os\n",
        "import sys\n",
        "import json\n",
        "import re\n",
        "from pathlib import Path\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import requests\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.metrics.pairwise import cosine_similarity\n",
        "\n",
        "# Detección dinámica y robusta de rutas\n",
        "current_dir = Path.cwd()\n",
        "project_root = current_dir if (current_dir / 'data').exists() else current_dir.parent\n",
        "if str(project_root) not in sys.path:\n",
        "    sys.path.insert(0, str(project_root))\n",
        "\n",
        "DATA_PROCESSED = project_root / 'data' / 'processed'\n",
        "DIR_PRETRAINING = project_root / 'pre_training'\n",
        "print('Entorno RAG configurado.\\n- Directorio de datos: data/processed\\n- Corpus pre_training: pre_training')\n"
    ]
    nb["cells"][1]["outputs"] = []

    # Celda 25
    nb["cells"][25]["source"] = [
        "json_path = DATA_PROCESSED / 'rag_knowledge_base.json'\n",
        "with open(json_path, 'w', encoding='utf-8') as f:\n",
        "    json.dump(kb_documents, f, indent=2, ensure_ascii=False)\n",
        "\n",
        "rel_json = json_path.relative_to(project_root)\n",
        "print(f'Base de conocimiento RAG exportada con éxito ({len(kb_documents)} documentos) en: {rel_json}')\n"
    ]
    nb["cells"][25]["outputs"] = []

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"✓ Sanitizado {path.name}")


if __name__ == "__main__":
    sanitize_01()
    sanitize_02()
    sanitize_03()
    sanitize_04()
    print("✓ Todos los notebooks fueron sanitizados exitosamente.")
