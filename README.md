# Sofia - Asistente Clínico Inteligente y RAG de Psicofarmacología

**Sofia** es un sistema inteligente de soporte a la decisión clínica, farmacovigilancia y generación asistida por recuperación (RAG) especializado en psiquiatría y psicofarmacología (antidepresivos, antipsicóticos, estabilizadores del ánimo, ansiolíticos, interacciones enzimáticas CYP450 y prevención de cascadas de prescripción).

---

## 📚 Fuentes Clínicas Principales

1. **Dra. Gabriela Jufe:** *Psicofarmacología Práctica* (640 páginas con perfiles de fármacos, rangos terapéuticos, poblaciones especiales y efectos colaterales).
2. **Dr. Stephen Stahl:** *Prescriber's Guide* (Perfil de receptores, interacciones con citocromos CYP1A2, 2D6, 3A4 y perlas clínicas).
3. **Guías CANMAT e ISBD 2018 / 2016:** Directrices oficiales basadas en evidencia para Trastorno Bipolar y Trastorno Depresivo Mayor (líneas de tratamiento, niveles de evidencia y alertas de viraje maníaco).
4. **Bases de Datos Médicas y Regulatorias:**
   - **CIMA (AEMPS):** Marcas comerciales oficiales, presentaciones y códigos ATC en español.
   - **OpenFDA:** Fichas técnicas, advertencias de caja negra (*boxed warnings*) y alertas de seguridad.
   - **RxNav / NIH:** Identificadores RxCUI y verificación determinista de interacciones farmacológicas.
   - **Europe PMC:** Acceso y descarga de artículos y guías clínicas en acceso abierto.

---

## 🏗️ Arquitectura Modular

```
Sofia/
├── data/
│   ├── raw/                  # Documentos PDF originales (Jufe, CANMAT BDI-20-97)
│   ├── cache/                # Caché comprimida en disco (JSON.gz de páginas y respuestas API)
│   └── processed/            # Datasets limpios generados (CSV y JSON para el RAG)
├── notebooks/                # Flujos interactivos en Jupyter Notebooks
│   ├── 01_preprocesamiento_jufe.ipynb
│   ├── 02_preprocesamiento_stahl.ipynb
│   ├── 03_preprocesamiento_canmat.ipynb
│   └── 04_rag_psicofarmacologia.ipynb
├── src/
│   ├── downloaders/          # Clientes de APIs médicas y descargadores de guías
│   │   ├── medical_apis.py       # OpenFDA, CIMA, RxNav y Europe PMC
│   │   └── guidelines_downloader.py  # Descarga de CANMAT, APA, NICE
│   ├── extractors/           # Extractores robustos de PDFs con caché
│   │   ├── pdf_extractor.py      # CachedPDFExtractor (acelera cargas x100)
│   │   ├── jufe_extractor.py     # Parser clínico de Jufe
│   │   └── canmat_extractor.py   # Extractor de tablas de recomendaciones CANMAT
│   └── enrichers/            # Normalización y clasificación ATC
│       └── atc_enricher.py       # Clasificación OMS y catálogo de marcas
├── scripts/
│   └── download_and_extract.py   # CLI unificado para ejecutar el pipeline de datos
├── tests/
│   └── test_downloaders_and_extractors.py  # Tests unitarios del sistema
└── requirements.txt
```

---

## 🚀 Inicio Rápido

### 1. Instalación de dependencias
```bash
pip install -r requirements.txt
```

### 2. Ejecutar la descarga y extracción unificada
Para ejecutar todo el pipeline de extracción con caché, generación de CSVs limpios y construcción de la base de conocimiento RAG:

```bash
python scripts/download_and_extract.py --all
```

Opciones disponibles en el CLI:
- `--all`: Procesa todas las fuentes (Jufe, CANMAT, Stahl, ATC, RAG).
- `--guidelines`: Comprueba y descarga guías oficiales si faltan en `data/raw/`.
- `--jufe`: Extrae monografías y efectos adversos de Dra. Jufe.
- `--canmat`: Extrae tablas y alertas clínicas de guías CANMAT.
- `--stahl`: Procesa monografías y citocromos CYP de Dr. Stahl.
- `--fetch-api <farmaco>`: Consulta en vivo CIMA, OpenFDA y RxNav (ej. `--fetch-api sertralina`).

### 3. Ejecución de Tests
```bash
python -m unittest tests/test_downloaders_and_extractors.py
```

### 4. Flujo interactivo con Jupyter
Puedes abrir cualquiera de los notebooks en `notebooks/`:
- `01_preprocesamiento_jufe.ipynb`: Inspección visual del catálogo de Jufe y efectos adversos.
- `02_preprocesamiento_stahl.ipynb`: Análisis de citocromos y perlas clínicas.
- `03_preprocesamiento_canmat.ipynb`: Algoritmos de decisión y alertas de contraindicación.
- `04_rag_psicofarmacologia.ipynb`: Motor RAG con almacén vectorial y consultas clínicas con LLM (Ollama).
