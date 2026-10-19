---
id: "MANIF-EMBED-000"
tipo: "especificacion_tecnica"
titulo: "Manifiesto y Especificación Técnica del Corpus pre_training"
version: "1.0.0"
proyecto: "Sofia - Asistente Clínico y RAG de Psicofarmacología"
autor: "Equipo de Arquitectura y Farmacología Clínica"
---

# Manifiesto y Guía de Ingestión para Modelos de Embeddings y Sistemas de Voz

## 1. Propósito del Corpus
El directorio `pre_training/` reúne el conocimiento clínico estructurado del proyecto **Sofia** (Dra. Gabriela Jufe, Dr. Stephen Stahl, Guías CANMAT/ISBD, clasificación ATC de la OMS y catálogo de marcas y laboratorios). Este material ha sido convertido y curado con un estándar unificado de **modalidad dual (verbal y escrita)**:
1. **Comunicación Verbal (Voice AI / Speech-to-Text / TTS):** Cada archivo contiene un bloque formal `Voice Snippet` (Resumen Ejecutivo Verbalizable) redactado en prosa fluida, sin tablas multidimensionales ni abreviaturas telegráficas, listo para ser sintetizado a audio en menos de 40 segundos sin distorsión prosódica.
2. **Consultas Escritas y Recuperación Semántica (RAG / Embeddings):** Cada archivo profundiza en la farmacodinamia, farmacocinética, interacciones CYP450, cascadas de prescripción y algoritmos de decisión, calibrado en longitud semántica (400 a 900 tokens) para evitar truncamientos y maximizar la precisión de recuperación (*recall* y *MRR*).

---

## 2. Esquema Estándar de Metadatos (YAML Frontmatter)
Cada documento inicia obligatoriamente con un bloque YAML estandarizado que permite el filtrado híbrido determinista (*metadata filtering*) antes o después de la búsqueda vectorial densa:

| Campo | Tipo | Descripción | Ejemplo |
| :--- | :--- | :--- | :--- |
| `id` | `string` | Identificador único del chunk | `"MONO-SERTRALINA-001"` |
| `tipo` | `enum` | Tipo de entidad médica | `monografia_farmacologica`, `interaccion_cyp450`, `cascada_prescripcion`, `guia_clinica`, `alerta_seguridad` |
| `farmaco` / `farmacos_involucrados` | `list[string]` | Principios activos mencionados | `["Sertralina"]` o `["Haloperidol", "Biperideno"]` |
| `atc_codigo` / `atc_codigos` | `list[string]` | Códigos ATC oficiales de la OMS | `["N06AB06"]` |
| `enzimas_cyp` | `list[string]` | Citocromos hepáticos vinculados | `["CYP2D6", "CYP1A2", "CYP3A4"]` |
| `fuentes_primarias` | `list[string]` | Libros y directrices que avalan el texto | `["Jufe - Psicofarmacología Práctica", "Stahl"]` |
| `nivel_evidencia` | `string` | Grado de evidencia científica | `"Level 1"`, `"Level 2"`, `"Guía Clínica"`, `"Alerta Regulatoria"` |
| `nivel_riesgo` | `enum` | Severidad clínica de la alerta | `Bajo`, `Moderado`, `Alto`, `Crítico / Emergencia` |
| `palabras_clave` | `list[string]` | Descriptores DeCS/MeSH en español | `["depresión", "inhibición enzimática", "parkinsonismo"]` |

---

## 3. Estrategia de Ingestión y Pipeline de Vectorización

### Paso 1: Extracción de Payload y Texto para el Embedding
Para maximizar la densidad semántica, se recomienda componer el texto de entrada al modelo de embeddings combinando el título, las palabras clave y el cuerpo completo, preservando el YAML como metadatos de la base de datos vectorial:

```python
import yaml

def parse_markdown_chunk(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    parts = text.split("---", 2)
    frontmatter = yaml.safe_load(parts[1])
    body = parts[2].strip()
    
    # Texto optimizado para vectorización
    embedding_text = f"TÍTULO: {frontmatter.get('titulo', frontmatter.get('id'))}\n" \
                     f"ENTIDADES: {', '.join(frontmatter.get('farmacos_involucrados', []))}\n" \
                     f"PALABRAS CLAVE: {', '.join(frontmatter.get('palabras_clave', []))}\n\n" \
                     f"{body}"
                     
    return frontmatter, embedding_text, body
```

### Paso 2: Modelos de Embeddings Recomendados
1. **Ollama Local (Entorno Sofia):** `embeddinggemma:latest` (o `bge-m3:latest` para embeddings densos multilingües con longitud de contexto de hasta 8192 tokens).
2. **Servicios Cloud / OpenAI:** `text-embedding-3-small` o `text-embedding-3-large` con dimensión configurada en 1536 o 3072.
3. **Métrica de Distancia Recomendada:** Distancia de Coseno (*Cosine Similarity*).

### Paso 3: Mapeo de Payload en Base de Datos Vectorial (Qdrant / Chroma / SQLite-vec)
Almacenar en el vector store:
- `vector`: Array flotante generado por el embedding.
- `payload`:
  - `id`: `frontmatter["id"]`
  - `tipo`: `frontmatter["tipo"]`
  - `farmacos`: `frontmatter.get("farmacos_involucrados", [frontmatter.get("farmaco")])`
  - `atc`: `frontmatter.get("atc_codigo", frontmatter.get("atc_codigos"))`
  - `voice_snippet`: Texto extraído de la sección `## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)`
  - `contenido_completo`: Texto íntegro del Markdown.
