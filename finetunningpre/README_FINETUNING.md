# Guía de Fine-Tuning Clínico para Modelos Qwen (Proyecto Sofia)

Esta carpeta contiene el dataset, las herramientas de auditoría y las especificaciones para realizar el ajuste fino (*fine-tuning*) supervisado (SFT) de un modelo de lenguaje de código abierto de la familia **Qwen** (como `Qwen2.5-7B-Instruct` o `Qwen2.5-3B-Instruct`), adaptándolo para hablar y razonar con el nivel de una **médica especialista en psicofarmacología y farmacovigilancia**.

---

## 1. Contenido de esta Carpeta

- **`dataset_sofia_medical_qwen.jsonl`**: El archivo de datos principal con ejemplos de diálogos complejos médico-a-médico en formato nativo **ChatML**.
- **`build_finetuning_dataset.py`**: Script en Python para regenerar, ampliar o personalizar el corpus de entrenamiento.
- **`validate_finetuning_dataset.py`**: Script de control de calidad y auditoría de métricas de tokens y esquemas de roles.
- **`Modelfile.example`**: Plantilla de configuración de Ollama con parámetros clínicos optimizados (temperatura 0.2, top_p 0.85, ventana 4096).

---

## 2. Estructura del Formato ChatML

Cada línea del archivo `dataset_sofia_medical_qwen.jsonl` es un objeto JSON con la estructura:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Eres Sofia, médica especialista en psicofarmacología clínica, psiquiatría basada en la evidencia y farmacovigilancia..."
    },
    {
      "role": "user",
      "content": "Pregunta técnica del médico sobre caso clínico, dosificación, CYP450, cascadas o CANMAT..."
    },
    {
      "role": "assistant",
      "content": "Respuesta médica estructurada: Juicio Clínico Inmediato + Razonamiento Farmacológico/Cinético + Pauta Operativa y Dosis + Fuentes Canónicas (Jufe, Stahl, CANMAT)..."
    }
  ]
}
```

---

## 3. Flujo Recomendado de Entrenamiento (Fine-Tuning con Unsloth)

**Unsloth** es la herramienta más rápida y eficiente en memoria para entrenar modelos Qwen2.5 mediante QLoRA en Google Colab (GPU T4 gratuita) o en servidores locales con GPU NVIDIA:

### Paso 1: Instalación de Unsloth
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

### Paso 2: Script de Entrenamiento Rápido en Python
```python
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True, # QLoRA 4-bit para bajo consumo de VRAM
)

# Configurar adaptadores LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Cargar el dataset de Sofia
dataset = load_dataset("json", data_files="dataset_sofia_medical_qwen.jsonl", split="train")

# Entrenar con SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="messages",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60, # Ajustar épocas según el tamaño del corpus
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="outputs",
    ),
)
trainer.train()

# Exportar directamente a formato GGUF para Ollama (Cuantización Q4_K_M)
model.save_pretrained_gguf("sofia_qwen_gguf", tokenizer, quantization_method="q4_k_m")
```

---

## 4. Cómo Desplegar el Modelo Entrenado en Ollama

Una vez generado el archivo GGUF (por ejemplo `sofia_qwen_gguf/unsloth.Q4_K_M.gguf`):

### 1. Crear un `Modelfile`
Edita el archivo `Modelfile` apuntando al GGUF:
```dockerfile
FROM ./sofia_qwen_gguf/unsloth.Q4_K_M.gguf

PARAMETER temperature 0.2
PARAMETER top_p 0.85
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 4096

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"

SYSTEM """Eres Sofia, médica especialista en psicofarmacología clínica, psiquiatría basada en la evidencia y farmacovigilancia. Responde con lenguaje médico profesional y fundamenta en Jufe, Stahl y CANMAT."""

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
```

### 2. Crear y Ejecutar en Ollama
```bash
# Crear el modelo en el registro local de Ollama
ollama create sofia-qwen -f Modelfile

# Ejecutar una consulta médica de prueba
ollama run sofia-qwen "Colega, paciente bipolar que toma litio con litemia en 0.8 mEq/L consulta por cefalea tensional. ¿Qué analgésico está formalmente contraindicado y por qué?"
```

---

## 5. Auditoría y Validación de Datos

Para verificar la integridad de cualquier versión nueva del dataset:
```bash
python validate_finetuning_dataset.py
```
El script valida la sintaxis JSON, que no existan campos vacíos, calcula la distribución de palabras y tokens y confirma el cumplimiento del formato ChatML.
