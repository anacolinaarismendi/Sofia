# Sofía - AI Medical Assistant

Asistente clínico inteligente impulsado por IA para la detección de interacciones farmacológicas.

Proyecto desarrollado como parte de **AudítaLas**, un prototipo de asistente de cumplimiento normativo y bioseguridad para laboratorios farmacéuticos, con [Laboratorio Fritze](https://laboratorio-fritze.vercel.app) como cliente ficticio de referencia.

> ⚠️ **Aviso:** Este es un prototipo académico con fines educativos. Las interacciones que detecta provienen de una base de datos simulada (mock) y **no reemplazan el juicio clínico profesional**. No debe usarse para tomar decisiones médicas reales.

## Características

- Conversación en lenguaje natural con Sofía, un agente basado en Gemini (`gemini-2.5-pro`) especializado en farmacología.
- Búsqueda de interacciones entre medicamentos mediante *function calling*, apoyada en una base de datos simulada de RxCUIs e interacciones conocidas.
- Reglas estrictas de comportamiento: Sofía nunca inventa una interacción, siempre consulta la base de datos antes de responder.

## Stack técnico

- Python 3.10+
- [google-genai](https://pypi.org/project/google-genai/) — SDK para la API de Gemini
- [python-dotenv](https://pypi.org/project/python-dotenv/) — carga de variables de entorno

## Instalación

1. Clona el repositorio y entra en la carpeta del proyecto.
2. Crea y activa un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate   # En Windows: venv\Scripts\activate
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Configura tu clave de la API de Gemini:
   ```bash
   cp .env.example .env
   ```
   Y reemplaza `tu_clave_aqui` por tu clave real (puedes generarla en [Google AI Studio](https://aistudio.google.com/apikey)).

   Si prefieres no hacerlo manualmente, puedes omitir este paso: al ejecutar el programa por primera vez, `main.py` te la pedirá automáticamente.

## Uso

```bash
python src/main.py
```

Ejemplos de consultas de prueba:
- "¿Hay interacción entre Ibuprofeno y Litio?"
- "Aspirina y Warfarina"
- "Sildenafil y Nitroglicerina"

Escribe `salir` para terminar la sesión.

## Estructura del proyecto

```
.
├── src/
│   ├── __init__.py
│   ├── agent.py         # Lógica del agente Sofía (Gemini + function calling)
│   ├── api_client.py     # Cliente simulado de bases de datos médicas (RxNav/OpenFDA)
│   └── main.py           # Punto de entrada / interfaz de terminal
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Limitaciones conocidas

- La base de datos de medicamentos e interacciones es simulada (mock), no una conexión real a RxNav u OpenFDA.
- Sofía no mantiene memoria entre consultas dentro de la sesión de terminal.
