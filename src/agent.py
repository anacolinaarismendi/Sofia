import os
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types
from src.api_client import MedicalAPIClient

# Cargar variables de entorno (el archivo .env)
load_dotenv()

logger = logging.getLogger(__name__)


class SofiaAgent:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY no encontrado. Asegúrate de tener el archivo .env configurado.")

        self.client = genai.Client(api_key=api_key)
        self.api_client = MedicalAPIClient()
        self.model = "gemini-2.5-pro"

        # Definición de la herramienta de Sofía
        self.check_drugs_tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="check_drug_interactions",
                    description="Busca interacciones reales entre una lista de medicamentos usando la base de datos médica.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "drug_names": types.Schema(
                                type=types.Type.ARRAY,
                                items=types.Schema(type=types.Type.STRING),
                                description="Lista de nombres de medicamentos (ej. ['ibuprofeno', 'litio'])"
                            )
                        },
                        required=["drug_names"]
                    )
                )
            ]
        )

        # La personalidad y reglas de Sofía
        self.system_instruction = (
            "Tu nombre es Sofía, eres un asistente clínico experto en farmacología. "
            "Tu objetivo es ayudar a los médicos a identificar interacciones medicamentosas. "
            "REGLAS ESTRICTAS:\n"
            "1. NUNCA inventes o asumas una interacción medicamentosa.\n"
            "2. SIEMPRE usa la herramienta 'check_drug_interactions' para consultar la base de datos.\n"
            "3. Responde de forma empática, profesional y en ESPAÑOL.\n"
            "4. Sé concisa y estructura tu respuesta con viñetas.\n"
            "5. Si la base de datos dice que no hay interacciones, informa al médico, pero recuérdale que mantenga su juicio clínico."
        )

    def process_query(self, user_query: str) -> str:
        """Procesa la consulta del usuario usando la IA y la herramienta."""
        try:
            # Primera llamada: forzamos el uso de la herramienta (mode="ANY").
            # Esto es lo que realmente hace cumplir la regla 2 del system_instruction;
            # sin esto, el modelo puede optar por responder sin consultar la base de datos.
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_query,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    tools=[self.check_drugs_tool],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="ANY")
                    ),
                    temperature=0.1
                )
            )

            if not response.candidates:
                return "Sofía no pudo generar una respuesta. Intenta reformular tu consulta."

            if not response.function_calls:
                # Con mode="ANY" esto no debería pasar, pero si el SDK cambia de
                # comportamiento no queremos devolver un response.text que puede no existir.
                return response.text or "Sofía no encontró una herramienta adecuada para responder."

            function_call = response.function_calls[0]

            if function_call.name != "check_drug_interactions":
                return f"Sofía intentó usar una herramienta desconocida: {function_call.name}."

            args = function_call.args or {}
            drug_names = args.get("drug_names", [])

            if not drug_names:
                return "Sofía no recibió ningún nombre de medicamento para consultar."

            logger.info("Sofía está consultando la base de datos para: %s", ", ".join(drug_names))

            # Ejecutar la búsqueda en la API
            api_result = self.api_client.check_drug_names(drug_names)

            # Enviar el resultado de vuelta a Sofía
            function_response_part = types.Part.from_function_response(
                name="check_drug_interactions",
                response={"result": api_result}
            )

            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text=user_query)]),
                response.candidates[0].content,
                types.Content(role="user", parts=[function_response_part])
            ]

            # Segunda llamada: SIN forzar tool_config, para que Sofía pueda
            # redactar la respuesta final en texto libre en vez de intentar
            # llamar a la herramienta de nuevo en un bucle.
            final_response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    temperature=0.1
                )
            )

            return final_response.text or "Sofía consultó la base de datos pero no pudo redactar una respuesta."

        except Exception as e:
            logger.exception("Error procesando la consulta de Sofía")
            return f"Error procesando la consulta: {str(e)}"