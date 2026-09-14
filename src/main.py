import os
import sys
from getpass import getpass

# Directorio raíz del proyecto (usado tanto para sys.path como para el .env)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
ENV_PATH = os.path.join(ROOT_DIR, ".env")

from src.agent import SofiaAgent
from dotenv import load_dotenv


def check_env():
    """Asegura que el usuario tenga configurada su API Key."""
    if os.path.exists(ENV_PATH):
        return

    print("\n[!] Archivo .env no encontrado.")

    key = ""
    while not key:
        key = getpass("Por favor, introduce tu GEMINI_API_KEY: ").strip()
        if not key:
            print("[!] La clave no puede estar vacía.")

    with open(ENV_PATH, "w") as f:
        f.write(f"GEMINI_API_KEY={key}\n")

    # Clave: sin esto, la variable recién escrita no está disponible en
    # este proceso, porque load_dotenv() de agent.py ya se ejecutó al
    # importar SofiaAgent, ANTES de que este archivo .env existiera.
    load_dotenv(dotenv_path=ENV_PATH, override=True)

    print("[+] Archivo .env configurado correctamente.\n")


def main():
    print("=" * 50)
    print(" 🩺 Bienvenido al Sistema Sofía AI 🩺")
    print("=" * 50)
    print("Despertando a Sofía...")

    check_env()

    try:
        agent = SofiaAgent()
    except Exception as e:
        print(f"Error inicializando a Sofía: {e}")
        return

    print("\n¡Sofía está lista! Escribe tu consulta clínica.")
    print("Prueba consultar por pares de prueba como:")
    print(" - Ibuprofeno y Litio")
    print(" - Aspirina y Warfarina")
    print(" - Sildenafil y Nitroglicerina")
    print("(Escribe 'salir' para terminar)\n")

    while True:
        try:
            query = input("🧑‍⚕️ Doctor/a: ")
            if query.lower() in ['salir', 'exit', 'quit']:
                print("Sofía: Ha sido un placer ayudarte. ¡Hasta pronto!")
                break

            if not query.strip():
                continue

            print("\n🤖 Sofía procesando...")
            response = agent.process_query(query)

            print("-" * 50)
            print(f"🤖 Sofía:\n{response}")
            print("-" * 50)
            print()

        except KeyboardInterrupt:
            print("\nSofía: Sesión terminada de emergencia. ¡Hasta pronto!")
            break


if __name__ == "__main__":
    main()