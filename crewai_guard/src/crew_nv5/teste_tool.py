import os
import sys
import warnings
import pprint # Apenas para imprimir a string de forma mais legível se for longa

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar a ferramenta
from crew_nv5.custom_tool import WebsiteContentScraperTool

# Suprimir avisos
warnings.filterwarnings("ignore", message="There is no current event loop", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Using Pydantic V1 type checking", category=RuntimeWarning)

# --- Teste da Ferramenta ---
if __name__ == "__main__":
    # URL de teste
    test_url = "https://blog.crewai.com/enhancing-crewai-with-copilotkit-integration/"

    print(f"\n--- Testando WebsiteContentScraperTool com URL: {test_url} ---")

    # Instanciar a ferramenta
    try:
        scraper_tool = WebsiteContentScraperTool()
        print("Ferramenta inicializada com sucesso.")
    except Exception as e:
        print(f"Erro ao inicializar a ferramenta: {e}")
        sys.exit(1)

    # Chamar o método _run
    try:
        # Passar apenas os argumentos definidos no args_schema (website_url)
        result = scraper_tool._run(website_url=test_url)
        print("\n--- Resultado da Execução da Ferramenta ---")

        # Verificar se o resultado é uma string e não começa com "[Tool Error]"
        if isinstance(result, str) and not result.startswith("[Tool Error]"):
             print("[INFO] Ferramenta retornou uma string (conteúdo raspado).")
             print(f"Início do conteúdo ({len(result)} caracteres):\n---")
             # Usar pprint para strings longas pode ajudar na legibilidade do terminal
             pprint.pprint(result[:1000] + ('...' if len(result) > 1000 else ''))
             print("---")
        elif isinstance(result, str):
             print(f"[ERRO] Ferramenta retornou uma mensagem de erro:\n{result}")
        else:
             print(f"[ERRO] Ferramenta retornou um tipo inesperado: {type(result)}")


    except Exception as e:
        print(f"\n--- Erro durante a execução de _run ---")
        import traceback
        traceback.print_exc()
        print(f"Erro: {e}")