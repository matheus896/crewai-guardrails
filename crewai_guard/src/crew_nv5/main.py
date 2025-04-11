"""
Nível 5: Ferramenta Customizada + Guardrails em Callback com Pydantic.

Este exemplo demonstra:
1. Uso de uma ferramenta customizada (WebsiteContentScraperTool) por um agente.
2. Separação de responsabilidades: um agente raspa, outro estrutura.
3. Validação explícita da saída estruturada usando Guardrails AI
   (Guard.for_pydantic) dentro de um callback da tarefa final.
"""

import os
from dotenv import load_dotenv
import sys
from typing import Optional, Union, Type, Dict, List
import warnings
import json
import pprint

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Imports Pydantic ---
# Mover a definição do modelo Pydantic para cá ou importar
# Para organização, vamos defini-lo aqui.
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

# --- Imports do CrewAI ---
from crewai import Crew, Process, Agent, Task, LLM
from crewai.tasks.task_output import TaskOutput

# --- Imports do Guardrails AI ---
from guardrails import Guard
from guardrails.errors import ValidationError as GuardrailsValidationError

# --- Import da Ferramenta Customizada ---
# Ajuste o path se necessário
try:
    from crew_nv5.custom_tool import WebsiteContentScraperTool
except ImportError:
    print("Erro: Não foi possível importar WebsiteContentScraperTool.")
    print("Verifique se custom_tool.py está no diretório correto (crew_nv5) e se não há erros de sintaxe nele.")
    sys.exit(1)


# Suprimir avisos desnecessários
warnings.filterwarnings("ignore", message="There is no current event loop", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Using Pydantic V1 type checking", category=RuntimeWarning)

# Carregar variáveis de ambiente
load_dotenv() # Ajuste o path se necessário


# --- Modelo Pydantic para a Saída Estruturada Esperada ---
class ExtractedWebData(BaseModel):
    """Modelo para os dados extraídos de uma página web."""
    title: str = Field(description="O título principal da página web.")
    summary: str = Field(description="Um breve resumo do conteúdo principal da página (máximo 2-3 frases).")


# --- Configuração do LLM ---
try:
    # Usaremos o mesmo LLM para ambos os agentes por simplicidade
    # llm = LLM(model='gemini/gemini-1.5-flash-latest')
    llm = LLM(model='gemini/gemini-2.0-flash-001')
    print(f"Usando LLM: {llm.model}")
except Exception as e:
    print(f"Erro ao inicializar o LLM: {e}")
    print("Verifique suas variáveis de ambiente.")
    sys.exit(1)

# --- Instanciar Ferramenta Customizada ---
try:
    scraper_tool = WebsiteContentScraperTool() # Não precisa de LLM aqui
except Exception as e:
    print(f"Erro ao inicializar WebsiteContentScraperTool: {e}")
    sys.exit(1)

# --- Função Callback com Guardrails AI (Pydantic) - CORRIGIDA ---
def validate_extracted_data(output: TaskOutput):
    """
    Callback para validar a saída da tarefa de estruturação usando
    Guardrails AI for_pydantic, passando a string JSON bruta para parse.
    """
    print("\n--- Executando Callback de Validação Guardrails (Pydantic) ---")
    raw_output_str = output.raw

    if not raw_output_str:
        print("--- Aviso: Saída bruta da tarefa está vazia. Pulando validação. ---")
        return

    # Preparar a string JSON para o Guardrails
    # Remover marcadores comuns e whitespace extra
    json_output_str = raw_output_str.strip().strip('```json').strip('```').strip()
    print(f"String JSON extraída para validação:\n'''\n{json_output_str}\n'''")

    # Verificar se a string parece minimamente com JSON (opcional, mas útil)
    if not (json_output_str.startswith('{') and json_output_str.endswith('}')):
         print(f"--- Falha Crítica: Saída bruta não parece ser um objeto JSON válido. ---")
         raise ValueError("A saída da tarefa de estruturação não é um objeto JSON.")


    # Configurar o Guardrails
    try:
        guard = Guard.for_pydantic(output_class=ExtractedWebData)
    except Exception as e:
        print(f"--- Erro ao inicializar Guard.for_pydantic: {e} ---")
        raise e

    try:
        # --- CORREÇÃO: Passar a string JSON para .parse() ---
        validation_outcome = guard.parse(json_output_str) # Passa a STRING JSON

        print(f"Resultado da validação Guardrails (Pydantic): {validation_outcome}")

        if validation_outcome.validation_passed and validation_outcome.validated_output is not None:
            print("--- Validação Guardrails (Pydantic) bem-sucedida! ---")
            validated_dict : dict = validation_outcome.validated_output
            print("Dados Validados (dict):")
            pprint.pprint(validated_dict) # Usar pprint
            # Verificação opcional de consistência
            try:
                ExtractedWebData(**validated_dict)
                print("[INFO] Dict validado corresponde ao modelo ExtractedWebData.")
            except PydanticValidationError as pyd_e:
                 print(f"[AVISO] Dict validado NÃO corresponde ao modelo: {pyd_e}")


        elif not validation_outcome.validation_passed:
             print("--- Validação Guardrails (Pydantic) Falhou ---")
             print(f"Erro reportado pelo Guardrails: {validation_outcome.error}")
             raise GuardrailsValidationError(f"Validação Pydantic falhou: {validation_outcome.error}")
        else:
             print("--- Validação Guardrails passou, mas sem dados validados (caso inesperado) ---")
             raise ValueError("Guardrails retornou sucesso mas sem dados validados.")

    except GuardrailsValidationError as gr_val_err:
        print(f"--- Falha na Validação Guardrails (Erro Capturado) ---")
        print(f"Erro: {gr_val_err}")
        raise gr_val_err
    except Exception as e:
        print(f"--- Falha no Callback de Validação (Erro Geral) ---")
        print(f"Erro: {type(e).__name__}: {e}")
        raise e

# --- Configuração da CrewAI ---

# Agente 1: Extrator de Conteúdo Web
web_extractor = Agent(
    role='Extrator de Conteúdo Web',
    goal='Raspar eficientemente o conteúdo de texto principal de uma página da web fornecida usando a ferramenta disponível.',
    backstory='Você é um profissional focado em tarefas de scraping. Sua única função é usar a ferramenta WebsiteContentScraperTool para obter o texto de uma URL.',
    tools=[scraper_tool], # Ferramenta customizada
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Agente 2: Estruturador de Dados
data_structurer = Agent(
    role='Estruturador de Dados de Conteúdo',
    goal='Analisar texto bruto de uma página web e extrair informações chave (título e resumo) em um formato JSON estruturado.',
    backstory='Você é um especialista em processamento de linguagem natural. Você recebe texto bruto e sua tarefa é identificar o título principal e criar um resumo conciso, formatando a saída como um objeto JSON.',
    llm=llm, # Usará este LLM para a tarefa de estruturação
    verbose=True,
    allow_delegation=False
)

# Tarefas
scrape_content_task = Task(
    description='Use a ferramenta WebsiteContentScraperTool para raspar o conteúdo principal da seguinte URL: {website_url}',
    expected_output='O conteúdo de texto bruto completo da página web.',
    agent=web_extractor
    # Sem callback aqui, apenas retorna o texto bruto
)

structure_data_task = Task(
    description=(
        'Analise o texto bruto fornecido no contexto (resultado da tarefa anterior). '
        'Identifique o título principal da página e crie um resumo conciso (2-3 frases) do conteúdo. '
        'Formate sua resposta final como um objeto JSON com as chaves "title" e "summary".'
    ),
    expected_output='Uma string contendo um único objeto JSON válido com as chaves "title" (string) e "summary" (string).',
    context=[scrape_content_task], # Recebe o texto bruto da tarefa anterior
    agent=data_structurer,
    callback=validate_extracted_data # <<< Guardrails valida a saída JSON estruturada
)

# Criando a Crew (Sequencial é mais simples para este caso)
crew = Crew(
    agents=[web_extractor, data_structurer],
    tasks=[scrape_content_task, structure_data_task],
    process=Process.sequential,
    verbose=True
)

# --- Execução ---
if __name__ == "__main__":
    # test_url = "https://www.guardrailsai.com/docs"
    test_url = "https://blog.crewai.com/enhancing-crewai-with-copilotkit-integration/"

    inputs = {'website_url': test_url}

    print(f"--- Iniciando a CrewAI (Nível 5 - Ferramenta Custom + Guardrails Callback Pydantic) ---")
    try:
        result = crew.kickoff(inputs=inputs)
        print("\n--- Execução da CrewAI Concluída (Validação Final Passou) ---")

        print("\n--- Saída Bruta da Tarefa de Scraping ---")
        print(scrape_content_task.output.raw if scrape_content_task.output else "N/A")

        print("\n--- Saída Final da Crew (Dados Estruturados e Validados) ---")
        if result and hasattr(result, 'raw'):
             # A saída final é a saída da última tarefa (structure_data_task)
             # Tentar parsear como JSON para visualização
             try:
                 final_data = json.loads(result.raw.strip().strip('```json').strip('```').strip())
                 pprint.pprint(final_data)
             except:
                 print(result.raw) # Imprimir como string se não for JSON
        else:
             print("A execução da crew não produziu uma saída final.")

    except Exception as e:
        print(f"\n--- Erro Capturado Durante a Execução da CrewAI ---")
        print(f"Ocorreu um erro, possivelmente devido à falha na validação.")
        # import traceback
        # traceback.print_exc()
        print(f"Erro: {e}")
        sys.exit(1)