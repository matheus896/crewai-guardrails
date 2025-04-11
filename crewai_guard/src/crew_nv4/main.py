"""
Nível 4: Processo Hierárquico CrewAI com Validação de Conteúdo Final via Guardrails.

Visão Geral:
Este script demonstra a configuração e execução de uma CrewAI utilizando o
Processo Hierárquico. Nesse modo, um agente gerente (gerado automaticamente
com base no `manager_llm` especificado) coordena o trabalho de múltiplos
agentes workers (Pesquisador, Escritor, Revisor) para completar uma série de tarefas.

O objetivo é pesquisar um tópico, escrever um relatório e, crucialmente, revisar
o relatório final, garantindo que ele contenha uma frase de rodapé específica.

Validação com Guardrails:

A validação da presença do rodapé é realizada utilizando o validador
`ContainsString` do Guardrails Hub. Esta validação é acionada através de uma
função de callback (`validate_report_contains_footer`) anexada à ÚLTIMA tarefa
da Crew (a tarefa de revisão).

"""

import os
from dotenv import load_dotenv
import sys
from typing import Optional, Union
import warnings
import json
import re

# --- Imports do CrewAI ---
from crewai import Crew, Process, Agent, Task, LLM
from crewai_tools import SerperDevTool
from crewai.tasks.task_output import TaskOutput

# --- Imports do Guardrails AI ---
from guardrails import Guard
from guardrails.hub import ContainsString # <<< Usar Contains
from guardrails.errors import ValidationError

# Suprimir avisos desnecessários
warnings.filterwarnings("ignore", message="There is no current event loop", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Using Pydantic V1 type checking", category=RuntimeWarning)

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# --- Configuração do LLM ---
try:
    worker_llm = LLM(model='gemini/gemini-2.0-flash-001')
    print(f"Usando LLM para Workers: {worker_llm.model}")
    manager_model = 'gemini/gemini-2.0-flash-001'
    print(f"Usando LLM para Manager: {manager_model}")
except Exception as e:
    print(f"Erro ao inicializar o LLM: {e}")
    sys.exit(1)

# --- Ferramentas ---
try:
    search_tool = SerperDevTool()
except Exception as e:
    print(f"Erro ao inicializar SerperDevTool: {e}")
    sys.exit(1)


# --- Função Callback com Guardrails AI (Contains) ---
def validate_report_contains_footer(output: TaskOutput):
    """
    Callback para validar se a saída do relatório final CONTÉM
    a string esperada usando Guardrails AI Contains.
    """
    print("\n--- Executando Callback de Validação Final Guardrails (Contains) ---")
    raw_output_str = output.raw

    if not raw_output_str:
        print("--- Aviso: Saída bruta da tarefa está vazia. Pulando validação. ---")
        return

    # A string que esperamos que esteja presente no relatório
    expected_footer = "Relatório gerado por CrewAI & Guardrails AI."

    try:
        # Usar ContainsString com on_fail='exception'
        guard = Guard().use(ContainsString(substring=expected_footer, on_fail="exception"))
    except Exception as e:
        print(f"--- Erro ao inicializar Guard ou Contains: {e} ---")
        print("Certifique-se de que o validador Contains está instalado (guardrails hub install hub://guardrails/contains_string)")
        raise e

    try:
        print(f"Texto bruto a ser validado pelo Guardrails (Verificando presença de '{expected_footer}'):\n'''\n{raw_output_str}\n'''")
        # Usar .validate() na string completa
        validation_outcome = guard.validate(raw_output_str)

        print(f"Resultado da validação Guardrails (Contains): {validation_outcome}")

        if validation_outcome.validation_passed:
            print("--- Validação Final Guardrails (Contains) bem-sucedida! ---")
            print(f"A saída contém a string esperada: '{expected_footer}'")

    except ValidationError as val_err:
        print(f"--- Falha na Validação Final Guardrails (Contains) ---")
        print(f"A saída NÃO contém a string esperada: '{expected_footer}'.")
        print(f"Erro Guardrails: {val_err}") # A mensagem de erro do Contains deve ser mais clara
        raise val_err
    except Exception as e:
        print(f"--- Falha no Callback de Validação (Erro Geral) ---")
        print(f"Erro: {type(e).__name__}: {e}")
        raise e

# --- Configuração da CrewAI ---
# Agentes (sem alterações)
researcher = Agent(
    role='Pesquisador de Mercado',
    goal='Encontrar dados relevantes e atuais sobre o tópico {topic}.',
    backstory='Você é um analista de pesquisa detalhista, focado em encontrar informações verificáveis e insights de mercado.',
    tools=[search_tool],
    llm=worker_llm,
    verbose=True,
    allow_delegation=False
)

writer = Agent(
    role='Escritor de Relatórios de Negócios',
    goal='Compilar as informações da pesquisa em um relatório claro e conciso sobre {topic}.',
    backstory='Você é um escritor experiente que transforma dados brutos em relatórios de negócios bem estruturados e fáceis de entender.',
    llm=worker_llm,
    verbose=True,
    allow_delegation=False
)

reviewer = Agent(
    role='Revisor de Qualidade e Editor Final',
    goal='Revisar o relatório final sobre {topic} para garantir clareza, coesão e conformidade com as diretrizes.',
    backstory='Você é um editor meticuloso com um olhar aguçado para detalhes. Você garante que todos os relatórios finais estejam polidos e atendam aos mais altos padrões antes da publicação, incluindo a adição da frase final.',
    llm=worker_llm,
    verbose=True,
    allow_delegation=False
)


# Tarefas (callback alterado na review_task)
research_task = Task(
    description='Pesquise informações abrangentes sobre: {topic}. Colete tendências de mercado, principais players e previsões futuras.',
    expected_output='Um resumo detalhado dos dados da pesquisa sobre {topic}, incluindo fontes.',
    agent=researcher
)

writing_task = Task(
    description='Escreva um relatório de negócios conciso baseado nas descobertas da pesquisa sobre {topic}. O relatório deve ter uma introdução, corpo principal com pontos chave e uma breve conclusão.',
    expected_output='Um relatório de negócios bem estruturado sobre {topic} em formato markdown.',
    context=[research_task],
    agent=writer
)

# Tarefa final de revisão com validação Guardrails (agora usando Contains)
review_task = Task(
    description=(
        'Revise o relatório escrito sobre {topic}. Verifique a clareza, precisão e fluxo geral. '
        'ADICIONE a seguinte frase no final do relatório: ' # Simplificando a instrução
        '"Relatório gerado por CrewAI & Guardrails AI."'
    ),
    expected_output=(
        'O relatório final revisado sobre {topic}, formatado em markdown, '
        'contendo a frase "Relatório gerado por CrewAI & Guardrails AI." em algum lugar.' # Ajustando a expectativa
    ),
    context=[writing_task],
    agent=reviewer,
    callback=validate_report_contains_footer
)

# Criando a Crew Hierárquica
crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.hierarchical,
    manager_llm=manager_model,
    verbose=True # Usar verbose 2 para ver a delegação e pensamentos
)

# --- Execução ---
if __name__ == "__main__":
    topic = "o impacto da IA generativa na criação de conteúdo de marketing"
    inputs = {'topic': topic}

    print(f"--- Iniciando a CrewAI Hierárquica (Nível 4 - com Contains) ---")
    try:
        result = crew.kickoff(inputs=inputs)
        print("\n--- Execução da CrewAI Hierárquica Concluída (Validação Final Passou) ---")
        print("\n--- Saída Final da Crew (Relatório Revisado e Validado) ---")
        if result and hasattr(result, 'raw'):
             print(result.raw)
        else:
             print("A execução da crew não produziu uma saída final.")

    except Exception as e:
        print(f"\n--- Erro Capturado Durante a Execução da CrewAI Hierárquica ---")
        print(f"Ocorreu um erro, possivelmente devido à falha na validação final (frase não encontrada).")
        # import traceback
        # traceback.print_exc()
        print(f"Erro: {e}")
        sys.exit(1)