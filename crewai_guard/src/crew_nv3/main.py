"""
Nível 3: Guardrails 'reask' em Callback Intermediário.

Este exemplo demonstra o uso de on_fail='reask' do Guardrails dentro
de um callback do CrewAI. O objetivo é que o Guardrails identifique
uma falha na validação e sugira uma nova tentativa com feedback
corretivo para o LLM.
"""

import os
from dotenv import load_dotenv
import sys
from typing import Optional, Union
import warnings
import json

# --- Imports Pydantic ---
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

# --- Imports do CrewAI ---
from crewai import Crew, Process, Agent, Task, LLM
from crewai.tasks.task_output import TaskOutput

# --- Imports do Guardrails AI ---
from guardrails import Guard
from guardrails.hub import ValidChoices
from guardrails.classes.validation.validation_result import FailResult

# Suprimir o DeprecationWarning específico do asyncio no Guardrails
warnings.filterwarnings("ignore", message="There is no current event loop", category=DeprecationWarning)

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# --- Configuração do LLM ---
try:
    # Use um modelo mais capaz para garantir a geração de JSON
    llm = LLM(model='gemini/gemini-2.0-flash-001') # Ou 'openai/gpt-4o-mini' se preferir OpenAI
    print(f"Usando LLM: {llm}")
except Exception as e:
    print(f"Erro ao inicializar o LLM: {e}")
    print("Verifique suas variáveis de ambiente (ex: GEMINI_API_KEY ou OPENAI_API_KEY).")
    sys.exit(1)

# --- Definição do Modelo Pydantic com Validador Guardrails ---
class SentimentAnalysis(BaseModel):
    sentiment: str = Field(
        description="O sentimento geral do texto (positivo, negativo ou neutro)",
        validators=[
            # Remover 'negativo' temporariamente para forçar o reask
            ValidChoices(choices=['positivo', 'negativo', 'neutro'], on_fail="reask")
            #ValidChoices(choices=['positivo', 'neutro'], on_fail="reask")
        ]
    )
    summary: str = Field(description="Um breve resumo justificando a análise de sentimento.")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Score de confiança da análise (0.0 a 1.0), se aplicável.")

# --- Função Callback com Guardrails AI ---
def validate_sentiment_analysis_with_reask(output: TaskOutput):
    """
    Callback para validar a saída da tarefa de análise de sentimento
    usando Guardrails AI com on_fail='reask' definido no Pydantic model.
    Se a validação falhar (ou sugerir reask), lança um erro.
    """
    print("\n--- Executando Callback de Validação Guardrails (com Reask no Pydantic) ---")
    raw_output_str = output.raw

    if not raw_output_str:
        print("--- Aviso: Saída bruta da tarefa está vazia. Pulando validação. ---")
        return # Importante retornar aqui para não dar erro no guard.parse(None)

    # Log inicial da saída bruta
    print(f"Saída Bruta Recebida:\n'''\n{raw_output_str}\n'''")

    # Configurar o Guardrails
    guard = Guard.for_pydantic(output_class=SentimentAnalysis)

    try:
        # Tentar extrair e validar a saída bruta
        validation_outcome = guard.parse(raw_output_str)

        print(f"Resultado da validação Guardrails: {validation_outcome}")

        if validation_outcome.validation_passed and validation_outcome.validated_output is not None:
            print("--- Validação Guardrails bem-sucedida! ---")
            validated_data : dict = validation_outcome.validated_output # É um dict
            print("Dados Validados (dict):", validated_data)
            # Não tentamos mais usar setattr aqui

        elif not validation_outcome.validation_passed and validation_outcome.reask is not None:
            print("--- Validação Guardrails Falhou (Reask sugerido) ---")
            print(f"Objeto (potencialmente inválido/parcial) retornado: {validation_outcome.validated_output}")
            print(f"Motivo da falha (último erro): {validation_outcome.error}")
            print(f"Sugestão de Reask do Guardrails: {validation_outcome.reask}")
            raise ValueError(f"Validação Guardrails falhou, 'reask' sugerido. Erro: {validation_outcome.error}.")

        elif not validation_outcome.validation_passed:
             print("--- Validação Guardrails falhou (Erro na Extração ou Validação Pydantic sem Reask) ---")
             print(f"Erro reportado pelo Guardrails: {validation_outcome.error}")
             raise ValueError(f"Guardrails falhou na validação/extração. Erro: {validation_outcome.error}")
        else:
             print("--- Validação Guardrails passou, mas sem dados validados (caso inesperado) ---")
             raise ValueError("Guardrails retornou sucesso mas sem dados validados.")

    # Captura erros específicos ou gerais que podem ocorrer
    except (PydanticValidationError, json.JSONDecodeError) as val_err:
         print(f"--- Falha no Callback: Erro de Validação/Parsing ---")
         print(f"Erro: {type(val_err).__name__}: {val_err}")
         raise val_err # Propaga o erro para parar a crew
    except Exception as e:
        print(f"--- Falha no Callback de Validação (Erro Geral) ---")
        print(f"Erro: {type(e).__name__}: {e}")
        raise e # Propaga a exceção

# --- Configuração da CrewAI ---

sentiment_analyst = Agent(
    role='Analista de Sentimento',
    goal='Analisar o sentimento do texto fornecido e fornecer um resumo formatado em JSON.',
    backstory='Você é um especialista em processamento de linguagem natural com foco em análise de sentimento. Você identifica com precisão o tom emocional em textos e retorna os resultados em JSON estritamente conforme o schema: {"sentiment": "string (positivo, negativo, ou neutro)", "summary": "string", "confidence_score": float (opcional)}.',
    llm=llm,
    verbose=True,
    allow_delegation=False
)

report_writer = Agent(
    role='Escritor de Relatórios',
    goal='Criar um pequeno relatório baseado na análise de sentimento fornecida.',
    backstory='Você transforma dados analíticos em relatórios claros e compreensíveis para stakeholders. Você espera receber o contexto como uma string JSON.',
    verbose=True,
    llm=llm,
    allow_delegation=False
)

analysis_task = Task(
    description='Analise o sentimento do seguinte texto: "{text_input}". Determine se é positivo, negativo ou neutro e forneça um breve resumo da sua análise. Formate a resposta como um único objeto JSON válido com as chaves "sentiment", "summary", e opcionalmente "confidence_score". Exemplo: {{"sentiment": "positivo", "summary": "O texto expressa grande satisfação."}}',
    expected_output='Uma string contendo um único objeto JSON válido com a análise de sentimento (chaves: "sentiment" como "positivo", "negativo" ou "neutro", "summary" como string, "confidence_score" opcional como float).',
    agent=sentiment_analyst,
    callback=validate_sentiment_analysis_with_reask
)

report_task = Task(
    description='Escreva um parágrafo de relatório baseado na análise de sentimento fornecida no contexto. O contexto será uma string JSON. Parseie o JSON para extrair "sentiment" e "summary".',
    expected_output='Um parágrafo formatado descrevendo os resultados da análise de sentimento.',
    agent=report_writer,
    context=[analysis_task] # Recebe output.raw da tarefa anterior
)

crew = Crew(
    agents=[sentiment_analyst, report_writer],
    tasks=[analysis_task, report_task],
    process=Process.sequential,
    verbose=True
)

# --- Execução ---
if __name__ == "__main__":
    # Teste 1: Deve passar na validação
    text_to_analyze_pass = "Adorei o novo celular! A câmera é incrível e a bateria dura muito."
    # Teste 2: Deve falhar na validação (espera-se 'reask')
    text_to_analyze_reask = "O atendimento foi rápido, o que é bom, mas o produto veio com defeito, o que é péssimo."
    # Teste 3: Pode falhar na geração de JSON pelo LLM
    text_to_analyze_fail_json = "Sim."

    # Escolha qual texto testar
    text_to_analyze = text_to_analyze_reask # Ou _reask ou _fail_json

    inputs = {'text_input': text_to_analyze}

    print(f"--- Iniciando a CrewAI (Nível 3 com Guardrails 'reask' no Pydantic - Corrigido) ---")
    try:
        result = crew.kickoff(inputs=inputs)
        print("\n--- Execução da CrewAI Concluída (Validação passou) ---")

        print("\n--- Saída Bruta da Tarefa de Análise (JSON String) ---")
        print(analysis_task.output.raw if analysis_task.output else "N/A")

        print("\n--- Saída Final da Crew (Relatório) ---")
        if result and hasattr(result, 'raw'):
             print(result.raw)
        else:
             print("A execução da crew não produziu uma saída final.")

    except Exception as e:
        print(f"\n--- Erro Capturado Durante a Execução da CrewAI ---")
        print(f"Ocorreu um erro, possivelmente devido à falha na validação ou outro problema.")
        # import traceback # Descomente para debug
        # traceback.print_exc()
        print(f"Erro: {e}")
        sys.exit(1)