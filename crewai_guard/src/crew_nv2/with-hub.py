"""
Nesse exemplo, utilizamos o Guardrails AI para validar a saída de uma tarefa de análise de sentimento.
A validação é feita através de um callback que verifica se a saída está no formato correto e se os valores estão dentro dos limites esperados. 
O Guardrails AI é configurado para validar a saída de acordo com o modelo Pydantic definido, garantindo que a análise de sentimento esteja correta e formatada adequadamente.
"""

import os
from dotenv import load_dotenv
import sys
from typing import Optional, Union
import warnings

# --- Imports Pydantic ---
from pydantic import BaseModel, Field, ValidationError

# --- Imports do CrewAI ---
from crewai import Crew, Process, Agent, Task, LLM
from crewai.tasks.task_output import TaskOutput

# --- Imports do Guardrails AI ---
from guardrails import Guard
from guardrails.hub import ValidChoices

# Suprimir o DeprecationWarning específico do asyncio no Guardrails
warnings.filterwarnings("ignore", message="There is no current event loop", category=DeprecationWarning)

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# --- Configuração do LLM ---
try:
    llm = LLM(model='gemini/gemini-2.0-flash-001')
    print(f"Usando LLM: {llm}")
except Exception as e:
    print(f"Erro ao inicializar o LLM: {e}")
    print("Verifique suas variáveis de ambiente (ex: GEMINI_API_KEY ou OPENAI_API_KEY).")
    sys.exit(1)
    
# --- Definição do Modelo Pydantic ---
class SentimentAnalysis(BaseModel):
    sentiment: str = Field(description="O sentimento geral do texto (positivo, negativo ou neutro)")
    summary: str = Field(description="Um breve resumo justificando a análise de sentimento.")
    confidence_score: Optional[float] = Field(None, description="Score de confiança da análise (0.0 a 1.0), se aplicável.")

# --- Função Callback com Guardrails AI ---
def validate_sentiment_analysis(output: TaskOutput):
    """
    Callback para validar a saída da tarefa de análise de sentimento
    usando Guardrails AI.
    """
    print("\n--- Executando Callback de Validação Guardrails ---")
    raw_output = output.raw

    if not raw_output:
        print("--- Aviso: Saída bruta da tarefa está vazia. Pulando validação. ---")
        # Poderia lançar um erro aqui se uma saída vazia for inaceitável
        # raise ValueError("Saída bruta da tarefa de análise está vazia.")
        return # Ou simplesmente retorna sem validar

    # Configurar o Guardrails a partir do modelo Pydantic
    # CORREÇÃO: Removido o argumento 'prompt' daqui
    guard = Guard.for_pydantic(
        output_class=SentimentAnalysis
    )

    # Adicionar validadores específicos
    guard.use(
        ValidChoices(choices=['positivo', 'negativo', 'neutro'], on_fail="exception"),
        on="sentiment"
    )
    # Poderíamos adicionar:
    # from guardrails.hub import InRange
    # guard.use(InRange(min=0.0, max=1.0, on_fail="exception"), on="confidence_score")

    try:
        print(f"Texto bruto a ser validado:\n'''\n{raw_output}\n'''")
        validation_outcome = guard.validate(raw_output)

        if validation_outcome.validation_passed and validation_outcome.validated_output is not None:
            print("--- Validação Guardrails bem-sucedida! ---")
            validated_data = validation_outcome.validated_output
            print("Dados Validados:", validated_data)
            # Opcional: Tentar anexar ao output para uso futuro (com ressalvas)
            # setattr(output, 'validated_pydantic_guardrails', validated_data)
        elif validation_outcome.validated_output is None and validation_outcome.reask is None:
             print("--- Validação Guardrails falhou ao extrair/validar dados (resultado None) ---")
             raise ValueError(f"Guardrails não conseguiu extrair/validar os dados da saída bruta. Erro: {validation_outcome.error}")
        else:
            # Se on_fail="exception" falhar, ele já lança a exceção antes daqui.
            # Este 'else' cobre casos onde on_fail não é 'exception' e a validação falha.
            print("--- Validação Guardrails Falhou ---")
            print(f"Sumário da Validação: {validation_outcome.validation_summaries}")
            raise ValueError(f"Validação falhou: {validation_outcome.error}")

    except ValidationError as pyd_e: # Erro de validação Pydantic
         print(f"--- Falha na Validação Pydantic (Guardrails) ---")
         print(f"Erro: {pyd_e}")
         raise pyd_e
    except Exception as e: # Outros erros (ex: falha do validador, erro do .validate)
        print(f"--- Falha na Validação Guardrails (Erro Geral) ---")
        print(f"Erro: {type(e).__name__}: {e}")
        raise e

# --- Configuração da CrewAI ---

# Agente Analista de Sentimento
sentiment_analyst = Agent(
    role='Analista de Sentimento',
    goal='Analisar o sentimento do texto fornecido e fornecer um resumo formatado em JSON.',
    backstory='Você é um especialista em processamento de linguagem natural com foco em análise de sentimento. Você identifica com precisão o tom emocional em textos e retorna os resultados em JSON.',
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Agente Relator
report_writer = Agent(
    role='Escritor de Relatórios',
    goal='Criar um pequeno relatório baseado na análise de sentimento fornecida.',
    backstory='Você transforma dados analíticos em relatórios claros e compreensíveis para stakeholders.',
    verbose=True,
    llm=llm,
    allow_delegation=False
)

# Tarefas
analysis_task = Task(
    description='Analise o sentimento do seguinte texto: "{text_input}". Determine se é positivo, negativo ou neutro e forneça um breve resumo da sua análise. Formate a resposta como um objeto JSON com as chaves "sentiment", "summary", e opcionalmente "confidence_score".',
    expected_output='Uma string contendo um único objeto JSON válido com a análise de sentimento (chaves: "sentiment", "summary", "confidence_score" opcional).',
    agent=sentiment_analyst,
    # Sem output_pydantic aqui
    callback=validate_sentiment_analysis # Callback para validação explícita
)

report_task = Task(
    description='Escreva um parágrafo de relatório baseado na análise de sentimento fornecida no contexto. Extraia "sentiment" e "summary" do JSON no contexto.',
    expected_output='Um parágrafo formatado descrevendo os resultados da análise de sentimento.',
    agent=report_writer,
    context=[analysis_task] # Ainda usa a saída bruta (string JSON) da tarefa anterior
)

# Criando a Crew
crew = Crew(
    agents=[sentiment_analyst, report_writer],
    tasks=[analysis_task, report_task],
    process=Process.sequential,
    verbose=True
)

# --- Execução ---
if __name__ == "__main__":
    text_to_analyze = "Adorei o novo celular! A câmera é incrível e a bateria dura muito. No entanto, achei o preço um pouco elevado."
    inputs = {'text_input': text_to_analyze}

    print(f"--- Iniciando a CrewAI (Nível 2 com Guardrails Explícito) ---")
    try:
        result = crew.kickoff(inputs=inputs)
        print("\n--- Execução da CrewAI Concluída ---")
        print("\n--- Saída Final da Crew (Relatório) ---")
        # Verifica se result não é None antes de acessar .raw
        if result and hasattr(result, 'raw'):
             print(result.raw)
        else:
             print("A execução da crew não produziu uma saída final.")

    except Exception as e:
        print(f"\n--- Erro durante a execução da CrewAI (Provavelmente da validação) ---")
        import traceback
        traceback.print_exc()
        sys.exit(1)
