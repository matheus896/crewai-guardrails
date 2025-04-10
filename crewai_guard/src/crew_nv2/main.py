"""
Nesse exemplo o guardrail não é usado explicitamente, mas o modelo Pydantic é utilizado 
para estruturar a saída da tarefa de análise de sentimento.
O modelo Pydantic é uma maneira de definir e validar a estrutura dos dados que você espera receber.
"""

import os
from dotenv import load_dotenv
import sys
from typing import Optional # Necessário para campos opcionais no Pydantic

# --- Imports Pydantic ---
from pydantic import BaseModel, Field

# --- Imports do CrewAI ---
from crewai import Crew, Process, Agent, Task, LLM


# --- Imports do Guardrails AI (Não usados explicitamente neste nível) ---
# from guardrails import Guard # Não precisamos no Nível 2

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

# --- Definição do Modelo Pydantic para Saída Estruturada (Nível 2) ---
class SentimentAnalysis(BaseModel):
    sentiment: str = Field(description="O sentimento geral do texto (positivo, negativo ou neutro)")
    summary: str = Field(description="Um breve resumo justificando a análise de sentimento.")
    confidence_score: Optional[float] = Field(None, description="Score de confiança da análise (0.0 a 1.0), se aplicável.")

# --- Configuração da CrewAI (Nível 2) ---

# Agente Analista de Sentimento
sentiment_analyst = Agent(
    role='Analista de Sentimento',
    goal='Analisar o sentimento do texto fornecido e fornecer um resumo.',
    backstory='Você é um especialista em processamento de linguagem natural com foco em análise de sentimento. Você identifica com precisão o tom emocional em textos.',
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
    description='Analise o sentimento do seguinte texto: "{text_input}". Determine se é positivo, negativo ou neutro e forneça um breve resumo da sua análise.',
    expected_output='Um objeto Pydantic do tipo SentimentAnalysis contendo o sentimento, um resumo e opcionalmente um score de confiança.',
    agent=sentiment_analyst,
    output_pydantic=SentimentAnalysis # <<< Ponto chave do Nível 2
)

report_task = Task(
    description='Escreva um parágrafo de relatório baseado na análise de sentimento. Inclua o sentimento identificado e o resumo da análise.',
    expected_output='Um parágrafo formatado descrevendo os resultados da análise de sentimento.',
    agent=report_writer,
    context=[analysis_task] # Usa a saída ESTRUTURADA da tarefa anterior
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
    # Define o texto de entrada
    text_to_analyze = "Adorei o novo celular! A câmera é incrível e a bateria dura muito. No entanto, achei o preço um pouco elevado."

    # Define os inputs para a crew
    inputs = {'text_input': text_to_analyze}

    print(f"--- Iniciando a CrewAI para analisar o sentimento do texto ---")
    # Executa a Crew
    try:
        result = crew.kickoff(inputs=inputs)

        print("\n--- Execução da CrewAI Concluída ---")

        # Acessando a saída estruturada da primeira tarefa
        analysis_output : SentimentAnalysis = analysis_task.output.pydantic
        if analysis_output:
            print("\n--- Saída Estruturada da Tarefa de Análise (Pydantic) ---")
            print(f"Sentimento: {analysis_output.sentiment}")
            print(f"Resumo: {analysis_output.summary}")
            if analysis_output.confidence_score:
                 print(f"Confiança: {analysis_output.confidence_score}")
        else:
            print("\n--- Saída Estruturada da Tarefa de Análise: Falhou ou Vazia ---")
            print(f"Saída bruta da tarefa 1: {analysis_task.output.raw if analysis_task.output else 'N/A'}")


        print("\n--- Saída Final da Crew (Relatório) ---")
        print(result.raw) # Saída da última tarefa (o relatório)

    except Exception as e:
        print(f"\n--- Erro durante a execução da CrewAI ---")
        # Imprimir o traceback pode ajudar a depurar
        import traceback
        traceback.print_exc()
        # print(f"Erro: {e}")
        sys.exit(1)