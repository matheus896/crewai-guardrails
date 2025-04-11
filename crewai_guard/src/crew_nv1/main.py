"""
Exemplo criando com a biblioteca sem utilizar o hub
"""

import os
import re # Importar regex para contar frases de forma um pouco melhor
from dotenv import load_dotenv
import sys
from typing import Union

# --- Imports do CrewAI ---
from crewai import Crew, Process, Agent, Task, LLM
from crewai_tools import SerperDevTool # Ferramenta de busca simples

# --- Imports do Guardrails AI ---
from guardrails import Guard
from guardrails.validators import Validator, register_validator # Mudança para nova importação
from guardrails.classes.validation.validation_result import PassResult, FailResult # Importar classes de resultado

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

llm = LLM(model='gemini/gemini-2.0-flash-001')

# --- Definição do Validador Guardrails (Nível 1) ---
@register_validator(name="has-exactly-n-sentences", data_type="string")
class HasExactlyNSentences(Validator):
    """
    Valida se o texto fornecido contém exatamente N frases.
    Conta frases baseando-se nos delimitadores ., ! e ?.
    """
    def __init__(self, n: int, on_fail: str | None = None):
        super().__init__(on_fail=on_fail, n=n)
        self._n = n # Armazena o número de frases esperado

    def validate(self, value: str, metadata: dict) -> Union[PassResult, FailResult]:
        """Executa a lógica de validação."""
        if not isinstance(value, str):
            return FailResult(error_message="Input should be a string.")

        # Contagem de frases um pouco mais robusta que value.count('.')
        # Divide por delimitadores e remove strings vazias resultantes
        delimiters = r"[.!?]"
        sentences = [s for s in re.split(delimiters, value) if s.strip()]
        count = len(sentences)

        if count == self._n:
            # Retorna PassResult se a validação passar
            return PassResult()
        else:
            # Retorna FailResult com uma mensagem de erro se falhar
            return FailResult(
                error_message=f"Esperava {self._n} frases, mas encontrou {count}."
            )

# --- Configuração da CrewAI (Nível 1) ---

# Ferramenta de pesquisa (DuckDuckGo é gratuito e não precisa de API key)
search_tool = SerperDevTool()

# Agente Pesquisador
researcher = Agent(
    role='Pesquisador de IA',
    goal='Encontrar informações concisas sobre o tópico {topic}',
    backstory='Você é um assistente de pesquisa IA eficiente, bom em encontrar fatos chave.',
    tools=[search_tool],
    llm=llm,
    verbose=True, # Mostrar o que o agente está fazendo
    allow_delegation=False
)

# Agente Escritor
writer = Agent(
    role='Escritor de Resumos IA',
    goal='Escrever um resumo conciso de {n_sentences} frases sobre as descobertas da pesquisa sobre {topic}',
    backstory='Você é um assistente de escrita IA, especializado em criar resumos curtos e informativos.',
    verbose=True,
    llm=llm,
    allow_delegation=False
)

# Tarefas
research_task = Task(
    description='Pesquise sobre o tópico: {topic}. Colete 3-5 fatos ou pontos chave.',
    expected_output='Uma lista de pontos chave sobre {topic}.',
    agent=researcher
)

write_task = Task(
    description='Escreva um resumo conciso de exatamente {n_sentences} frases baseado nas descobertas da pesquisa sobre {topic}.',
    expected_output='Um resumo bem escrito sobre {topic} contendo exatamente {n_sentences} frases.',
    agent=writer,
    context=[research_task] # Usa a saída da tarefa anterior como contexto
)

# Criando a Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True
)

# --- Execução e Validação ---
if __name__ == "__main__":
    # Define o tópico e o número de frases desejado
    topic = "o futuro da computação quântica no Brasil"
    num_sentences = 3

    # Define os inputs para a crew
    inputs = {'topic': topic, 'n_sentences': num_sentences}

    print(f"--- Iniciando a CrewAI para pesquisar e resumir '{topic}' em {num_sentences} frases ---")
    # Executa a Crew
    # Tratamento de erro básico para a execução da crew
    try:
        result = crew.kickoff(inputs=inputs)
        crew_raw_output = result.raw
        print("\n--- Saída Bruta da CrewAI ---")
        print(crew_raw_output)
    except Exception as e:
        print(f"\n--- Erro durante a execução da CrewAI ---")
        print(f"Erro: {e}")
        sys.exit(1) # Termina se a crew falhar

    print(f"\n--- Validando a saída com Guardrails AI (Esperando {num_sentences} frases) ---")
    # Configura o Guardrails para validar a saída
    # Usando 'exception' como on_fail para simplicidade no Nível 1
    guard = Guard().use(HasExactlyNSentences(n=num_sentences, on_fail="exception"))

    # Valida a saída bruta da Crew
    try:
        validated_output = guard.validate(crew_raw_output)
        print("\n--- Validação do Guardrails AI bem-sucedida ---")
        print("A saída está no formato esperado (exatamente 3 frases).")
        print("\n--- Saída Final Validada ---")
        print(validated_output) # Se passar, validated_output é igual a crew_raw_output
    except Exception as e:
        print(f"\n--- Falha na Validação do Guardrails AI ---")
        print(f"A saída não atende ao critério de {num_sentences} frases.")
        print(f"Erro detalhado: {e}")
