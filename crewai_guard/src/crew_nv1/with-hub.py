"""
Utilizando o Guardrails AI com RegexMatch do Hub para validar a saída da CrewAI

"""

import os
from dotenv import load_dotenv
import sys

# --- Imports do CrewAI ---
from crewai import Crew, Process, Agent, Task, LLM
from crewai_tools import SerperDevTool 

# --- Imports do Guardrails AI ---
from guardrails import Guard
from guardrails.hub import RegexMatch # Importar o validador do Hub

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# --- Configuração do LLM (Exemplo com Gemini, ajuste conforme seu .env) ---
# Certifique-se de ter GEMINI_API_KEY no seu .env se for usar Gemini
# Ou ajuste para 'openai' e tenha OPENAI_API_KEY
try:
    # Tenta usar um LLM do ambiente, ou fallback para um modelo OpenAI se não definido
    # Se você tiver OPENAI_API_KEY, ele pode usar 'gpt-4o-mini' por padrão
    # Se tiver GEMINI_API_KEY, pode usar 'gemini-2.0-flash'
    # Se ambos, o comportamento exato pode depender de outras env vars ou padrões internos
    # Para ter certeza, defina OPENAI_MODEL_NAME ou MODEL no .env
    # Exemplo explícito com Gemini Flash:
    llm = LLM(model='gemini/gemini-2.0-flash-001')
    print(f"Usando LLM: {llm}")
except Exception as e:
    print(f"Erro ao inicializar o LLM: {e}")
    print("Verifique suas variáveis de ambiente (ex: GEMINI_API_KEY ou OPENAI_API_KEY).")
    sys.exit(1)


# --- Configuração da CrewAI (Nível 1) ---

# Ferramenta de pesquisa (Serper precisa de SERPER_API_KEY no .env)
try:
    search_tool = SerperDevTool()
except Exception as e:
    print(f"Erro ao inicializar SerperDevTool: {e}")
    print("Verifique se SERPER_API_KEY está configurado no seu arquivo .env.")
    sys.exit(1)


# Agente Pesquisador
researcher = Agent(
    role='Pesquisador de IA Quântica',
    goal='Encontrar os desenvolvimentos mais recentes sobre {topic}',
    backstory='Você é um especialista em computação quântica, focado em avanços no Brasil.',
    tools=[search_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Agente Escritor
writer = Agent(
    role='Escritor Técnico Conciso',
    goal='Escrever uma introdução muito breve (1-2 frases) sobre as descobertas da pesquisa sobre {topic}',
    backstory='Você é um escritor técnico que vai direto ao ponto, criando introduções curtas e impactantes.',
    verbose=True,
    llm=llm,
    allow_delegation=False
)

# Tarefas
research_task = Task(
    description='Pesquise sobre o tópico: {topic}. Encontre 1 ou 2 pontos principais recentes.',
    expected_output='Uma lista curta com os pontos chave sobre {topic}.',
    agent=researcher
)

# Ajustando a tarefa de escrita para ser mais curta, facilitando a validação com RegexMatch
write_task = Task(
    description='Baseado na pesquisa, escreva uma única frase introdutória sobre os últimos desenvolvimentos de {topic}.',
    expected_output='Uma única frase concisa resumindo o estado atual de {topic} no Brasil.',
    agent=writer,
    context=[research_task]
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
    # Define o tópico
    topic = "o futuro da computação quântica no Brasil"

    # Define os inputs para a crew
    # Não precisamos mais de 'n_sentences' aqui
    inputs = {'topic': topic}

    print(f"--- Iniciando a CrewAI para pesquisar e introduzir '{topic}' ---")
    # Executa a Crew
    try:
        result = crew.kickoff(inputs=inputs)
        crew_raw_output = result.raw
        print("\n--- Saída Bruta da CrewAI ---")
        print(crew_raw_output)
    except Exception as e:
        print(f"\n--- Erro durante a execução da CrewAI ---")
        print(f"Erro: {e}")
        sys.exit(1)

    # --- Validação com Guardrails AI usando RegexMatch ---
    # MUDANÇA: Verificar se a palavra "quântica" (case-insensitive) existe na string
    validation_regex = r"(?i)quântica" # (?i) para case-insensitive, sem ^ ou .*
    print(f"\n--- Validando a saída com Guardrails AI (Esperando conter 'quântica') ---")

    # Configura o Guardrails com RegexMatch
    guard = Guard().use(
        RegexMatch(regex=validation_regex, match_type="search", on_fail="exception")
        # match_type='search' é o padrão e verifica se o regex aparece em qualquer lugar
    )

    # Valida a saída bruta da Crew
    try:
        # Adiciona verificação se crew_raw_output existe
        if crew_raw_output:
            validated_output = guard.validate(crew_raw_output)
            print("\n--- Validação do Guardrails AI (RegexMatch) bem-sucedida ---")
            print(f"A saída corresponde ao padrão regex: '{validation_regex}'.")
            print("\n--- Saída Final Validada ---")
            print(validated_output) # Se passar, validated_output é igual a crew_raw_output
        else:
            print("\n--- Validação do Guardrails AI não executada (saída da crew vazia) ---")

    except Exception as e:
        print(f"\n--- Falha na Validação do Guardrails AI (RegexMatch) ---")
        print(f"A saída NÃO corresponde ao padrão regex: '{validation_regex}'.")
        print(f"Erro detalhado: {e}")