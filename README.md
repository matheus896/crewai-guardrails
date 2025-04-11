# CrewAI & Guardrails AI Integration Examples 🤖🛡️

Este repositório contém uma série de exemplos progressivos demonstrando como integrar a framework de agentes autônomos **CrewAI** com a biblioteca de validação e garantia **Guardrails AI**. O objetivo é explorar diferentes padrões de como adicionar camadas de validação, estruturação e segurança aos fluxos de trabalho de agentes CrewAI, começando com validações simples e avançando para integrações mais complexas, incluindo ferramentas customizadas.

## ✨ Key Concepts Demonstrated

*   **CrewAI:**
    *   Criação de Agentes (Roles, Goals, Backstories)
    *   Definição de Tarefas (Description, Expected Output, Context)
    *   Orquestração de Crews (Processo Sequencial e Hierárquico)
    *   Uso de Ferramentas (Built-in com `SerperDevTool` e Customizadas)
    *   Callbacks de Tarefas
    *   Saída Estruturada com Pydantic (`output_pydantic`)
    *   Configuração de LLM
*   **Guardrails AI:**
    *   Validação de Saída Pós-Execução (`guard.validate()`)
    *   Validação em Callbacks de Tarefas CrewAI
    *   Uso de Validadores Customizados (`@register_validator`)
    *   Uso de Validadores do Guardrails Hub (e.g., `RegexMatch`, `DetectJailbreak`, `ContainsString`, `ValidChoices`)
    *   Configuração de `Guard` (`Guard()`, `Guard.for_pydantic()`, `Guard.use()`)
    *   Ações em Caso de Falha (`on_fail="exception"`, `on_fail="reask"`)
    *   Validação de Strings e Saídas Estruturadas (JSON/Pydantic)
    *   Integração dentro de Ferramentas Customizadas CrewAI

## 📂 Project Structure

O código é organizado em níveis dentro do diretório `crewai_guard/src/`:

*   **`crew_nv1/`**: Nível 1 - Introdução à validação pós-execução da CrewAI com Guardrails (Validador customizado e do Hub com `on_fail='exception'`).
*   **`crew_nv2/`**: Nível 2 - Comparação entre a validação implícita do CrewAI (`output_pydantic`) e a validação explícita com Guardrails (`Guard.for_pydantic`) em um callback.
*   **`crew_nv3/`**: Nível 3 - Explorando `on_fail='reask'` do Guardrails em um callback para detectar falhas e sugerir correções (usando `ValidChoices` e `DetectJailbreak`).
*   **`crew_nv4/`**: Nível 4 - Utilizando o processo hierárquico do CrewAI (`Process.hierarchical` com `manager_llm`) e aplicando validação Guardrails (`ContainsString`) na tarefa final, simulando um controle de qualidade.
*   **`crew_nv5/`**: Nível 5 - Criação de uma ferramenta customizada CrewAI (`WebsiteContentScraperTool`) e integração da validação/estruturação Guardrails (`Guard.for_pydantic`) *dentro* da lógica da ferramenta e validando a saída final com um callback.

## 🚀 Setup & Installation

1.  **Pré-requisitos:**
    *   Python >= 3.12 (conforme `pyproject.toml`)
    *   `git`
    *   Recomendado: `uv` (ou `pip` e `venv`)
    *   Docker (apenas para o Nível 5, `CodeInterpreterTool`, se usado no futuro, ou para validadores Guardrails que necessitem)

2.  **Clonar o Repositório:**
    ```bash
    git clone https://github.com/matheus896/crewai-guardrails.git
    cd crewai-guardrails/crewai_guard
    ```

3.  **Criar e Ativar Ambiente Virtual (Recomendado):**
    ```bash
    # Usando venv
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # OU
    .\.venv\Scripts\activate  # Windows

    # Ou usando uv (ele gerencia o venv)
    # uv venv
    # source .venv/bin/activate # (ou script equivalente)
    ```

4.  **Instalar Dependências:**
    ```bash
    # Usando uv
    uv pip install -e .

    # Ou usando pip
    pip install -e .
    ```
    Isso instalará `crewai`, `crewai-tools`, `guardrails-ai`, `python-dotenv` e outras dependências listadas em `pyproject.toml`.

5.  **Instalar Validadores do Guardrails Hub (Necessário para Níveis 1, 3, 4):**
    Execute os seguintes comandos no terminal (com seu ambiente virtual ativo):
    ```bash
    guardrails hub install hub://guardrails/regex_match
    guardrails hub install hub://guardrails/detect_jailbreak
    guardrails hub install hub://guardrails/ends_with  # Embora substituído, pode ser útil
    guardrails hub install hub://guardrails/contains_string # Usado no Nv4 final
    guardrails hub install hub://guardrails/valid_choices
    # Adicione outros validadores se usados no futuro
    ```

## ⚙️ Configuration

1.  **Renomeie `.env.example` para `.env`:**
    ```bash
    # Na raiz do projeto (matheus896-crewai-guardrails/)
    cp .env.example .env
    ```
    *Alternativamente, crie um arquivo `.env` dentro de `crewai_guard/`.*

2.  **Edite o arquivo `.env` e adicione suas chaves de API:**
    ```dotenv
    # Escolha UM provedor de LLM e adicione a chave correspondente
    # Para Gemini:
    GEMINI_API_KEY=SUA_CHAVE_API_GEMINI_AQUI

    # OU Para OpenAI:
    # OPENAI_API_KEY=SUA_CHAVE_API_OPENAI_AQUI
    # OPENAI_MODEL_NAME=gpt-4o-mini # Opcional, usado se LLM() for chamado sem 'model'

    # Chave da API Serper (Necessária para exemplos que usam SerperDevTool, como Nv1, Nv4)
    SERPER_API_KEY=SUA_CHAVE_API_SERPER_AQUI

    # Chave da API Guardrails Hub (Opcional - necessária para submeter validadores ou usar recursos avançados do Hub)
    # GUARDRAILS_API_KEY=SUA_CHAVE_GUARDRAILS_HUB_AQUI
    ```

*   **Importante:** Os scripts (`main.py`, `with-hub.py`) estão atualmente configurados para usar Gemini (`gemini/gemini-2.0-flash-001` ou similar). Se você preferir usar OpenAI, comente/descomente as linhas relevantes de `LLM()` nos scripts e certifique-se que `OPENAI_API_KEY` está no `.env`.
*   A `SERPER_API_KEY` é necessária para os níveis que utilizam a ferramenta de busca.
*   A `GUARDRAILS_API_KEY` pode ser necessária no futuro ou para baixar/submeter validadores privados.

## ▶️ Running the Examples

1.  **Ative seu Ambiente Virtual:** (Se não estiver ativo)
    ```bash
    source .venv/bin/activate # ou equivalente
    ```
2.  **Navegue até o diretório do nível desejado:**
    ```bash
    cd crewai_guard/src/crew_nv<NIVEL>
    # Exemplo: cd crewai_guard/src/crew_nv5
    ```
3.  **Execute o script principal:**
    ```bash
    python main.py
    ```
    *   Para os níveis que têm `with-hub.py`, execute:
        ```bash
        python with-hub.py
        ```
    *   Para testar a ferramenta do Nível 5 isoladamente:
        ```bash
        # Estando dentro de crewai_guard/src/crew_nv5/
        python teste_tool.py
        ```

Observe o output no terminal para ver os logs do CrewAI (pensamentos e ações dos agentes), os logs de validação do Guardrails (executados nos callbacks) e o resultado final.

## 📜 License

Este projeto é licenciado sob a MIT License.
