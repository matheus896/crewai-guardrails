"""
Nível 5: Ferramenta Customizada CrewAI - Apenas Scraping.

Define a WebsiteContentScraperTool que APENAS raspa o conteúdo de uma URL
e retorna o texto bruto. A estruturação/validação será feita por uma tarefa
do agente que usa esta ferramenta.
"""

import warnings
import requests
from bs4 import BeautifulSoup
from typing import Type, Optional, Any, Dict, Union
import re

# --- Imports Pydantic ---
from pydantic import BaseModel, Field

# --- Imports do CrewAI ---
from crewai.tools import BaseTool

# Suprimir avisos
warnings.filterwarnings("ignore", message="There is no current event loop", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Using Pydantic V1 type checking", category=RuntimeWarning)


# --- Modelo Pydantic para o Input da Ferramenta ---
class WebsiteInputSchema(BaseModel):
    """Input schema for WebsiteContentScraperTool."""
    website_url: str = Field(..., description="A URL completa do website para raspar o conteúdo.")


# --- Definição da Ferramenta Customizada de Scraping ---
class WebsiteContentScraperTool(BaseTool):
    name: str = "Website Content Scraper"
    description: str = (
        "Raspa o conteúdo principal de texto de uma URL fornecida. "
        "Útil para obter o texto bruto de uma página web para análise posterior."
    )
    args_schema: Type[BaseModel] = WebsiteInputSchema

    def _run(self, website_url: str) -> str:
        """
        Executa a lógica da ferramenta: raspa a web e retorna o texto bruto.
        Retorna o texto raspado ou uma string de erro.
        """
        print(f"\n--- [Tool._run] Executando scraping para: {website_url} ---")

        # 1. Scraping
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(website_url, headers=headers, timeout=15)
            response.raise_for_status() # Lança erro para status HTTP ruins

            soup = BeautifulSoup(response.text, 'html.parser')

            # Tentativa de obter conteúdo principal
            main_content = soup.find('main') or soup.find('article') or soup.find('body')

            if main_content:
                 # Remover tags indesejadas
                 for tag in main_content(['script', 'style', 'nav', 'footer', 'aside', 'header', 'form', 'button', 'iframe', 'img']):
                     tag.decompose()

                 # Obter texto, remover linhas vazias excessivas e limitar tamanho
                 raw_content = main_content.get_text(separator='\n', strip=True)
                 # Limpar múltiplas linhas vazias
                 raw_content = re.sub(r'\n\s*\n', '\n\n', raw_content)

                 max_chars = 8000 # Aumentar um pouco o limite se necessário
                 if len(raw_content) > max_chars:
                     print(f"[Tool Warning] Conteúdo raspado truncado para {max_chars} caracteres.")
                     raw_content = raw_content[:max_chars]

                 print(f"[Tool Info] Scraping concluído. {len(raw_content)} caracteres extraídos (início):\n'''\n{raw_content[:300]}...\n'''")

                 if len(raw_content.strip()) < 50: # Verificar conteúdo mínimo
                      return f"[Tool Error] Conteúdo significativo não encontrado ou muito curto em {website_url}"

                 return raw_content # Retorna o texto bruto

            else:
                 return f"[Tool Error] Tag 'body' não encontrada em {website_url}"


        except requests.exceptions.RequestException as e:
            return f"[Tool Error] Erro de Rede ao acessar {website_url}: {e}"
        except Exception as e:
            return f"[Tool Error] Erro durante o scraping de {website_url}: {type(e).__name__}: {e}"