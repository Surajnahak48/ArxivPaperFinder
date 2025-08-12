from dotenv import load_dotenv
load_dotenv()
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
import os
import arxiv
from typing import List,Dict,AsyncGenerator

openai_brain = OpenAIChatCompletionClient(model='gpt-4o', api_key=os.getenv('OPENAI_API_KEY'))

ollama_brain = OllamaChatCompletionClient(
    model='deepseek-llm-7b-base',  # or the name you see in ollama list
    base_url='http://localhost:11434'
)


def arxiv_search(query: str, max_results: int = 5) -> List[Dict]:
    """Return a compact list of arXiv papers matching *query*.

    Each element contains: ``title``, ``author``, ``published``, ``summary``, and 
    ``pdf_url``. The helper is wrapped as an AutoGen *FunctionTool* below so it
    can be invoked by agent through the normal tool-use mechanism.
    """

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers: List[Dict] = []
    for result in client.results(search):
        papers.append(
            {
                "title": result.title,
                "author": [a.name for a in result.authors],
                "published": result.published.strftime("%Y-%m-%d"),
                "summary": result.summary,
                "pdf_url": result.pdf_url,
            }
        )
    return papers


arxiv_researcher_agent = AssistantAgent(
    name='arxiv_search_agent',
    description='Create arXiv queries and retriveves candidates papers',
    model_client=openai_brain,
    tools = [arxiv_search],
    system_message = ''
)

print(arxiv_search(query='agents'))