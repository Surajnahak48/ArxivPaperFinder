from dotenv import load_dotenv
load_dotenv()
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
import os
import asyncio
from autogen_agentchat.teams import RoundRobinGroupChat
import arxiv
from typing import List,Dict,AsyncGenerator

# Use OllamaChatCompletionClient for local model (e.g., llama2, deepseek, etc.)
ollama_brain = OllamaChatCompletionClient(model='llama2', base_url='http://localhost:11434')


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
    model_client=ollama_brain,
    tools = [arxiv_search],
    system_message = (
        "Give a user topic, think of the best arXiv query. when the tool"
        "returns, choose exactly the number of papers requested and pass"
        "them as concise JSON to the summarizer."
    ),
)

summarizer_agent = AssistantAgent(
    name='summarizer_agent',
    description='An agent that summarizes result',
    model_client=ollama_brain,
    system_message=(
        "yoou are an expert researcher. when you recieve the JSON list of "
        "papers, write a literature-review style report im markdown:\n" \
        "1. Start with a 2-3 sentance introduction of the topic.\n" \
        "2. Then include one bullet per paper with: title (as Markdown "
        "link), author, the specific problem tackled, and its key"
        "contribution.\n" \
        "3. Close with a single-sentance takeaway."
    ),
)

team = RoundRobinGroupChat(
    participants=[arxiv_researcher_agent, summarizer_agent],
    max_turns=2
)

async def run_team():

    task= 'Conduct a literature review on the topic - Autogen and return exactly 5 papers'

    async for msg in team.run_stream(task=task):
        print(msg)



if (__name__ == '__main__'):
    asyncio.run(run_team())
