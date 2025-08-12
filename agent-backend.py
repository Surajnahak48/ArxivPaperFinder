Command 'ollama' not found, but can be installed with:
sudo snap install ollamaimport os
import asyncio
import arxiv
from typing import List, Dict

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.tools import FunctionTool

# Local Ollama model
ollama_brain = OllamaChatCompletionClient(
    model='llama2',
    base_url='http://localhost:9090'
)

# Tool: Search arXiv
def arxiv_search(query: str, max_results: int = 5) -> List[Dict]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers: List[Dict] = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "author": [a.name for a in result.authors],
            "published": result.published.strftime("%Y-%m-%d"),
            "summary": result.summary,
            "pdf_url": result.pdf_url,
        })
    return papers

arxiv_tool = FunctionTool(arxiv_search)

# Agents
arxiv_researcher_agent = AssistantAgent(
    name='arxiv_search_agent',
    description='Creates arXiv queries and retrieves papers',
    model_client=ollama_brain,
    tools=[arxiv_tool],
    system_message=(
        "Given a user topic, think of the best arXiv query. "
        "When the tool returns, choose exactly the number of papers requested "
        "and pass them as concise JSON to the summarizer."
    ),
)

summarizer_agent = AssistantAgent(
    name='summarizer_agent',
    description='Summarizes research papers',
    model_client=ollama_brain,
    system_message=(
        "You are an expert researcher. When you receive the JSON list of papers, "
        "write a literature-review style report in markdown:\n"
        "1. Start with a 2-3 sentence introduction of the topic.\n"
        "2. Then include one bullet per paper with: title (as Markdown link), "
        "author, the specific problem tackled, and its key contribution.\n"
        "3. Close with a single-sentence takeaway."
    ),
)

team = RoundRobinGroupChat(
    participants=[arxiv_researcher_agent, summarizer_agent],
    max_turns=2
)

async def run_team():
    task = 'Conduct a literature review on the topic - Autogen and return exactly 5 papers'
    async for msg in team.run_stream(task=task):
        print(msg)

if __name__ == '__main__':
    asyncio.run(run_team())
