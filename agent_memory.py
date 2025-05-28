from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType

import os

from dotenv import load_dotenv
load_dotenv()

#os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

agent = Agent(
    name="Memory Agent",
    model=OpenAIChat(id="gpt-4.1-mini"),
    description="you are a thai cuisine expert",
    instructions=["search your knowledge base for thai recepies",
                  "if question is better suited for web then search the web to fill gaps",
                  "prefer information in your knowledge base over web results"],
    knowledge=PDFUrlKnowledgeBase(
        urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="recipes",
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(id="text-embedding-3-small"),
        )
    ),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)

if agent.knowledge is not None:
    agent.knowledge.load()

agent.print_response("how do i make chicken and galangal in coconut milk")
agent.print_response("whats the history of thai curry", stream=True)