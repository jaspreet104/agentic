from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

import os

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

agent1 = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Groq(id="qwen-qwq-32b"),
    tools=[DuckDuckGoTools(),],
    instructions="Always include the source of your information.",
    show_tool_calls=True,
    markdown=True
)

agent2 = Agent(
    name="Finance Agent",
    role="Get financial information",
    model=OpenAIChat(id="gpt-4.1-mini"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True),],
    instructions="use tables to present financial data.",
    show_tool_calls=True,
    markdown=True
)

#agent1.print_response("What is the capital of France?")
#agent2.print_response("Should i invest in Tesla or Nvidia?")

agent_team = Agent(
    name="Multi-Agent Team",
    model=Groq(id="qwen-qwq-32b"),
    team=[agent1, agent2],
    instructions=["always include sources", "use tables to show data"],
    show_tool_calls=True,
    markdown=True
)

agent_team.print_response("anayze companies like tesla, nvidia, apple and suggest which to buy for long term")