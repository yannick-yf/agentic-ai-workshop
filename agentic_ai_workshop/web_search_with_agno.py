"""🗽 Agent with Tools - Your AI News Buddy that can search the web

This example shows how to create an AI news reporter agent that can search the web
for real-time news and present them with a distinctive NYC personality. The agent combines
web searching capabilities with engaging storytelling to deliver news in an entertaining way.

Run `pip install openai duckduckgo-search agno` to install dependencies.
"""

from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

# Create a News Reporter Agent with a fun personality
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    # instructions=dedent(""""""),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
)

# Example usage
agent.print_response(
    "Give me the results of all the nba games played during the night of the 30th / 31st of March 2025.", stream=True
)
