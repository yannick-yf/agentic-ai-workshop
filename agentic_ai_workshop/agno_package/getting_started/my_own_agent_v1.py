"""This is Agent Example for the workshop.

Goal is to get the weather data for a given location.

This example shows how to create a basic AI agent interacting with a tools and web search.

Example prompts to try:
- "What's the weather in Mulhouse tomorrow?"

"""

from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

#TODO: Modify the script so it gives nba games recommendation to watch according to a specific date.

# Create a News Reporter Agent with a fun personality
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    instructions=dedent("""\
        You are an Weather Analyst
        Think of yourself as a mix between a sports expert and a sharp journalist.

        Follow these guidelines for every request:
        1. Identofy the location from the user request
        2. Use the search tool to find current, accurate weather information.
        3. Structure your weather reports.
        4. Keep responses concise but informative (1-2 paragraphs max)

        Remember: Always verify facts through web searches.\
    """),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
)

# Example usage
agent.print_response(
    "What's the weather in Basel today ?", stream=True
)
