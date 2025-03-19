"""🗽 Web Searching News Reporter - Your AI News Buddy that searches the web

This example shows how to create an AI news reporter agent that can search the web
for real-time news and present them with a distinctive NYC personality. The agent combines
web searching capabilities with engaging storytelling to deliver news in an entertaining way.

Example prompts to try:
- "What's the latest headline from Wall Street?"
- "Tell me about any breaking news in Central Park"
- "What's happening at Yankees Stadium today?"
- "Give me updates on the newest Broadway shows"
- "What's the buzz about the latest NYC restaurant opening?"

Run `pip install openai duckduckgo-search agno` to install dependencies.
"""

from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

#TODO: Modify the script so it gives nba games recommendation to watch according to a specific date.

# Create a News Reporter Agent with a fun personality
agent = Agent(
    model=OpenAIChat(id="gpt-4o-search-preview"),
    instructions=dedent("""\
        You are an NBA Analyst and sports news reporter.
        Think of yourself as a mix between a sports expert and a sharp journalist.

        Follow these guidelines for every request:
        1. Start by creating the list of NBA games of the requested day. If no date provided use the last games played.
        2. Use the search tool to find current, accurate information.
        3. Structure your reports in clear sections:
        - Catchy headline of each games
        - Brief recap of every games
        4. Keep responses concise but informative (1-2 paragraphs max)
        5. End with a recommendation on which game I should watch.

        Remember: Always verify facts through web searches without spoiling the results of the night while being engaging I want to see the games.\
    """),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
)

# Example usage
agent.print_response(
    "Tell me about the NBA games played on the 17th of March 2025.", stream=True
)

# More example prompts to try:
"""
Try these engaging news queries:
1. "What's the latest development in NYC's tech scene?"
2. "Tell me about any upcoming events at Madison Square Garden"
3. "What's the weather impact on NYC today?"
4. "Any updates on the NYC subway system?"
5. "What's the hottest food trend in Manhattan right now?"
"""
