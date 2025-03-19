"""
Goal: Create Multi Agetns with multiple components:
- Web Search Capabilites
- Calling API with custom tools
- RAG system
- Use context, State and Session
- Github action to run the E2E agent via batch

Agent Definition & Capabilites:
- Web Search for the international and French National News
- Web Search for NBA games Results 
- Agent Recommending What is the game to watch using The 'Web Search for NBA games Results'
- Own tool to call weather data and get the weather data in Basel
- Agent that sumerize all the output of previous Agents
- Tool creating a pdf from the Agent summary answer for a given date
- Tool to Ingest and RAG system the report for a given day
- Tool to query data for a previous day

Additional functionalities:
- Send email with report
- Create voice from report

"""
from os import getenv
from agno.agent import Agent, RunResponse
from agno.models.openai.like import OpenAILike

agent = Agent(
    model=OpenAILike(
        id="gpt-4o-search-preview",
        api_key=getenv("OPENAI_API_KEY")
    )
)

# # Print the response in the terminal
# agent.print_response("Using the recent review of Assassins creed Shaodws, create a paragraph telling me if I should buy or not the game.")

# Print the response in the terminal
agent.print_response("Give me the nba games results for the game of the 19th of March.")