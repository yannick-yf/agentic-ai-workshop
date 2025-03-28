from os import getenv
from agno.agent import Agent, RunResponse
from agno.models.openai.like import OpenAILike

agent = Agent(
    model=OpenAILike(
        id="gpt-4o-mini-search-preview",
        api_key=getenv("OPENAI_API_KEY")
    )
)

# Print the response in the terminal
agent.print_response("Give me the nba games results for the game of the 19th of March.")