from agents import Agent, Runner, WebSearchTool
import asyncio

agent = Agent(
    name="Assistant",
    tools=[
        WebSearchTool(),
    ],
)

async def main():
    result = await Runner.run(agent, "Give me the nba games results for the game of the 30th of March.")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())