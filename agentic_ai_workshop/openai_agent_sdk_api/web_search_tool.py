from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4o",
    tools=[{"type": "web_search_preview"}],
    input="Give me the nba games results for the game of the 17th of March"
)

print(response.output_text)