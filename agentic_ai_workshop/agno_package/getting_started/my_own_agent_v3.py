"""This is Agent Example for the workshop.

Goal is to read data from a CSV with the nba schedule for the 2025 season,
Filter the data accoring to the user requests
Perform a web search 

This example shows how to create a basic AI agent interacting with a tools and web search.

Example prompts to try:
- "What's the weather in Mulhouse tomorrow?"

"""

from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-search-preview",
    #web_search_options={},
    messages=[
        {
            "role": "user",
            "content": "Give me the nba games results for the game of the 17th of March?",
        }
    ],
)

print(completion.choices[0].message.content)