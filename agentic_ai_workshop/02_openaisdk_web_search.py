from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-mini-search-preview",
    web_search_options={},
    messages=[
        {
            "role": "user",
            "content": "Give me the nba games results for the game of the 19th of March.",
        }
    ],
)

print(completion.choices[0].message.content)