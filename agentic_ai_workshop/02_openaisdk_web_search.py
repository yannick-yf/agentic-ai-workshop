from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-mini-search-preview",
    web_search_options={},
    messages=[
        {
            "role": "user",
            "content": "Tell me about Yannick Flores experience",
        }
    ],
)

print(completion.choices[0].message.content)