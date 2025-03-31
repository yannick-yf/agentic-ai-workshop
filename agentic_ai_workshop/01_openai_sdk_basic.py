from openai import OpenAI
from textwrap import dedent

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    instructions=dedent(""""""),
    input="Tell me about Yannick Flores Experience"
)

print(response.output_text)