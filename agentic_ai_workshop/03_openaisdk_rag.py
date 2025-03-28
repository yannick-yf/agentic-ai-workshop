"""
Go to OpenAI urls and populate the VectorDataBase

"""

from openai import OpenAI
from rich import print
client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    input="Tell me about the Yannick Flores Experience",
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["vs_67d9a15c84d48191a5b9f38791be8ff0"]
    }]
)
print(response)