"""
UI for the weather Agent
"""

import gradio as gr
from openai import OpenAI

from openai import OpenAI
import requests
import json
from rich import print
from geopy.geocoders import Nominatim

client = OpenAI()

# 1. Create Custom Tool
def get_weather(location):
    geolocator = Nominatim(user_agent="weather_app")
    location_data = geolocator.geocode(location)
    latitude, longitude = location_data.latitude, location_data.longitude if location_data else (0, 0)
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    return data['current']['temperature_2m']

# 2. Create Custom Tool Definition
tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country e.g. Bogotá, Colombia"
            }
        },
        "required": [
            "location"
        ],
        "additionalProperties": False
    }
}]


client = OpenAI()

def weather_ai_agent(question):
    input_messages = [{"role": "user", "content": question}]
    response = client.responses.create(
        model="gpt-4o-mini",
        input=input_messages,
        tools=tools
    )

    tool_call = response.output[0]
    args = json.loads(tool_call.arguments)
    result = get_weather(args["location"])

    input_messages.append(tool_call)  
    input_messages.append({           
        "type": "function_call_output",
        "call_id": tool_call.call_id,
        "output": str(result)
    })

    response_2 = client.responses.create(
        model="gpt-4o-mini",
        input=input_messages,
        tools=tools,
    )

    return response_2.output_text

demo = gr.Interface(
    fn=weather_ai_agent,
    inputs="text",
    outputs="text",
    title="AI Assistant"
)

if __name__ == "__main__":
    demo.launch()