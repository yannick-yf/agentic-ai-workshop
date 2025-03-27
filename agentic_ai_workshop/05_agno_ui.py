"""
UI for the weather Agent
"""

"""🛠️ Writing Your Own Tool - An Example Using Hacker News API

This example shows how to create and use your own custom tool with Agno.
You can replace the Hacker News functionality with any API or service you want!

Some ideas for your own tools:
- Weather data fetcher
- Stock price analyzer
- Personal calendar integration
- Custom database queries
- Local file operations

Run `pip install openai httpx agno` to install dependencies.
"""
import gradio as gr
import requests
import json
from textwrap import dedent
import httpx
from agno.agent import Agent
from agno.models.openai import OpenAIChat

from geopy.geocoders import Nominatim

# 1. Create Custom Tool
def get_weather(location: str = "Martigues, France"):
    """Use this function to get weather data from a given location

    Args:
        location (str): Location from where to get the weather data

    Returns:
        float: Temperature at the float format
    """

    geolocator = Nominatim(user_agent="weather_app")
    location_data = geolocator.geocode(location)
    latitude, longitude = location_data.latitude, location_data.longitude if location_data else (0, 0)
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    return str(data['current']['temperature_2m'])

def weather_ai_agent(question):
    # Create a Tech News Reporter Agent with a Silicon Valley personality
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=dedent("""\
            You role is to give the weather for the location provided by the user.
            You will have to use the tool get_weather.
            The tool details are:
            {
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
            }
            
            From the user question you will have to exctract the location and pass it to the tool.
        """),
        tools=[get_weather],
        show_tool_calls=True,
        markdown=True,
    )

    response = agent.run(question, stream_intermediate_steps=False)
    return(str(response.content))

demo = gr.Interface(
    fn=weather_ai_agent,
    inputs="text",
    outputs="text",
    title="AI Assistant"
)

if __name__ == "__main__":
    demo.launch()
