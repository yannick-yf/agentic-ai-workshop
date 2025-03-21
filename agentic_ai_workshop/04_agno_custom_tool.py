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

import requests
import json
from rich import print
import json
from textwrap import dedent

import httpx
from agno.agent import Agent
from agno.models.openai import OpenAIChat

from geopy.geocoders import Nominatim

# 1. Create Custom Tool
def get_weather(location: str) -> float:
    geolocator = Nominatim(user_agent="weather_app")
    location_data = geolocator.geocode(location)
    latitude, longitude = location_data.latitude, location_data.longitude if location_data else (0, 0)
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    return data['current']['temperature_2m']



def get_top_hackernews_stories(num_stories: int = 10) -> str:
    """Use this function to get top stories from Hacker News.

    Args:
        num_stories (int): Number of stories to return. Defaults to 10.

    Returns:
        str: JSON string of top stories.
    """

    # Fetch top story IDs
    response = httpx.get("https://hacker-news.firebaseio.com/v0/topstories.json")
    story_ids = response.json()

    # Fetch story details
    stories = []
    for story_id in story_ids[:num_stories]:
        story_response = httpx.get(
            f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        )
        story = story_response.json()
        if "text" in story:
            story.pop("text", None)
        stories.append(story)
    return json.dumps(stories)


# Create a Tech News Reporter Agent with a Silicon Valley personality
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    instructions=dedent("""\
        You are a tech-savvy Hacker News reporter with a passion for all things technology! 🤖
        Think of yourself as a mix between a Silicon Valley insider and a tech journalist.

        Your style guide:
        - Start with an attention-grabbing tech headline using emoji
        - Present Hacker News stories with enthusiasm and tech-forward attitude
        - Keep your responses concise but informative
        - Use tech industry references and startup lingo when appropriate
        - End with a catchy tech-themed sign-off like 'Back to the terminal!' or 'Pushing to production!'

        Remember to analyze the HN stories thoroughly while keeping the tech enthusiasm high!\
    """),
    tools=[get_top_hackernews_stories],
    show_tool_calls=True,
    markdown=True,
)

# Example questions to try:
# - "What are the trending tech discussions on HN right now?"
# - "Summarize the top 5 stories on Hacker News"
# - "What's the most upvoted story today?"
agent.print_response("Summarize the top 5 stories on hackernews?", stream=True)