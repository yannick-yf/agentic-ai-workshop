"""This is Agent Example for the workshop.

Goal is to get the weather data for a given location.

This example shows how to create a basic AI agent interacting with a tools and web search.

Example prompts to try:
- "What's the weather in Mulhouse tomorrow?"

"""

from agno.agent import Agent, Message, Response, Tool
import requests
from datetime import datetime, timedelta
import os
from typing import Dict, Any, Optional


class WeatherTool(Tool):
    name = "get_weather"
    description = "Get weather information for a specific location and date"
    
    def run(self, location: str, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Gets weather data for a specific location and date.
        
        Args:
            location: City name or location
            date: Date string (today, tomorrow, or YYYY-MM-DD format). Defaults to today if not specified.
            
        Returns:
            Dictionary with weather information
        """
        # Replace with your actual API key
        API_KEY = os.environ.get("OPENWEATHER_API_KEY", "YOUR_OPENWEATHER_API_KEY")
        
        # Parse date
        if not date or date.lower() == "today":
            target_date = datetime.now()
        elif date.lower() == "tomorrow":
            target_date = datetime.now() + timedelta(days=1)
        else:
            try:
                target_date = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                return {"error": f"Invalid date format: {date}. Please use 'today', 'tomorrow', or YYYY-MM-DD."}
        
        # Calculate if we need current weather or forecast
        days_diff = (target_date.date() - datetime.now().date()).days
        
        if days_diff == 0:
            # Get current weather
            url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
            response = requests.get(url)
            
            if response.status_code != 200:
                return {"error": f"Could not get weather data for {location}. Error: {response.json().get('message', 'Unknown error')}"}
            
            data = response.json()
            
            return {
                "location": location,
                "date": "today",
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "description": data["weather"][0]["description"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"],
                "pressure": data["main"]["pressure"]
            }
        
        elif 0 < days_diff <= 5:  # Most free APIs limit to 5-day forecast
            # Get forecast
            url = f"https://api.openweathermap.org/data/2.5/forecast?q={location}&appid={API_KEY}&units=metric"
            response = requests.get(url)
            
            if response.status_code != 200:
                return {"error": f"Could not get forecast data for {location}. Error: {response.json().get('message', 'Unknown error')}"}
            
            data = response.json()
            target_date_str = target_date.strftime("%Y-%m-%d")
            
            # Find forecast for the target date (most appropriate time - noon)
            forecast_items = data["list"]
            target_forecast = None
            
            for item in forecast_items:
                item_date = datetime.fromtimestamp(item["dt"])
                if item_date.strftime("%Y-%m-%d") == target_date_str and 10 <= item_date.hour <= 14:
                    target_forecast = item
                    break
            
            if target_forecast:
                return {
                    "location": location,
                    "date": target_date_str,
                    "temperature": target_forecast["main"]["temp"],
                    "feels_like": target_forecast["main"]["feels_like"],
                    "description": target_forecast["weather"][0]["description"],
                    "humidity": target_forecast["main"]["humidity"],
                    "wind_speed": target_forecast["wind"]["speed"],
                    "pressure": target_forecast["main"]["pressure"]
                }
            else:
                return {"error": f"Could not find forecast for {location} on {target_date_str}"}
        else:
            return {"error": f"Cannot provide weather forecast for {days_diff} days in the future. Limited to 5-day forecast."}


# Create the weather agent
def create_weather_agent():
    weather_agent = Agent(
        name="Weather Assistant",
        description="An agent that provides weather information for different locations and dates",
        tools=[WeatherTool()]
    )
    
    @weather_agent.on_message
    async def handle_message(message: Message) -> Response:
        """Process incoming messages and respond with weather information."""
        user_query = message.content
        
        # Simple response for basic greeting
        if user_query.lower() in ["hello", "hi", "hey"]:
            return Response("Hello! I can provide weather information for any location. Just ask me something like 'What's the weather in London tomorrow?'")
        
        # Process the query to extract location and date
        import re
        
        # Extract location 
        location_match = re.search(r"(?:weather|temperature)(?:\s+in|\s+at|\s+for)?\s+([A-Za-z\s]+?)(?:\s+on|\s+for|\s+tomorrow|\s+today|\s+in|\s+at|\s*\?|$)", user_query, re.IGNORECASE)
        if not location_match:
            return Response("I couldn't determine the location from your query. Please specify a city or location, for example: 'What's the weather in Paris tomorrow?'")
        
        location = location_match.group(1).strip()
        
        # Extract date
        date = None
        if "tomorrow" in user_query.lower():
            date = "tomorrow"
        elif "today" in user_query.lower():
            date = "today"
        elif "in" in user_query.lower() and "days" in user_query.lower():
            days_match = re.search(r"in\s+(\d+)\s+days", user_query, re.IGNORECASE)
            if days_match:
                days = int(days_match.group(1))
                date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        
        # Default to today if no date is specified
        if not date:
            date = "today"
        
        # Get weather data using the tool
        weather_tool = weather_agent.get_tool("get_weather")
        weather_data = weather_tool.run(location=location, date=date)
        
        if "error" in weather_data:
            return Response(f"Sorry, I couldn't get the weather information: {weather_data['error']}")
        
        # Format the response
        date_str = "today" if weather_data["date"] == "today" else f"on {weather_data['date']}"
        response_text = (
            f"Weather in {weather_data['location']} {date_str}:\n"
            f"• Temperature: {weather_data['temperature']}°C (feels like {weather_data['feels_like']}°C)\n"
            f"• Conditions: {weather_data['description']}\n"
            f"• Humidity: {weather_data['humidity']}%\n"
            f"• Wind speed: {weather_data['wind_speed']} m/s\n"
            f"• Pressure: {weather_data['pressure']} hPa"
        )
        
        return Response(response_text)
    
    return weather_agent


if __name__ == "__main__":
    # Create and run the agent
    weather_agent = create_weather_agent()
    
    # For demonstration, you could add a simple CLI interface
    print("Weather Agent is ready! (Type 'exit' to quit)")
    while True:
        user_input = input("> ")
        if user_input.lower() == "exit":
            break
        
        # Process the message and get response
        response = weather_agent.process_message(Message(content=user_input))
        print(response.content)