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

import json
from textwrap import dedent

import httpx
from agno.agent import Agent
from agno.models.openai import OpenAIChat


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
agent.print_response("Summarize the top 10 stories on hackernews?", stream=True)

#Idea for Workshop
"""
1. Interactive Data Explorer Agent
Overview: Create an agent that ingests data files (e.g., CSV, JSON) and provides an interactive, natural language interface to explore the dataset.
Features:
Accepts natural language queries (e.g., “Show me trends in sales over the last year”).
Automatically generates visualizations such as graphs or heatmaps.
Integrates with Python libraries like Pandas, Matplotlib, or Seaborn for data processing and plotting.
Workshop Benefit: Participants get hands-on experience with data manipulation, visualization, and natural language processing in a single tool.

2. Code Assistant & Debugging Agent
Overview: Develop an agent that assists with code reviews, debugging, and style suggestions.
Features:
Analyzes Python scripts for common pitfalls or style issues.
Offers suggestions or code snippets for optimization.
Integrates with static analysis tools (e.g., pylint) to provide real-time feedback.
Workshop Benefit: Demonstrates how AI can streamline the coding process and improve code quality, which is practical for both beginners and experienced developers.

3. Automated Workflow Orchestrator
Overview: Build an agent that helps automate routine data science workflows—from data ingestion to model training and reporting.
Features:
Orchestrates multiple steps (data cleaning, feature engineering, model evaluation).
Provides a dashboard to track progress and visualize outcomes.
Supports custom plugin modules to extend functionality.
Workshop Benefit: Gives insight into building modular, extensible systems and shows how agents can handle complex pipelines.

4. AI-Powered Meeting and Q&A Assistant
Overview: Develop an agent to assist during workshops or meetings.
Features:
Transcribes speech in real time and summarizes key points.
Answers technical questions on-the-fly using pre-loaded documentation or code examples.
Manages workshop schedules, agenda items, and participant queries.
Workshop Benefit: Illustrates real-world applications of AI in managing information overload and improving engagement during live events.

. Multimodal Data Insights Agent
Overview: Create an agent that combines text and image data analysis.
Features:
Processes textual reports along with visual data (charts, diagrams).
Uses computer vision and NLP to derive insights and correlations.
Generates comprehensive reports that merge findings from both data types.
Workshop Benefit: Demonstrates integration of different AI techniques and the practical use of multimodal data analysis.
"""
