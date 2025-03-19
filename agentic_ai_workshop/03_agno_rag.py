"""🧠 Recipe Expert with Knowledge - Your AI Thai Cooking Assistant!

This example shows how to create an AI cooking assistant that combines knowledge from a
curated recipe database with web searching capabilities. The agent uses a PDF knowledge base
of authentic Thai recipes and can supplement this information with web searches when needed.

Example prompts to try:
- "How do I make authentic Pad Thai?"
- "What's the difference between red and green curry?"
- "Can you explain what galangal is and possible substitutes?"
- "Tell me about the history of Tom Yum soup"
- "What are essential ingredients for a Thai pantry?"
- "How do I make Thai basil chicken (Pad Kra Pao)?"

Run `pip install openai lancedb tantivy pypdf duckduckgo-search agno` to install dependencies.
"""

from textwrap import dedent

from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.vectordb.lancedb import LanceDb, SearchType

# Create a Recipe Expert Agent with knowledge of Thai recipes
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    instructions=dedent("""\
        You are a Data Science Manager who is hiring for a open position! 🧑‍🍳
        Think of yourself tech leader tht need to add a good candidate in your team.

        Follow these steps when answering questions:
        1. First, search the knowledge base for CVs
        2. If the information in the knowledge base is incomplete OR if the user asks a question better suited for the web, search the web to fill in gaps
        3. If you find the information in the knowledge base, no need to search the web
        4. Always prioritize knowledge base information over web results for accurate information about candidates

        Remember:
        - Always verify information with the knowledge base
        - Clearly indicate when information comes from web sources if web sources is used
    """),
    knowledge=PDFKnowledgeBase(
        path="./agentic_ai_workshop/data/CV_Yannick_Flores_Sr_Data_Scientist.pdf",
        vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="recipe_knowledge",
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(id="text-embedding-3-small"),
        ),
    ),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
    add_references=True,
)

# Comment out after the knowledge base is loaded
if agent.knowledge is not None:
    agent.knowledge.load()

agent.print_response(
    "Tell me about the Yannick Flores Experience", stream=True
)