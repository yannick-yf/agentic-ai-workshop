"""🎬 Movie Script Generator - Your AI Screenwriting Partner

This example shows how to use structured outputs with AI agents to generate
well-formatted movie script concepts. It shows two approaches:
1. JSON Mode: Traditional JSON response parsing
2. Structured Output: Enhanced structured data handling

Example prompts to try:
- "Tokyo" - Get a high-tech thriller set in futuristic Japan
- "Ancient Rome" - Experience an epic historical drama
- "Manhattan" - Explore a modern romantic comedy
- "Amazon Rainforest" - Adventure in an exotic location
- "Mars Colony" - Science fiction in a space settlement

Run `pip install openai agno` to install dependencies.
"""

from textwrap import dedent
from typing import List

from agno.agent import Agent, RunResponse  # noqa
from agno.models.openai import OpenAIChat
from pydantic import BaseModel, Field


class MovieScript(BaseModel):
    setting: str = Field(
        ...,
        description="A richly detailed, atmospheric description of the movie's primary location and time period. Include sensory details and mood.",
    )
    ending: str = Field(
        ...,
        description="The movie's powerful conclusion that ties together all plot threads. Should deliver emotional impact and satisfaction.",
    )
    genre: str = Field(
        ...,
        description="The film's primary and secondary genres (e.g., 'Sci-fi Thriller', 'Romantic Comedy'). Should align with setting and tone.",
    )
    name: str = Field(
        ...,
        description="An attention-grabbing, memorable title that captures the essence of the story and appeals to target audience.",
    )
    characters: List[str] = Field(
        ...,
        description="4-6 main characters with distinctive names and brief role descriptions (e.g., 'Sarah Chen - brilliant quantum physicist with a dark secret').",
    )
    storyline: str = Field(
        ...,
        description="A compelling three-sentence plot summary: Setup, Conflict, and Stakes. Hook readers with intrigue and emotion.",
    )


# Agent that uses JSON mode
json_mode_agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    description=dedent("""\
        You are an acclaimed Hollywood screenwriter known for creating unforgettable blockbusters! 🎬
        With the combined storytelling prowess of Christopher Nolan, Aaron Sorkin, and Quentin Tarantino,
        you craft unique stories that captivate audiences worldwide.

        Your specialty is turning locations into living, breathing characters that drive the narrative.\
    """),
    instructions=dedent("""\
        When crafting movie concepts, follow these principles:

        1. Settings should be characters:
           - Make locations come alive with sensory details
           - Include atmospheric elements that affect the story
           - Consider the time period's impact on the narrative

        2. Character Development:
           - Give each character a unique voice and clear motivation
           - Create compelling relationships and conflicts
           - Ensure diverse representation and authentic backgrounds

        3. Story Structure:
           - Begin with a hook that grabs attention
           - Build tension through escalating conflicts
           - Deliver surprising yet inevitable endings

        4. Genre Mastery:
           - Embrace genre conventions while adding fresh twists
           - Mix genres thoughtfully for unique combinations
           - Maintain consistent tone throughout

        Transform every location into an unforgettable cinematic experience!\
    """),
    response_model=MovieScript,
)

# Agent that uses structured outputs
structured_output_agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    description=dedent("""\
        You are an acclaimed Hollywood screenwriter known for creating unforgettable blockbusters! 🎬
        With the combined storytelling prowess of Christopher Nolan, Aaron Sorkin, and Quentin Tarantino,
        you craft unique stories that captivate audiences worldwide.

        Your specialty is turning locations into living, breathing characters that drive the narrative.\
    """),
    instructions=dedent("""\
        When crafting movie concepts, follow these principles:

        1. Settings should be characters:
           - Make locations come alive with sensory details
           - Include atmospheric elements that affect the story
           - Consider the time period's impact on the narrative

        2. Character Development:
           - Give each character a unique voice and clear motivation
           - Create compelling relationships and conflicts
           - Ensure diverse representation and authentic backgrounds

        3. Story Structure:
           - Begin with a hook that grabs attention
           - Build tension through escalating conflicts
           - Deliver surprising yet inevitable endings

        4. Genre Mastery:
           - Embrace genre conventions while adding fresh twists
           - Mix genres thoughtfully for unique combinations
           - Maintain consistent tone throughout

        Transform every location into an unforgettable cinematic experience!\
    """),
    response_model=MovieScript,
    structured_outputs=True,
)

# Example usage with different locations
# json_mode_agent.print_response("Tokyo", stream=True)
structured_output_agent.print_response("Ancient Rome", stream=True)

# More examples to try:
"""
Creative location prompts to explore:
1. "Underwater Research Station" - For a claustrophobic sci-fi thriller
2. "Victorian London" - For a gothic mystery
3. "Dubai 2050" - For a futuristic heist movie
4. "Antarctic Research Base" - For a survival horror story
5. "Caribbean Island" - For a tropical adventure romance
"""

# To get the response in a variable:
# from rich.pretty import pprint

# json_mode_response: RunResponse = json_mode_agent.run("New York")
# pprint(json_mode_response.content)
# structured_output_response: RunResponse = structured_output_agent.run("New York")
# pprint(structured_output_response.content)

# json output
"""
{                                                                                                                                                                                     ┃
┃   "setting": "Set in the neon-soaked streets of Shibuya during the vibrant summer phase of the Tokyo 2020 Olympics, this bustling metropolis serves as a character unto itself. The s ┃
┃   "ending": "In a climactic race against time, the characters unite in the heart of Shibuya Crossing. Just as the fireworks symbolize victory, they expose a high-stakes conspiracy t ┃
┃   "genre": "Action-Drama",                                                                                                                                                            ┃
┃   "name": "Chasing Shadows in Tokyo",                                                                                                                                                 ┃
┃   "characters": [                                                                                                                                                                     ┃
┃     "Kai Tanaka - a determined 17-year-old sprinter striving for Olympic gold while hiding a family secret.",                                                                         ┃
┃     "Aiko Yamamoto - a sharp-witted journalist who seeks to uncover the truth behind a corruption scandal linked to the games.",                                                      ┃
┃     "Hiroki Sato - an ex-gang member turned street artist, grappling with his tumultuous past while offering street-smart solutions.",                                                ┃
┃     "Lena Fischer - an ambitious foreign athlete dealing with cultural clashes and the pressure of performance.",                                                                     ┃
┃     "Yuki Nakamura - an enigmatic hacker who deeply understands Tokyo's digital landscape, with motives shrouded in secrecy."                                                         ┃
┃   ],                                                                                                                                                                                  ┃
┃   "storyline": "Amidst the fervor of the Tokyo Olympics, a teenage sprinter, Kai, accidentally stumbles upon a nefarious plot to sabotage the games. As he teams up with a journalist ┃
┃ }  
"""

# structured output
"""
{                                                                                                                                                                                     ┃
┃   "setting": "Ancient Rome is a sprawling, sun-drenched city filled with the roar of crowds and the scent of fresh bread wafting from local bakeries. The grandeur of the Colosseum l ┃
┃   "ending": "In a heart-stopping climax, Juno, once a powerless slave, stands triumphant at the Colosseum's apex, revealing the true identities of the conspirators who wished to und ┃
┃   "genre": "Historical Drama, Action",                                                                                                                                                ┃
┃   "name": "To Rule the Shadows",                                                                                                                                                      ┃
┃   "characters": [                                                                                                                                                                     ┃
┃     "Juno Decius - a fierce and cunning female slave who dreams of freedom and justice.",                                                                                             ┃
┃     "Marcus Aurelius - a charismatic yet ambitious senator torn between loyalty and power.",                                                                                          ┃
┃     "Caius Lucius - a battle-hardened gladiator seeking to escape the arena and unite the oppressed.",                                                                                ┃
┃     "Octavia - Juno's fierce best friend and fellow slave, whose loyalty drives the emotional core of the story.",                                                                    ┃
┃     "Cassius - a corrupt senator orchestrating chaos for personal gain, embodying the story's darkest element."                                                                       ┃
┃   ],                                                                                                                                                                                  ┃
┃   "storyline": "In the heart of Ancient Rome, Juno Decius fights for her freedom amidst political upheaval and the deadly allure of gladiatorial combat. As she uncovers a sinister p ┃
┃ }  
"""