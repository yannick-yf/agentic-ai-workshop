"""🗞️ Agent Team - Your Professional News & Finance Squad!

This example shows how to create a powerful team of AI agents working together
to provide comprehensive financial analysis and news reporting. The team consists of:
1. Web Agent: Searches and analyzes latest news
2. Finance Agent: Analyzes financial data and market trends
3. Lead Editor: Coordinates and combines insights from both agents

Example prompts to try:
- "What's the latest news and financial performance of Apple (AAPL)?"
- "Analyze the impact of AI developments on NVIDIA's stock (NVDA)"
- "How are EV manufacturers performing? Focus on Tesla (TSLA) and Rivian (RIVN)"
- "What's the market outlook for semiconductor companies like AMD and Intel?"
- "Summarize recent developments and stock performance of Microsoft (MSFT)"

Run: `pip install openai duckduckgo-search yfinance agno` to install the dependencies
"""

from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[DuckDuckGoTools()],
    instructions=dedent("""\
        You are an experienced web researcher and news analyst! 🔍

        Follow these steps when searching for information:
        1. Start with the most recent and relevant sources
        2. Cross-reference information from multiple sources
        3. Prioritize reputable news outlets and official sources
        4. Always cite your sources with links
        5. Focus on market-moving news and significant developments

        Your style guide:
        - Present information in a clear, journalistic style
        - Use bullet points for key takeaways
        - Include relevant quotes when available
        - Specify the date and time for each piece of news
        - Highlight market sentiment and industry trends
        - End with a brief analysis of the overall narrative
        - Pay special attention to regulatory news, earnings reports, and strategic announcements\
    """),
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)
    ],
    instructions=dedent("""\
        You are a skilled financial analyst with expertise in market data! 📊

        Follow these steps when analyzing financial data:
        1. Start with the latest stock price, trading volume, and daily range
        2. Present detailed analyst recommendations and consensus target prices
        3. Include key metrics: P/E ratio, market cap, 52-week range
        4. Analyze trading patterns and volume trends
        5. Compare performance against relevant sector indices

        Your style guide:
        - Use tables for structured data presentation
        - Include clear headers for each data section
        - Add brief explanations for technical terms
        - Highlight notable changes with emojis (📈 📉)
        - Use bullet points for quick insights
        - Compare current values with historical averages
        - End with a data-driven financial outlook\
    """),
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=dedent("""\
        You are the lead editor of a prestigious financial news desk! 📰

        Your role:
        1. Coordinate between the web researcher and financial analyst
        2. Combine their findings into a compelling narrative
        3. Ensure all information is properly sourced and verified
        4. Present a balanced view of both news and data
        5. Highlight key risks and opportunities

        Your style guide:
        - Start with an attention-grabbing headline
        - Begin with a powerful executive summary
        - Present financial data first, followed by news context
        - Use clear section breaks between different types of information
        - Include relevant charts or tables when available
        - Add 'Market Sentiment' section with current mood
        - Include a 'Key Takeaways' section at the end
        - End with 'Risk Factors' when appropriate
        - Sign off with 'Market Watch Team' and the current date\
    """),
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    markdown=True,
)

# Example usage with diverse queries
agent_team.print_response(
    "Summarize analyst recommendations and share the latest news for S&P 500", stream=True
)
# agent_team.print_response(
#     "What's the market outlook and financial performance of AI semiconductor companies?",
#     stream=True,
# )
# agent_team.print_response(
#     "Analyze recent developments and financial performance of TSLA", stream=True
# )

# More example prompts to try:
"""
Advanced queries to explore:
1. "Compare the financial performance and recent news of major cloud providers (AMZN, MSFT, GOOGL)"
2. "What's the impact of recent Fed decisions on banking stocks? Focus on JPM and BAC"
3. "Analyze the gaming industry outlook through ATVI, EA, and TTWO performance"
4. "How are social media companies performing? Compare META and SNAP"
5. "What's the latest on AI chip manufacturers and their market position?"
"""


# gpt-4o answer
"""
┃ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ ┃
┃ ┃                                                                S&P 500: Navigating Volatility amid Tariff Concerns                                                                ┃ ┃
┃ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛ ┃
┃                                                                                                                                                                                       ┃
┃                                                                                                                                                                                       ┃
┃                                                                                   Executive Summary                                                                                   ┃
┃                                                                                                                                                                                       ┃
┃ The S&P 500 is currently navigating a turbulent landscape influenced by geopolitical tensions, with particular focus on tariff policies. Despite recent volatility and fears of a     ┃
┃ market correction, historical data suggests potential long-term resilience and gains.                                                                                                 ┃
┃                                                                                                                                                                                       ┃
┃                                                                                                                                                                                       ┃
┃                                                                         Financial Data and Analyst Sentiment                                                                          ┃
┃                                                                                                                                                                                       ┃
┃                                                                          Analyst Sentiment and Market Trends                                                                          ┃
┃                                                                                                                                                                                       ┃
┃  • Market Outlook: Analysts are currently weighing economic indicators such as GDP growth and inflation concerns which could impact the index.                                        ┃
┃  • Valuation Metrics: The overall price-to-earnings (P/E) ratio and current valuations compared to historical averages are key considerations.                                        ┃
┃  • Sector Performance: Dominant sectors like technology heavily influence the index, with the tech sector typically buoying overall performance.                                      ┃
┃  • Global Influences: External geopolitical events, notably U.S. tariff policies, significantly impact investor sentiment and market strategies.                                      ┃
┃                                                                                                                                                                                       ┃
┃                                                                                                                                                                                       ┃
┃                                                                             Latest News and Developments                                                                              ┃
┃                                                                                                                                                                                       ┃
┃  • Market Volatility Fueled by Tariff Concerns: The S&P 500 is nearing correction territory due to concerns over President Trump's tariff policies. Despite attempts, market          ┃
┃    sentiment remains stressed.                                                                                                                                                        ┃
┃     • MarketWatch                                                                                                                                                                     ┃
┃  • Trump's Stance on Tariffs: Escalations in tariff issues under President Trump have affected market stability, reflecting in a drop in market indices.                              ┃
┃     • Forbes                                                                                                                                                                          ┃
┃  • Historical Performance Vs. Current Volatility: The S&P 500 historically achieves robust annual returns despite intra-year drawdowns, suggesting potential for resilience.          ┃
┃     • Seeking Alpha                                                                                                                                                                   ┃
┃  • Goldman Sachs Lowers S&P 500 Target: Reflecting skepticism about the U.S. economic outlook, Goldman Sachs has reduced its year-end target for the S&P 500 to 6,200.                ┃
┃     • Bloomberg                                                                                                                                                                       ┃
┃                                                                                                                                                                                       ┃
┃                                                                                                                                                                                       ┃
┃                                                                                   Market Sentiment                                                                                    ┃
┃                                                                                                                                                                                       ┃
┃ Current sentiment is bearish, reflecting caution due to ongoing policy uncertainties and economic concerns. However, historical precedents suggest optimism for potential long-term   ┃
┃ gains.                                                                                                                                                                                ┃
┃                                                                                                                                                                                       ┃
┃                                                                                                                                                                                       ┃
┃                                                                                     Key Takeaways                                                                                     ┃
┃                                                                                                                                                                                       ┃
┃  • The S&P 500 faces short-term volatility due to tariff concerns, but may achieve long-term resilience.                                                                              ┃
┃  • Analysts recommend focusing on broader economic indicators and sector performances.                                                                                                ┃
┃  • Recent market adjustments reflect cautious optimism as political and economic uncertainties unfold.                                                                                ┃
┃                                                                                                                                                                                       ┃
┃                                                                                                                                                                                       ┃
┃                                                                                     Risk Factors                                                                                      ┃
┃                                                                                                                                                                                       ┃
┃  • Ongoing geopolitical tensions, particularly relating to U.S. tariff policies, remain a significant risk.                                                                           ┃
┃  • Inflationary pressures and economic indicators need close monitoring for future market strategies.                                                                                 ┃
┃                                                                                                                                                                                       ┃
┃ Market Watch Team                                                                                                                                                                     ┃
┃ March 12, 2025  
"""

#gpt-4o-mini answer
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ ┃
┃ ┃                                                          S&P 500 Market Update: Analyst Recommendations and Latest News                                                           ┃ ┃
┃ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛ ┃
┃                                                                                                                                                                                       ┃
┃                                                                                   Executive Summary                                                                                   ┃
┃                                                                                                                                                                                       ┃
┃ The S&P 500 is currently navigating a landscape fraught with challenges, primarily stemming from political developments, particularly regarding tariffs under the Trump               ┃
┃ administration. Market sentiment has turned cautious, with analysts adjusting their forecasts downward amidst uncertainties. Despite these headwinds, some sectors, particularly      ┃
┃ technology, are showing signs of recovery, which could provide a buffer against broader market declines.                                                                              ┃
┃                                                                                                                                                                                       ┃
┃ ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── ┃
┃                                                                           Analyst Recommendations Overview                                                                            ┃
┃                                                                                                                                                                                       ┃
┃                                                                  Summary of Analyst Recommendations for the S&P 500                                                                   ┃
┃                                                                                                                                                                                       ┃
┃ While exact analyst recommendations for the S&P 500 index are not available, a typical proportion for major indices may look like this:                                               ┃
┃                                                                                                                                                                                       ┃
┃                                                                                                                                                                                       ┃
┃   Rating   Proportion (%)                                                                                                                                                             ┃
┃  ━━━━━━━━━━━━━━━━━━━━━━━━━                                                                                                                                                            ┃
┃   Buy      40%                                                                                                                                                                        ┃
┃   Hold     50%                                                                                                                                                                        ┃
┃   Sell     10%                                                                                                                                                                        ┃
┃                                                                                                                                                                                       ┃
┃                                                                                                                                                                                       ┃
┃  • Buy: Belief that the index will outperform the market over the next year.                                                                                                          ┃
┃  • Hold: Suggests keeping current positions without expecting significant movement.                                                                                                   ┃
┃  • Sell: Indicates a divestment recommendation due to potential underperformance.                                                                                                     ┃
┃                                                                                                                                                                                       ┃
┃ Keeping an eye on macroeconomic trends, policy changes, and corporate earnings reports is crucial for investors.                                                                      ┃
┃                                                                                                                                                                                       ┃
┃ ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── ┃
┃                                                                              Latest News on the S&P 500                                                                               ┃
┃                                                                                                                                                                                       ┃
┃ Here are the latest developments impacting the S&P 500:                                                                                                                               ┃
┃                                                                                                                                                                                       ┃
┃  1 S&P 500 Nears Correction Territory as Tariff Fears Loom                                                                                                                            ┃
┃     • Published: March 11, 2025                                                                                                                                                       ┃
┃     • Source: MarketWatch                                                                                                                                                             ┃
┃     • Key Points: Investor sentiment dims as the index approaches correction levels amid potential new tariffs by President Trump. Read more.                                         ┃
┃  2 Goldman Sachs Lowers S&P 500 Target Amid Skepticism                                                                                                                                ┃
┃     • Published: March 12, 2025                                                                                                                                                       ┃
┃     • Source: Bloomberg                                                                                                                                                               ┃
┃     • Key Points: The firm revises its target for the S&P 500 to 6,200 from 6,500 due to rising policy uncertainties and a bleak economic outlook. Read more.                         ┃
┃  3 Market Response to Tariff Announcement                                                                                                                                             ┃
┃     • Published: March 11, 2025                                                                                                                                                       ┃
┃     • Source: Forbes                                                                                                                                                                  ┃
┃     • Key Points: The Dow experienced a significant drop of nearly 500 points following tariff announcements, heightening concern among investors. Read more.                         ┃
┃  4 Goldman Sachs Cuts Target Again for S&P 500                                                                                                                                        ┃
┃     • Published: March 12, 2025                                                                                                                                                       ┃
┃     • Source: Reuters                                                                                                                                                                 ┃
┃     • Key Points: The financial institution reiterates its cautious stance on the S&P 500 by adjusting its year-end target in light of uncertain economic conditions. Read more.      ┃
┃  5 S&P 500 Futures Gain as Tech Sector Rallies                                                                                                                                        ┃
┃     • Published: March 12, 2025                                                                                                                                                       ┃
┃     • Source: MarketWatch                                                                                                                                                             ┃
┃     • Key Points: Futures for the S&P 500 are up 0.4%, buoyed by a rebound in the tech sector ahead of key inflation data. Read more.                                                 ┃
┃                                                                                                                                                                                       ┃
┃ ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── ┃
┃                                                                                   Market Sentiment                                                                                    ┃
┃                                                                                                                                                                                       ┃
┃ The current market environment is characterized by heightened caution due to geopolitical factors, particularly tariffs. Analyst downgrades are contributing to a bearish sentiment,  ┃
┃ yet there is a flicker of optimism as some sectors demonstrate resilience.                                                                                                            ┃
┃                                                                                                                                                                                       ┃
┃                                                                                     Key Takeaways                                                                                     ┃
┃                                                                                                                                                                                       ┃
┃  • The S&P 500 is close to correction territory amid tariff fears.                                                                                                                    ┃
┃  • Analyst recommendations indicate a tilt toward caution, with an increase in 'Hold' ratings.                                                                                        ┃
┃  • Market sentiment is cautious but showing some rebound potential in technology.                                                                                                     ┃
┃                                                                                                                                                                                       ┃
┃                                                                                     Risk Factors                                                                                      ┃
┃                                                                                                                                                                                       ┃
┃  • Political uncertainties surrounding trade policies.                                                                                                                                ┃
┃  • Potential for economic slowdown affecting corporate earnings.                                                                                                                      ┃
┃  • Market volatility driven by investor sentiment fluctuations.                                                                                                                       ┃
┃                                                                                                                                                                                       ┃
┃ ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── ┃
┃ Market Watch Team                                                                                                                                                                     ┃
┃ Date: March 12, 2025     
"""