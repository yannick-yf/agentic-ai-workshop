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

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import dedent
from io import StringIO
import base64
from typing import Dict, List, Optional, Union, Any
import tempfile

import typer
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.agent.sqlite import SqliteAgentStorage
from rich import print

agent_storage = SqliteAgentStorage(table_name="analyst_agent", db_file="tmp/agents.db")

class DataExplorer:
    """Data Explorer class that handles data loading and operations."""
    
    def __init__(self):
        self.data = None
        self.file_path = None
        self.file_type = None
    
    def load_data(self, file_path: str) -> str:
        """
        Load data from a CSV or JSON file.
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            str: Message indicating success or failure
        """
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' not found."
        
        self.file_path = file_path
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                self.data = pd.read_csv(file_path)
                self.file_type = 'csv'
            elif file_extension == '.json':
                self.data = pd.read_json(file_path)
                self.file_type = 'json'
            else:
                return f"Error: Unsupported file format '{file_extension}'. Please use CSV or JSON files."
            
            return f"Successfully loaded data from '{file_path}'. {len(self.data)} rows and {len(self.data.columns)} columns found."
        
        except Exception as e:
            return f"Error loading data: {str(e)}"
    
    def get_data_info(self) -> str:
        """
        Get basic information about the loaded data.
        
        Returns:
            str: JSON string with data information
        """
        if self.data is None:
            return "No data loaded. Please load a data file first."
        
        info = {
            "file_path": self.file_path,
            "file_type": self.file_type,
            "rows": len(self.data),
            "columns": len(self.data.columns),
            "column_names": self.data.columns.tolist(),
            "data_types": {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            "sample": self.data.head(5).to_dict(orient='records')
        }
        
        return json.dumps(info, indent=2)
    
    def get_data_summary(self) -> str:
        """
        Get a statistical summary of the loaded data.
        
        Returns:
            str: Summary statistics as a string
        """
        if self.data is None:
            return "No data loaded. Please load a data file first."
        
        # Capture the summary as a string
        buffer = StringIO()
        self.data.describe(include='all').to_string(buffer)
        summary = buffer.getvalue()
        
        return summary
    
    def run_query(self, query: str) -> str:
        """
        Run a pandas query on the data.
        
        Args:
            query (str): Pandas query string
            
        Returns:
            str: JSON string with query results
        """
        if self.data is None:
            return "No data loaded. Please load a data file first."
        
        try:
            # Simple string replacement for common natural language terms
            query = query.replace("greater than", ">")
            query = query.replace("less than", "<")
            query = query.replace("equal to", "==")
            query = query.replace("equals", "==")
            
            result = self.data.query(query)
            return json.dumps({
                "rows_returned": len(result),
                "data": result.head(20).to_dict(orient='records')
            }, indent=2)
        
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    def generate_visualization(self, vis_type: str, x_column: str, y_column: Optional[str] = None, 
                              hue: Optional[str] = None, title: Optional[str] = None) -> str:
        """
        Generate a visualization based on the specified parameters.
        
        Args:
            vis_type (str): Type of visualization (line, bar, scatter, heatmap, etc.)
            x_column (str): Column to use for x-axis
            y_column (str, optional): Column to use for y-axis
            hue (str, optional): Column to use for color grouping
            title (str, optional): Title for the visualization
            
        Returns:
            str: Base64 encoded image
        """
        if self.data is None:
            return "No data loaded. Please load a data file first."
        
        if x_column not in self.data.columns:
            return f"Error: Column '{x_column}' not found in data."
        
        if y_column and y_column not in self.data.columns:
            return f"Error: Column '{y_column}' not found in data."
        
        if hue and hue not in self.data.columns:
            return f"Error: Column '{hue}' not found in data."
        
        plt.figure(figsize=(10, 6))
        
        try:
            if vis_type.lower() == 'line':
                if y_column:
                    if hue:
                        sns.lineplot(x=x_column, y=y_column, hue=hue, data=self.data)
                    else:
                        sns.lineplot(x=x_column, y=y_column, data=self.data)
                else:
                    return "Error: Y-column is required for line plots."
            
            elif vis_type.lower() == 'bar':
                if y_column:
                    if hue:
                        sns.barplot(x=x_column, y=y_column, hue=hue, data=self.data)
                    else:
                        sns.barplot(x=x_column, y=y_column, data=self.data)
                else:
                    # Count plot if no y_column provided
                    if hue:
                        sns.countplot(x=x_column, hue=hue, data=self.data)
                    else:
                        sns.countplot(x=x_column, data=self.data)
            
            elif vis_type.lower() == 'scatter':
                if y_column:
                    if hue:
                        sns.scatterplot(x=x_column, y=y_column, hue=hue, data=self.data)
                    else:
                        sns.scatterplot(x=x_column, y=y_column, data=self.data)
                else:
                    return "Error: Y-column is required for scatter plots."
            
            elif vis_type.lower() == 'histogram':
                sns.histplot(data=self.data, x=x_column, hue=hue)
            
            elif vis_type.lower() == 'heatmap':
                if x_column and y_column:
                    # Create pivot table for heatmap
                    pivot_data = self.data.pivot_table(
                        index=y_column, 
                        columns=x_column, 
                        aggfunc='size', 
                        fill_value=0
                    )
                    sns.heatmap(pivot_data, annot=True, cmap='viridis')
                else:
                    # If only correlation heatmap is needed
                    numeric_data = self.data.select_dtypes(include=['number'])
                    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
            
            elif vis_type.lower() == 'box':
                if y_column:
                    sns.boxplot(x=x_column, y=y_column, data=self.data)
                else:
                    sns.boxplot(x=x_column, data=self.data)
            
            elif vis_type.lower() == 'violin':
                if y_column:
                    sns.violinplot(x=x_column, y=y_column, data=self.data)
                else:
                    sns.violinplot(x=x_column, data=self.data)
            
            else:
                return f"Error: Unsupported visualization type '{vis_type}'."
            
            # Set title if provided
            if title:
                plt.title(title)
            else:
                plt.title(f"{vis_type.capitalize()} plot of {x_column}" + (f" vs {y_column}" if y_column else ""))
            
            plt.tight_layout()
            
            # Save figure to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                plt.savefig(temp_file.name)
                temp_file_path = temp_file.name
            
            # Read the saved image and encode it to base64
            with open(temp_file_path, 'rb') as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Clean up
            plt.close()
            os.remove(temp_file_path)
            
            return f"data:image/png;base64,{encoded_image}"
        
        except Exception as e:
            plt.close()
            return f"Error generating visualization: {str(e)}"


# Create an instance of the DataExplorer
data_explorer = DataExplorer()

# Define tool functions that the agent can use
def load_data_file(file_path: str) -> str:
    """Load data from a CSV or JSON file.
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        str: Message indicating success or failure
    """
    return data_explorer.load_data(file_path)

def get_data_info() -> str:
    """Get basic information about the loaded data.
    
    Returns:
        str: JSON string with data information
    """
    return data_explorer.get_data_info()

def get_data_summary() -> str:
    """Get a statistical summary of the loaded data.
    
    Returns:
        str: Summary statistics as a string
    """
    return data_explorer.get_data_summary()

def run_data_query(query: str) -> str:
    """Run a pandas query on the data.
    
    Args:
        query (str): Pandas query string
        
    Returns:
        str: JSON string with query results
    """
    return data_explorer.run_query(query)

def create_visualization(
    vis_type: str, 
    x_column: str, 
    y_column: Optional[str] = None, 
    hue: Optional[str] = None, 
    title: Optional[str] = None) -> str:
    """Generate a visualization based on the specified parameters.
    
    Args:
        vis_type (str): Type of visualization (line, bar, scatter, heatmap, etc.)
        x_column (str): Column to use for x-axis
        y_column (str, optional): Column to use for y-axis
        hue (str, optional): Column to use for color grouping
        title (str, optional): Title for the visualization
        
    Returns:
        str: Base64 encoded image that can be displayed
    """
    return data_explorer.generate_visualization(vis_type, x_column, y_column, hue, title)

def analyst_agent(user: str = "user"):
    session_id: Optional[str] = None

    # Ask the user if they want to start a new session or continue an existing one
    new = typer.confirm("Do you want to start a new session?")

    if not new:
        existing_sessions: List[str] = agent_storage.get_all_session_ids(user)
        if len(existing_sessions) > 0:
            session_id = existing_sessions[0]
                
    # Create the Interactive Data Explorer Agent
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=dedent("""\
            You are an Interactive Data Explorer Agent, a specialized assistant designed to help users explore and visualize datasets. 📊
            
            Your capabilities:
            - Loading data from CSV and JSON files
            - Providing information and summaries about datasets
            - Executing data queries based on user questions
            - Creating appropriate visualizations to help users understand their data
            
            Your communication style:
            - Explain your thought process when interpreting the user's request
            - Suggest appropriate visualizations or analyses based on the data
            - Provide insights about the data when you spot interesting patterns
            - Use clear, concise language with data science terminology where appropriate
            
            When creating visualizations:
            - Choose the appropriate chart type based on the data and question
            - Explain why you selected a particular visualization
            - Describe what the visualization shows
            - Suggest follow-up analyses that might be interesting
            
            Always help the user load their data first before attempting any analysis or visualization.
            Be precise in your explanations and interpretations of the data.
        """),
        tools=[
            load_data_file,
            get_data_info,
            get_data_summary,
            run_data_query,
            create_visualization
        ],
        storage=agent_storage,
        # Show tool calls in the response
        show_tool_calls=True,
        # To provide the agent with the chat history
        # We can either:
        # 1. Provide the agent with a tool to read the chat history
        # 2. Automatically add the chat history to the messages sent to the model
        #
        # 1. Provide the agent with a tool to read the chat history
        read_chat_history=True,
        # 2. Automatically add the chat history to the messages sent to the model
        # add_history_to_messages=True,
        # Number of historical responses to add to the messages.
        # num_history_responses=3,
        markdown=True,
    )

    print("You are about to chat with an agent!")
    if session_id is None:
        session_id = agent.session_id
        if session_id is not None:
            print(f"Started Session: {session_id}\n")
        else:
            print("Started Session\n")
    else:
        print(f"Continuing Session: {session_id}\n")

    # Runs the agent as a command line application
    agent.cli_app(markdown=True)


if __name__ == "__main__":
    typer.run(analyst_agent)

# # Example usage
# if __name__ == "__main__":
#     # # Example queries to try:
#     # print("Interactive Data Explorer Agent is ready!")
#     # print("Try these example queries after loading a data file:")
#     # print("- 'Load the sales_data.csv file'")
#     # print("- 'What columns are in this dataset?'")
#     # print("- 'Show me trends in sales over the last year'")
#     # print("- 'Create a bar chart of product categories by revenue'")
#     # print("- 'What's the correlation between customer age and purchase amount?'")
    
#     # Start with a data loading prompt
#     agent.print_response("I have a sales_data.csv file. Can you help me explore it? The file path is './agentic_ai_workshop/getting_started/sales_data.csv'", stream=True)