import gradio as gr
from openai import OpenAI

client = OpenAI()

def ask_ai(question):
    response = client.responses.create(
        model="gpt-4o",
        input=question
    )
    return response.output_text

demo = gr.Interface(
    fn=ask_ai,
    inputs="text",
    outputs="text",
    title="AI Assistant"
)

if __name__ == "__main__":
    demo.launch()