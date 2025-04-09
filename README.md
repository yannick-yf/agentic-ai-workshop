# Workshop (3h): Build Your First AI Agent with OpenAI and Agno

## Objective
Learn how to design, develop, and run an AI agent using OpenAI SDK and Agno/Phidata.

## Format
- Brief presentation  
- Live demonstrations  
- Hands-on activities  
- Q&A session  

## Required Materials:
- Laptops with Python 3.11+ installed  
- Stable internet access  
- IDE (recommended: VS Code)  
- Active OpenAI API key (participants should create one in advance if possible)  
- Installed Python libraries: See the Requirements.txt file  
- Access to the GitHub repository prepared for demos and practical exercises  

Important Note: This workshop provides a strong foundation for building AI agents. Participants are encouraged to explore the provided resources and continue experimenting to develop more advanced agents tailored to their specific needs.

---

## Introduction and Foundations of AI Agents

### 1.0 Defining Key Concepts: LLM, Token, RAG, etc.

#### Core Concepts of LLMs and AI Agents for Your Workshop

##### Large Language Models (LLMs)

A **Large Language Model (LLM)** is an AI system specialized in understanding and generating natural language through deep neural networks. These "large" models are trained on massive data corpora and have a large number of **parameters** (e.g., GPT-3 with 175 billion), enabling them to coherently generate text.

LLMs are **foundation models**, pre-trained on unlabeled data using self-supervised learning, allowing them to recognize patterns and perform various NLP tasks. They are built primarily on **Transformer neural network architectures**, which analyze word context within a sequence to better understand meaning.

The **training** of an LLM involves predicting the next word in a sentence, gradually adjusting its parameters to improve predictions. Once pre-trained, the model can be **fine-tuned** on a specific dataset to specialize in a given task.

###### Architecture and Functioning

LLMs consist of multiple neural network layers, including recurrent layers, feedforward layers, embedding layers, and attention layers that work together to process input text and generate coherent responses. This complex architecture allows them to produce text and interpret language with a fluency that mimics human speech.

##### Foundation Models

A foundation model is a large-scale AI model trained on massive amounts of unlabeled data, typically using self-supervised learning. These models can be adapted for a wide variety of downstream tasks without needing a complete rewrite.

Foundation models represent a paradigm shift in AI development—pre-trained on diverse data, they can be adapted to numerous applications. Early examples include large language models based on transformer architecture, like BERT and GPT-3.

###### Evolution and Characteristics

Foundation models are characterized by two key features: transfer learning and the ability to generate base knowledge that can be tailored for specific tasks. The term was popularized by Stanford’s Center for Research on Foundation Models (CRFM), part of the Human-Centered Artificial Intelligence (HAI) institute.

##### Tokens

Tokens are the fundamental text units processed by LLMs. A token can represent a full word, a word fragment, a character, or a punctuation symbol, depending on the model’s tokenizer. Tokenization breaks down the text into smaller units the model can understand.

LLMs usually have a context window defined in number of tokens (e.g., 2048, 4096, or 8192 tokens). This limit determines how much text the model can process at once, affecting its ability to remain coherent over long passages.

##### Retrieval Augmented Generation (RAG)

RAG (Retrieval Augmented Generation) is a technique that enhances LLMs by augmenting them with external knowledge, such as databases or documents. This method works by retrieving relevant documents from an external source and concatenating them with the original prompt before sending it to the language model.

This enables the LLM to access up-to-date information without needing retraining, which is particularly useful since the parametric knowledge in LLMs is static. RAG also helps reduce hallucinations (generating incorrect info) and improves performance in rapidly evolving environments.

##### Embeddings

Textual embeddings, also called word embeddings, are numerical representations of textual data where each word is represented as a real-valued vector. These vectorized representations help machine learning algorithms understand and efficiently process human language.

There are two main techniques for creating embeddings:
1. Frequency-based embeddings, which use word frequency to build vectors  
2. Prediction-based embeddings, which capture semantic relationships and context, providing rich conceptual representations

Embeddings enabled the rapid evolution of language models like RNNs, BERT, and GPT. They are fundamental for applications like text classification, information retrieval, and semantic similarity detection.

##### Prompt

A **prompt** is a textual or vocal instruction given to a generative AI to trigger specific content generation. It serves as the initial input that the model analyzes to produce coherent responses.

###### Key Features:
- **Flexible format**: Text (e.g., ChatGPT), voice (e.g., Siri/Alexa)  
- **Dual role**: Action trigger + contextual guide  
- **Common applications**:
  - Creative content generation (poems, scripts)
  - Technical problem-solving (coding, debugging)
  - Complex data analysis (medical, financial)

**Example structure**:  
"Write a blog post about the impact of RAG on LLMs in an educational tone, with 3 concrete case studies and recent statistics."

###### Prompt Engineering

**Prompt engineering** is the practice of optimizing prompt formulations to maximize the relevance of AI outputs. It blends linguistic skills with technical model understanding.

###### Key Techniques:
|| Technique | Goal | Example |
|---|---|---|---|
| **Context precision** | Add semantic tags | Improve contextual understanding | "[As a senior data scientist] Explain embeddings..." |
| **Structural constraints** | Define output formats | Standardize responses | "Generate a comparison table with columns X/Y/Z" |
| **Iterative optimization** | Refinement cycles | Fine-tune outputs | Version 1 → Review → Version 2 |

###### Advanced Applications:
- **Cross-model adaptation**: Tailoring prompts for GPT-4 (better at synthesis) vs Bard (real-time web access)
- **Hallucination reduction**: Built-in fact-checking ("Verify facts before answering")
- **Industrial automation**: Integrated into MLOps pipelines for data preprocessing

### 1.1 Understanding AI Agents

https://www.anthropic.com/engineering/building-effective-agents

#### What is an AI Agent? (Definition and Key Concepts)

The definition of an “agent” is complex and varies across sources. We can define agents as fully autonomous systems operating independently over long periods, using various tools to perform complex tasks. Alternatively, we can consider more prescriptive implementations following predefined workflows.

For this workshop, we will group all versions under the term **agentic systems**.

It's important to distinguish between **workflows** and **agents** architecturally:

- **Workflows** are systems where LLMs and tools are orchestrated along predefined code paths. Actions and tool usage are hardcoded in advance.
- **Agents**, on the other hand, dynamically control their process and tool usage, deciding in real-time based on reasoning and context.

An **AI Agent** is an entity that perceives its environment, reasons, and acts to achieve goals. It is designed to think, act, and use external tools or data autonomously and in real-time.

The relationship between AI Agents and **LLMs** is symbiotic: the LLM often acts as the agent’s brain. AI agents extend LLM capabilities by allowing them to interact with the outside world, overcoming the limits of static training data. For example, an LLM alone cannot check a live order status, but an agent can.

The core idea is that of an intelligent system capable of solving problems autonomously, using tools and relying on an LLM for dynamic decision-making (per Anthropic’s “agent” definition) or following predefined steps (per their “workflow” definition).

#### Core Components of an AI Agent

According to various sources, key components of an AI agent include:

- **Agent Core**: The decision engine, usually LLM-based, that interprets inputs and decides on actions. Uses prompts like *ReAct Prompt* to manage thoughts and actions.
- **Memory**: The agent’s ability to retain and use past interaction data. Crucial for conversation history and workflow building.
- **Tools (Actions)**: External functions or APIs the agent can use to interact with the real world or access data. Examples: web search, database queries, calculators.
- **Planning Module (if applicable)**: For more complex agents, the ability to break down tasks into steps and execute them sequentially.
- **Prompt System**: A set of core instructions that define the agent’s behavior, tone, and decision-making approach.

#### Why Use AI Agents?

AI agents offer many benefits across different contexts:

- **Advanced Task Automation**: Automating complex workflows beyond simple scripts  
- **Productivity Boost**: Delegating repetitive or research-heavy tasks  
- **Real-Time Data Access & Processing**: Querying up-to-date data for informed decision-making  
- **Smart Customer Support**: FAQs, order tracking, knowledge base interactions  
- **Business Workflow Automation**: Document analysis, report generation, event-based triggers  
- **Data Analysis & Research**: Querying databases, web search, summarizing information  
- **Advanced Personal Assistants**: Scheduling, managing personal information  
- **Multi-Agent Systems**: Collaboration between specialized agents to solve complex problems  

### 1.2 Tool Introduction: OpenAI SDK and Agno

#### What is OpenAI SDK?

The **OpenAI SDK** is a Python library that allows developers to interact with OpenAI APIs.

With the OpenAI SDK, you can:

- Configure your API key for authentication  
- Use `openai.chat.completions.create()` to send prompts to models and receive responses  
- Define *system prompts* to guide model behavior  
- Send *user prompts* to ask questions or give instructions  
- Parse API responses to extract generated content  
- Define and use external tools in the OpenAI function-calling format  

#### What is Agno?

**Agno** (formerly *Phidata*) is an open-source Python framework for building multimodal AI agents with memory, knowledge, and tools. It’s fast, simple, and model-agnostic.

Agno supports building agents that can work with text, images, audio, and video. It simplifies multi-agent orchestration and provides built-in support for memory, vector knowledge bases (*RAG*), and structured output generation.

#### Why Use Agno?

- **Fast and Simple**: Create agents in just three lines of code  
- **Model-Agnostic**: Compatible with OpenAI, Mistral, Anthropic, etc.  
- **Multimodal Support**: Native handling of text, images, audio, video  
- **Multi-Agent Support**: Enables collaboration between agents  
- **Advanced Features**: Built-in memory, knowledge management (RAG), structured outputs  
- **Code-as-AI**: Uses standard Python constructs (*if, else, while, for*)  
- **Easy Tool Integration**: Web search, finance, databases, etc.  
- **User Interface (UI)**: Agno provides a GUI to interact with agents  

In short, **Agno** is a powerful and flexible framework to develop intelligent and adaptive AI agents.

### 1.3 Define Your AI Agent
- **Interactive Workshop**: Define a simple use case  
  - Agent’s objective  
  - Expected inputs and outputs  
  - Required actions/tools  
- Examples: support chatbot, text analyzer, web searcher  

---

## Second Hour – Practical Agent Development

### 2.1 Environment Setup
- Dependency installation  
- Managing API keys with environment variables  
- Running a simple first test with the OpenAI SDK  

### 2.2 Interaction with OpenAI in Python
- Creating a basic conversational agent  
- Structuring prompts and the role of the system message  

### 2.3 Adding Tools to Your Agent with Agno
- Defining and registering actions in Agno  
- Integrating a simple action (e.g., weather lookup, web search)  
- Running and testing the agent with multiple inputs  
Sure! Here's the English translation of your French markdown while preserving the formatting:

---

## 3rd Hour – Improvement, Scalability, and Deployment

### 3.1 Optimization and Advanced Use Cases
- Multi-tool agents  
- RAG (Retrieval Augmented Generation)  
- Discussion on limitations and best practices

### 3.2 Hands-On Project: Create Your AI Agent
- Develop an agent in pairs or individually  
- Example tasks:
  - A chatbot assistant  
  - An intelligent text analyzer  
  - An automatic monitoring agent  
- Live support and debugging

### 3.3 Next Steps and Resources
- Explore advanced topics:
  - Adding memory for context management  
  - Using more complex tools (database connection, advanced text generation)  
  - Deployment in a cloud environment  
- Resources to go further: OpenAI and Agno documentation, online courses, project GitHub

### Citations and Sources Used for the Workshop
#### AI Tools:
- perplexity.ai  
- NotebookLM  
- Claude.ai  
- OpenAI  

#### Resources  
[1] https://www.data-bird.co/blog/llm-definition  
[2] https://fr.wikipedia.org/wiki/Mod%C3%A8le_de_fondation  
[3] https://www.promptingguide.ai/research/rag  
[4] https://bigblue.academy/en/text-embeddings  
[5] https://aws.amazon.com/fr/what-is/ai-agents/  
[6] https://www.dataleon.ai/blog/quest-ce-quun-large-language-model-llm  
[7] https://www.redhat.com/fr/topics/ai/what-are-foundation-models  
[8] https://cloud.google.com/use-cases/retrieval-augmented-generation  
[9] https://datascientest.com/le-word-embedding  
[10] https://www.ibm.com/fr-fr/think/topics/ai-agents  
[11] https://www.cloudflare.com/fr-fr/learning/ai/what-is-large-language-model/  
[12] https://www.lebigdata.fr/modele-de-fondation-une-notion-fondamentale-en-intelligence-artificielle  
[13] https://aws.amazon.com/what-is/retrieval-augmented-generation/  
[14] https://infoscience.epfl.ch/record/221430/?v=%5B%27pdf%27%5D  
[15] https://fr.wikipedia.org/wiki/Agent_intelligent  
[16] https://www.hpe.com/ch/fr/what-is/large-language-model.html  
[17] https://aws.amazon.com/fr/what-is/foundation-models/  
[18] https://www.redhat.com/fr/topics/ai/what-is-retrieval-augmented-generation  
[19] https://www.datacamp.com/fr/blog/what-is-text-embedding-ai  
[20] https://botpress.com/fr/blog/what-is-an-ai-agent  
[21] https://datascientest.com/prompt-tout-savoir  
[22] https://www.ibm.com/fr-fr/think/topics/prompt-engineering  
[23] https://www.lebigdata.fr/prompt-definition  
[24] https://datascientest.com/prompt-engineer-tout-savoir  
[25] https://www.intelligence-artificielle-school.com/ecole/technologies/quest-ce-quun-prompt-en-ia/  
[26] https://platform.openai.com/docs/guides/prompt-engineering  
[27] https://www.lacreme.ai/post/quest-ce-quun-prompt-definition-intelligence-artificielle  
[28] https://en.wikipedia.org/wiki/Prompt_engineering  
[29] https://www.orientaction-groupe.com/vocabulaire-ia-prompt/