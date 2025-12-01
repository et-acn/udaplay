# %% [markdown]
# # [STARTER] Udaplay Project

# %% [markdown]
# ## Part 01 - Offline RAG
# 
# In this part of the project, you'll build your VectorDB using Chroma.
# 
# The data is inside folder `project/starter/games`. Each file will become a document in the collection you'll create.
# Example.:
# ```json
# {
#   "Name": "Gran Turismo",
#   "Platform": "PlayStation 1",
#   "Genre": "Racing",
#   "Publisher": "Sony Computer Entertainment",
#   "Description": "A realistic racing simulator featuring a wide array of cars and tracks, setting a new standard for the genre.",
#   "YearOfRelease": 1997
# }
# ```
# 

# %% [markdown]
# ### Setup

# %%
# Only needed for Udacity workspace

import importlib.util
import sys

# Check if 'pysqlite3' is available before importing
if importlib.util.find_spec("pysqlite3") is not None:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# %%
import os
import json
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# %%
# TODO: Create a .env file with the following variables
# OPENAI_API_KEY="YOUR_KEY"
# CHROMA_OPENAI_API_KEY="YOUR_KEY"
# TAVILY_API_KEY="YOUR_KEY"

# %%
# TODO: Load environment variables
load_dotenv()
assert os.getenv('OPENAI_API_KEY')
assert os.getenv('CHROMA_OPENAI_API_KEY')
assert os.getenv('TAVILY_API_KEY')

# %% [markdown]
# ### VectorDB Instance

# %%
# TODO: Instantiate your ChromaDB Client
# Choose any path you want
chroma_client = chromadb.PersistentClient(path="chromadb")

# %% [markdown]
# ### Collection

# %%
# TODO: Pick one embedding function
# If picking something different than openai, 
# make sure you use the same when loading it
# embedding_fn = embedding_functions.OpenAIEmbeddingFunction()

# %%
# TODO: Create a collection
# Choose any name you want
collectionName = "udaplay11"
collection = chroma_client.create_collection(
    name=collectionName
#    embedding_function=embedding_fn
)

# %% [markdown]
# ### Add documents

# %%
# Make sure you have a directory "project/starter/games"
data_dir = "games"

for file_name in sorted(os.listdir(data_dir)):
    if not file_name.endswith(".json"):
        continue

    file_path = os.path.join(data_dir, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        game = json.load(f)

    # You can change what text you want to index
    content = f"[{game['Platform']}] {game['Name']} ({game['YearOfRelease']}) - {game['Description']}"

    # Use file name (like 001) as ID
    doc_id = os.path.splitext(file_name)[0]

    collection.add(
        ids=[doc_id],
        documents=[content],
        metadatas=[game]
    )

# %% [markdown]
# # [STARTER] Udaplay Project

# %% [markdown]
# ## Part 02 - Agent
# 
# In this part of the project, you'll use your VectorDB to be part of your Agent as a tool.
# 
# You're building UdaPlay, an AI Research Agent for the video game industry. The agent will:
# 1. Answer questions using internal knowledge (RAG)
# 2. Search the web when needed
# 3. Maintain conversation state
# 4. Return structured outputs
# 5. Store useful information for future use

# %% [markdown]
# ### Setup

# %%
# Only needed for Udacity workspace

import importlib.util
import sys

# Check if 'pysqlite3' is available before importing
if importlib.util.find_spec("pysqlite3") is not None:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# %%
# TODO: Import the necessary libs
# For example: 
import os

from lib.agents import Agent
from lib.llm import LLM
from lib.messages import UserMessage, SystemMessage, ToolMessage, AIMessage
from lib.tooling import tool

# %%
# TODO: Load environment variables
load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# %%


# %% [markdown]
# ### Tools

# %% [markdown]
# Build at least 3 tools:
# - retrieve_game: To search the vector DB
# - evaluate_retrieval: To assess the retrieval performance
# - game_web_search: If no good, search the web
# 

# %% [markdown]
# #### Retrieve Game Tool

# %%
# TODO: Create retrieve_game tool
# It should use chroma client and collection you created
# chroma_client = chromadb.PersistentClient(path="chromadb")
# collection = chroma_client.get_collection(collectionName)
def retrieve_game(query: str) -> list:
    """
    Semantic search: Finds most results in the vector DB.

    args:
    - query: a question about the game industry.

    You'll receive results as a list. Each element contains:
    - Platform: like Game Boy, Playstation 5, Xbox 360...
    - Name: Name of the Game
    - YearOfRelease: Year when that game was released for that platform
    - Description: Additional details about the game
    """
    # Perform a semantic search in the vector database
    return collection.query(
       query_texts=[query],
       n_results=10,
       include=['metadatas', 'documents']
       )

# %% [markdown]
# #### Evaluate Retrieval Tool

# %%
# TODO: Create evaluate_retrieval tool
# You might use an LLM as judge in this tool to evaluate the performance
# You need to prompt that LLM with something like:
# "Your task is to evaluate if the documents are enough to respond the query. "
# "Give a detailed explanation, so it's possible to take an action to accept it or not."
# Use EvaluationReport to parse the result
# Tool Docstring:
#    Based on the user's question and on the list of retrieved documents, 
#    it will analyze the usability of the documents to respond to that question. 
#    args: 
#    - question: original question from user
#    - retrieved_docs: retrieved documents most similar to the user query in the Vector Database
#    The result includes:
#    - useful: whether the documents are useful to answer the question
#    - description: description about the evaluation result

from lib.messages import UserMessage, SystemMessage
from lib.tooling import tool
from lib.llm import LLM
from lib.parsers import (
    StrOutputParser,
    JsonOutputParser, 
    PydanticOutputParser, 
    ToolOutputParser,
)

@tool
def format_evaluation(useful: bool, description: str):
    return {
        'useful': useful,
        'description': description
    }

def evaluate_retrieval(query: str, documents: list):
    chat_model = LLM(api_key=os.getenv('OPENAI_API_KEY'), tools=[format_evaluation])
    messages = [
        SystemMessage(content="""
        You're vector DB retrieval evaluator.
        Your task is to evaluate if the documents are enough to respond the query.
        Give a detailed explanation, so it's possible to take an action to accept it or not.
        Format your response by calling the format_evaluation tool.
        """),
        UserMessage(content="Query: " + query + ". " + "Documents: " + '\n'.join(documents))
    ]
    response = chat_model.invoke(messages)
    return response.tool_calls[0].function.arguments

# %% [markdown]
# #### Game Web Search Tool

# %%
# TODO: Create game_web_search tool
# Please use Tavily client to search the web
# Tool Docstring:
#    Semantic search: Finds most results in the vector DB
#    args:
#    - question: a question about game industry. 
from tavily import TavilyClient
def game_web_search(query: str):
    tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
    response = tavily_client.search(query)
    snippets = [result['content'] for result in response.get('results', [])]
    return '\n'.join(snippets)

# %% [markdown]
# ### Agent

# %%
# TODO: Create your Agent abstraction using StateMachine
# Equip with an appropriate model
# Craft a good set of instructions 
# Plug all Tools you developed

# %%
# TODO: Invoke your agent
# - When Pokémon Gold and Silver was released?
# - Which one was the first 3D platformer Mario game?
# - Was Mortal Kombat X realeased for Playstation 5?

# %% [markdown]
# ### (Optional) Advanced

# %%
# TODO: Update your agent with long-term memory
# TODO: Convert the agent to be a state machine, with the tools being pre-defined nodes



