import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
import chromadb.utils.embedding_functions as embedding_functions
import openai

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding_function():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",  # or "text-embedding-3-large" for better quality
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
