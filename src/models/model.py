import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

def get_gemini_model():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY not set in environment or .env file")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=key,
        temperature=0
    )

def get_perplexity_model():
    key = os.getenv("PERPLEXITY_API_KEY")
    if not key:
        raise ValueError("PERPLEXITY_API_KEY not set in environment or .env file")
    return ChatOpenAI(
        model="sonar",
        base_url="https://api.perplexity.ai",
        api_key=key,
        temperature=0
    )

def get_openai_model():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set in environment or .env file")
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=key,
        temperature=0
    )

def get_llm_model():
    """Return the first available LLM based on environment variables"""
    if os.getenv("OPENAI_API_KEY"):
        return get_openai_model()
    elif os.getenv("GOOGLE_API_KEY"):
        return get_gemini_model()
    elif os.getenv("PERPLEXITY_API_KEY"):
        return get_perplexity_model()
    else:
        raise ValueError(
            "No API keys found for any LLM. "
            "Check your .env or environment variables."
        )

def get_embed_model():
    """Return embedding model"""
    return OpenAIEmbeddings(model="text-embedding-ada-002")

