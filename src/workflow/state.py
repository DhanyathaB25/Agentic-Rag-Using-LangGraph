from typing import Annotated, Dict, List, TypedDict
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    """Represents the state of our graph."""
    original_question:str
    question: str
    generation: Annotated[str, ""]
    web_search: bool
    documents: List[BaseMessage]
    unique_sources: set
    web_sources: list
    retry_count: int
    messages: Annotated[List[BaseMessage], "List of conversation messages"]
    # New fields for multi-LLM
    gemini_answer: Annotated[str, ""]
    perplexity_answer: Annotated[str, ""]
    validation_reasoning: Annotated[str, ""]
    chosen_model: Annotated[str, ""]
    validation_result: dict 

