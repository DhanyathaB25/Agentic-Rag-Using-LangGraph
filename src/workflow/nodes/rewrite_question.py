from typing import Dict, Any
from src.models.model import get_llm_model
from langchain_core.prompts import ChatPromptTemplate

def rewrite_question(state: Dict[str, Any]) -> Dict[str, Any]:
    """Correct spelling and rewrite the user question using LLM only."""
    print("---REWRITE QUESTION---")
    llm = get_llm_model()
    original_question = state.get("question", "")

    print(f"---ORIGINAL QUESTION: {original_question}---")

    # Use prompt template for better structure
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that rewrites questions to be more clear and effective for retrieval.
        
Your tasks:
1. Correct any spelling mistakes
2. Make the question grammatically correct
3. Clarify ambiguous terms if needed
4. Keep the original meaning intact

Return ONLY the rewritten question, nothing else."""),

        ("human", "Original question: {question}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"question": original_question})
    rewritten = response.content.strip()
    
    print(f"---REWRITTEN QUESTION: {rewritten}---")

    # Store both original and rewritten questions
    state["original_question"] = original_question
    state["question"] = rewritten

    return state
