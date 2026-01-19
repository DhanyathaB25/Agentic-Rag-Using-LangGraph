from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from src.models.model import get_openai_model

class BestAnswer(BaseModel):
    """Select the best answer from multiple options."""
    best_answer: str = Field(description="The selected best answer text")
    chosen_model: str = Field(description="Which model provided the best answer: 'gemini' or 'perplexity'")
    reasoning: str = Field(description="Brief explanation why this answer was chosen")

# Use GPT-4o-mini for validation
llm = get_openai_model()
structured_llm_validator = llm.with_structured_output(BestAnswer)

system = """You are an expert at evaluating answer quality. Compare two answers to the same question and select the best one.

Evaluation Criteria:
1. Accuracy and factual correctness based on the context
2. Relevance to the question
3. Clarity and conciseness (3 sentences maximum)
4. Completeness in addressing the question
5. Proper use of context from retrieved documents
6. Avoidance of hallucinations

Choose the answer that best meets these criteria. If both are equally good, choose the more concise one."""

validation_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", """Question: {question}

Context Documents:
{context}

Answer from Gemini:
{gemini_answer}

Answer from Perplexity:
{perplexity_answer}

Please select the best answer and provide your reasoning:""")
])

answer_validator: RunnableSequence = validation_prompt | structured_llm_validator