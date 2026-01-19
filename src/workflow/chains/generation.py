from langchain_core.output_parsers import StrOutputParser
from src.models.model import get_gemini_model, get_perplexity_model, get_llm_model
from langchain_core.prompts import ChatPromptTemplate

# Create separate chains for each model
gemini_llm = get_gemini_model()
perplexity_llm = get_perplexity_model()

system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, say that you don't know. 
Use three sentences maximum and keep the answer concise.
Conversation History: {history}
Question: {question}
Context: {context}
Answer:"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])

# Create separate generation chains
gemini_generation_chain = prompt | gemini_llm | StrOutputParser()
perplexity_generation_chain = prompt | perplexity_llm | StrOutputParser()

# Keep original for backward compatibility
original_llm = get_llm_model()
generation_chain = prompt | original_llm | StrOutputParser()