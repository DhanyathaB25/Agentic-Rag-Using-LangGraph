from typing import Any, Dict
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from src.workflow.chains.answer_grader import answer_grader
from src.workflow.chains.hallucination_grader import hallucination_grader
from src.workflow.chains.router import RouteQuery, question_router
from src.workflow.consts import REWRITE_QUESTION, GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from src.workflow.nodes.generate import generate
from src.workflow.nodes.grade_documents import grade_documents
from src.workflow.nodes.retrieve import retrieve
from src.workflow.nodes.web_search import web_search
from src.workflow.nodes.rewrite_question import rewrite_question
from src.workflow.state import GraphState

load_dotenv()

MAX_RETRIES = 3
checkpointer = MemorySaver()

def is_greeting(query: str) -> bool:
    greetings = [
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon", 
        "good evening", "howdy", "yo", "sup"
    ]
    query_lower = query.lower().strip()
    return any(greeting in query_lower for greeting in greetings)

def decide_to_generate(state):
    print(f"---ASSESS DOCUMENTS: web_search={state.get('web_search', False)}---")
    return WEBSEARCH if state.get("web_search", False) else GENERATE

def grade_generation_grounded_in_documents_and_question(state):
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state.get("documents", [])
    generation = state.get("generation", "")
    retry_count = state.get("retry_count", 0)

    print(f"---GENERATED ANSWER: {generation}---")
    
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    print(f"---HALLUCINATION SCORE: {score.binary_score}---")
    
    if retry_count >= MAX_RETRIES:
        print("---MAX RETRIES HIT: ACCEPT GENERATION AS-IS AND END---")
        return "useful"
    
    if score.binary_score:
        score = answer_grader.invoke({"question": question, "generation": generation})
        print(f"---ANSWER SCORE: {score.binary_score}---")
        if score.binary_score:
            print("---GRADE: USEFUL ANSWER---")
            return "useful"
        else:
            print("---GRADE: NOT USEFUL, RETRY VIA WEB---")
            state["retry_count"] = retry_count + 1
            return "not useful"
    else:
        retry_count += 1
        print(f"---HALLUCINATION FAILED: RETRY {retry_count}/{MAX_RETRIES}---")
        state["retry_count"] = retry_count
        return "not supported"

def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    
    # Initialize state fields if not present
    state.setdefault("messages", [])
    state.setdefault("documents", [])
    state.setdefault("web_search", False)
    state.setdefault("web_sources", [])
    state.setdefault("unique_sources", set())
    state.setdefault("retry_count", 0)
    
    if is_greeting(question):
        print("---ROUTED TO: GREETING (direct generation)---")
        state["generation"] = "Hello! How can I assist you with your query today? I'm here to help with AI, research, or anything else."
        return GENERATE
    
    source: RouteQuery = question_router.invoke({"question": question})
    print(f"---ROUTED TO: {source.datasource}---")
    return WEBSEARCH if source.datasource == "websearch" else RETRIEVE

# Build the workflow
workflow = StateGraph(GraphState)

# Add all nodes including the rewrite_question node
workflow.add_node(REWRITE_QUESTION, rewrite_question)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

# Set rewrite_question as the entry point
workflow.set_entry_point(REWRITE_QUESTION)

# After rewriting, route the question
workflow.add_conditional_edges(
    REWRITE_QUESTION,
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
        GENERATE: GENERATE,
    },
)

# Continue the rest of the workflow
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE},
)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)

app = workflow.compile(checkpointer=checkpointer)
