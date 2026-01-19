from typing import Any, Dict, Set
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_tavily import TavilySearch
from src.workflow.state import GraphState

load_dotenv()
MAX_RETRIES=3
web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    
    documents = state.get("documents", [])
    unique_sources: Set[str] = state.get("unique_sources", set())
    
    # Reset generation to avoid stale bad answers
    state["generation"] = ""  
    
    # Increment retry_count for web search attempts
    retry_count = state.get("retry_count", 0) + 1
    
    if retry_count > MAX_RETRIES:
        print("---MAX RETRIES EXCEEDED IN WEB SEARCH---")
        return {
            "documents": documents,
            "question": question,
            "generation": "Unable to find a reliable answer after multiple attempts. Please refine your query.",
            "web_search": False,
            "web_sources": [],
            "unique_sources": unique_sources,
            "retry_count": retry_count
        }
    
    tavily_results = web_search_tool.invoke({"query": question})["results"]
    
    # Deduplicate by URL
    seen_urls = set(doc.metadata.get("source", "") for doc in documents)
    new_results = []
    new_sources = []
    
    for result in tavily_results:
        url = result["url"]
        if url not in seen_urls and url not in new_sources:
            new_results.append(result)
            new_sources.append(url)
            seen_urls.add(url)
            unique_sources.add(url)
    
    # Create individual Documents per result (as before)
    for result in new_results:
        doc = Document(
            page_content=result["content"],
            metadata={"source": result["url"], "title": result.get("title", "Web Result")}
        )
        documents.append(doc)
    
    return {
        "documents": documents,
        "question": question,
        "web_search": True,
        "web_sources": new_sources,
        "unique_sources": unique_sources,
        "retry_count": retry_count  # Updated retry_count
    }