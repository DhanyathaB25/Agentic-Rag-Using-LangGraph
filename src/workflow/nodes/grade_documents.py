from typing import Any, Dict, Set
from src.workflow.chains.retrieval_grader import retrieval_grader
from src.workflow.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question.
    If no documents are relevant, sets a flag to run web search.
    Relies on retrieve node for deduplication.

    Args:
        state (GraphState): The current graph state

    Returns:
        Dict[str, Any]: Cleaned documents and updated web_search state
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    unique_sources: Set[str] = state.get("unique_sources", set())
    
    print(f"---GRADE: Initial unique_sources: {unique_sources}---")

    filtered_docs = []
    web_search = False
    
    for i, d in enumerate(documents):
        source = d.metadata.get("source", "Unknown Source")
        content = d.page_content.strip()
        
        if len(content) < 100 or not content:
            print(f"---GRADE: Skipping short/generic doc {i} from {source}---")
            web_search = True
            continue
        
        score = retrieval_grader.invoke(
            {"question": question, "document": content}
        )
        #print(f"---DOCUMENT {i} CONTENT: {content[:200]}... (Source: {source})---")
        print(f"---DOCUMENT {i} GRADE: {score.binary_score}---")
        
        if score.binary_score.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
            unique_sources.add(source)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
    
    if not filtered_docs:
        documents = []
        web_search = True
        print("---GRADE: No relevant docs; cleared documents list and triggering web search---")
    else:
        documents = filtered_docs[:5]  # Increased to 5 for more diverse sources
        print(f"---GRADE: Kept {len(documents)} relevant docs---")
    
    print(f"---FINAL WEB_SEARCH FLAG: {web_search}---")
    print(f"---FINAL unique_sources: {unique_sources}---")
    return {
        "documents": documents,
        "question": question,
        "web_search": web_search,
        "unique_sources": unique_sources,
        "retry_count": state.get("retry_count", 0)
    }