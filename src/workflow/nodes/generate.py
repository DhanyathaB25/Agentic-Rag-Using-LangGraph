
#UPDATED TO RETURN METRICS

from typing import Any, Dict
from langchain_core.messages import HumanMessage, AIMessage

from src.workflow.chains.generation import gemini_generation_chain, perplexity_generation_chain
from src.workflow.state import GraphState
from src.workflow.chains.metric_evaluator import evaluate_answers
from src.workflow.ground_truths import GROUND_TRUTHS


def generate(state: GraphState) -> Dict[str, Any]:
    """Generate answers using both Gemini and Perplexity, then evaluate using metrics."""
    
    print("---GENERATE WITH MULTI-LLM + METRIC EVALUATION---")

    question = state["question"]
    original_question = state.get("original_question", question)
    documents = state.get("documents", [])
    messages = state.get("messages", [])
    retry_count = state.get("retry_count", 0)

    # Retry safety
    if retry_count > 3:
        fallback = "I couldn't generate a reliable answer after multiple attempts. Please try rephrasing your question."
        return {
            "documents": documents,
            "question": question,
            "generation": fallback,
            "messages": messages,
            "chosen_model": "fallback",
            "validation_reasoning": "Retry limit exceeded",
            "validation_result": None
        }

    # Build context
    history = "\n".join([f"{m.type}: {m.content}" for m in messages[-10:]])
    context = "\n\n".join([doc.page_content for doc in documents])

    # ----------------------------
    # 1️⃣ GENERATE FROM GEMINI
    # ----------------------------
    try:
        gemini_answer = gemini_generation_chain.invoke({
            "context": context,
            "question": question,
            "history": history
        })
    except Exception as e:
        gemini_answer = f"Error: {str(e)}"

    # ----------------------------
    # 2️⃣ GENERATE FROM PERPLEXITY
    # ----------------------------
    try:
        perplexity_answer = perplexity_generation_chain.invoke({
            "context": context,
            "question": question,
            "history": history
        })
    except Exception as e:
        perplexity_answer = f"Error: {str(e)}"

    # ----------------------------
    # 3️⃣ METRIC VALIDATION
    # ----------------------------

    # Normalize lookup
    q_norm = original_question.lower().strip()
    ground_truth = None

    for key, val in GROUND_TRUTHS.items():
        if key in q_norm:
            ground_truth = val
            break

    eval_result = None  # <-- NEW

    if ground_truth is None:
        chosen_model = "perplexity"
        generation = perplexity_answer
        validation_reasoning = "No ground truth available for metric evaluation."

    else:
        # Both crashed
        if "Error:" in gemini_answer and "Error:" in perplexity_answer:
            chosen_model = "error"
            generation = "Both models failed. Try again."
            validation_reasoning = "Both Gemini and Perplexity errored."

        elif "Error:" in gemini_answer:
            chosen_model = "perplexity"
            generation = perplexity_answer
            validation_reasoning = "Gemini failed to generate."

        elif "Error:" in perplexity_answer:
            chosen_model = "gemini"
            generation = gemini_answer
            validation_reasoning = "Perplexity failed to generate."

        else:
            # RUN METRIC EVALUATOR
            eval_result = evaluate_answers(
                gemini_answer,
                perplexity_answer,
                ground_truth
            )

            chosen_model = eval_result["winner"]
            generation = gemini_answer if chosen_model == "gemini" else perplexity_answer
            validation_reasoning = "Selected best model based on metric evaluation."

    # ----------------------------
    # 4️⃣ UPDATE CHAT HISTORY
    # ----------------------------
    messages.append(HumanMessage(content=question))
    messages.append(AIMessage(content=generation))

    # ----------------------------
    # 5️⃣ RETURN STATE (NOW WITH SCORES)
    # ----------------------------
    return {
        "documents": documents,
        "question": question,
        "original_question": original_question,
        "generation": generation,
        "messages": messages,
        "gemini_answer": gemini_answer,
        "perplexity_answer": perplexity_answer,
        "chosen_model": chosen_model,
        "validation_reasoning": validation_reasoning,
        "validation_result": eval_result,   # <-- THE METRIC SCORES ARE RETURNED HERE
    }
