
#UPDATED TO RETURN METRICS
from dotenv import load_dotenv
from src.workflow.graph import app

load_dotenv()


# =============================================
# PRETTY OUTPUT FORMATTER
# =============================================
def pretty_print(result: dict):
    print("\n" + "=" * 60)
    print("                 ADAPTIVE RAG SYSTEM")
    print("=" * 60)

    # ---------------------------
    # QUERY & REWRITE
    # ---------------------------
    print("\n🔍 Query:")
    print(result.get("original_question", result.get("question", "N/A")))

    if "rewritten_question" in result:
        print("\n✏️ Rewritten Query:")
        print(result["rewritten_question"])

    # ---------------------------
    # RETRIEVED DOCUMENTS
    # ---------------------------
    documents = result.get("documents", [])
    if documents:
        print("\n📄 Retrieved Documents (Unique Sources)")
        print("-" * 60)
        unique_sources = set()

        for doc in documents:
            src = doc.metadata.get("source", "Unknown")
            unique_sources.add(src)

        for i, src in enumerate(sorted(unique_sources), start=1):
            print(f"{i}. {src}")

    # ---------------------------
    # MODEL ANSWERS
    # ---------------------------
    gem = result.get("gemini_answer", None)
    per = result.get("perplexity_answer", None)

    if gem or per:
        print("\n🤖 Model Outputs")
        print("-" * 60)

        if gem:
            print("Gemini:")
            print(gem[:300] + "..." if len(gem) > 300 else gem)
            print()

        if per:
            print("Perplexity:")
            print(per[:300] + "..." if len(per) > 300 else per)
            print()

    # ---------------------------
    # METRIC EVALUATION RESULT
    # ---------------------------
    val_res = result.get("validation_result", None)

    if val_res:
        print("\n📊 Evaluation Metrics (Model Comparison)")
        print("-" * 60)
        print(f"Winner: ⭐ {val_res['winner'].upper()} ⭐")
        print(f"Gemini Score:     {val_res['gemini_score']:.4f}")
        print(f"Perplexity Score: {val_res['perplexity_score']:.4f}")

        # Optional: show detailed metrics
        print("\n🔬 Detailed Gemini Metrics")
        for k, v in val_res["gemini_metrics"].items():
            print(f"{k}: {v:.4f}")

        print("\n🔬 Detailed Perplexity Metrics")
        for k, v in val_res["perplexity_metrics"].items():
            print(f"{k}: {v:.4f}")

    # ---------------------------
    # FINAL ANSWER
    # ---------------------------
    print("\n🏆 Final Answer")
    print("-" * 60)
    print(result.get("generation", "No final answer."))

    print("\n" + "=" * 60)
    print("                   OUTPUT COMPLETE")
    print("=" * 60 + "\n")


# =============================================
# CLI LOOP
# =============================================
def main():
    print("Adaptive RAG System")
    print("Type 'quit' to exit.\n")

    thread_id = "user_session_1"

    while True:
        try:
            question = input("Question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                break

            if not question:
                continue

            print("Processing...\n")

            config = {"configurable": {"thread_id": thread_id}}
            result = None

            for output in app.stream({"question": question}, config=config):
                for key, value in output.items():
                    result = value

            if result:
                pretty_print(result)
            else:
                print("⚠ No response generated.\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}\n")


if __name__ == "__main__":
    main()
