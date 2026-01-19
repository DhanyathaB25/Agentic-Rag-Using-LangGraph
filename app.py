import streamlit as st
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd

# Load environment
load_dotenv()

st.set_page_config(
    page_title="Adaptive RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
#                     CSS STYLES
# =========================================================
st.markdown(
    """
    <style>
      .main-header { font-size: 2.4rem; color: #1f77b4; text-align:center; font-weight:700; margin-bottom:10px;}
      .model-box { border:1px solid #e0e0e0; border-radius:8px; padding:12px; background:#fafafa; }
      .selected-model { border-color:#4CAF50; background:#f0f8f0; box-shadow: 0 4px 8px rgba(76,175,80,0.08); }
      .validation-box { border:1px solid #ff9800; border-radius:8px; padding:10px; background:#fff8ef; }
      .source-box { border:1px solid #d0d7de; border-radius:6px; padding:8px; background:#f5f7fa; margin-bottom:6px;}
      .metric-table { border-collapse: collapse; width:100%; }
      .metric-table th, .metric-table td { border: 1px solid #e6e6e6; padding:8px; text-align:center; }
      .metric-table th { background: #fafafa; }
      .winner-cell { background: #e6f4ea; font-weight:700; }
      .loser-cell { background: #fff6f6; }
      .neutral-cell { background:#ffffff; }
      .small { font-size:0.9rem; color:#666; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
#                     UTILITY FUNCTIONS
# =========================================================
def format_response(result: Dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return str(result)
    if "generation" in result:
        return result["generation"]
    if "answer" in result:
        return result["answer"]
    return json.dumps(result, indent=2)


# -------------------- EXTRACT METRICS --------------------
def extract_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts output from updated generate.py:
        result["validation_result"] = { winner, scores, metrics }
    Ensures table always displays correctly.
    """

    default = {
        "rouge1": None, "rouge2": None, "rougeL": None,
        "bleu": None, "meteor": None, "bert_f1": None,
        "cosine_sim": None, "faithfulness": None,
        "terminology_precision": None, "coverage": None
    }

    output = {
        "gemini_metrics": dict(default),
        "perplexity_metrics": dict(default),
        "gemini_score": None,
        "perplexity_score": None
    }

    # NEW: updated generate.py always puts metrics here
    vr = result.get("validation_result", None)
    if vr and isinstance(vr, dict):
        output["gemini_metrics"] = vr.get("gemini_metrics", default)
        output["perplexity_metrics"] = vr.get("perplexity_metrics", default)
        output["gemini_score"] = vr.get("gemini_score")
        output["perplexity_score"] = vr.get("perplexity_score")
        return output

    # Fallback (very unlikely)
    return output


# -------------------- METRIC TABLE DATAFRAME --------------------
def build_metrics_table(gem, perp, gem_score, perp_score):
    rows = [
        ("ROUGE-1", "rouge1"),
        ("ROUGE-2", "rouge2"),
        ("ROUGE-L", "rougeL"),
        ("BLEU", "bleu"),
        ("METEOR", "meteor"),
        ("BERT-F1", "bert_f1"),
        ("CosineSim", "cosine_sim"),
        ("Faithfulness", "faithfulness"),
        ("Terminology Precision", "terminology_precision"),
        ("Coverage", "coverage"),
    ]

    ranges = {
        "rouge1": "0.4–0.6 OK, >0.6 strong",
        "rouge2": "0.2–0.4 OK, >0.4 strong",
        "rougeL": "0.4–0.6 OK, >0.6 strong",
        "bleu": "0.2–0.4 OK, >0.4 strong",
        "meteor": "0.2–0.4 OK, >0.4 strong",
        "bert_f1": ">0.4 good, >0.6 strong",
        "cosine_sim": ">0.7 good",
        "faithfulness": ">0.6 good",
        "terminology_precision": ">0.6 good",
        "coverage": ">0.6 good",
    }

    table = []
    for label, key in rows:

        def fmt(x):
            try:
                return f"{float(x):.3f}"
            except:
                return "n/a"

        table.append({
            "Metric": label,
            "Gemini": fmt(gem.get(key)),
            "Perplexity": fmt(perp.get(key)),
            "Good Range": ranges[key]
        })

    table.append({
        "Metric": "Composite Score",
        "Gemini": f"{gem_score:.4f}" if gem_score else "n/a",
        "Perplexity": f"{perp_score:.4f}" if perp_score else "n/a",
        "Good Range": "Higher is better"
    })

    return pd.DataFrame(table)


# -------------------- RENDER HTML TABLE --------------------
def render_html_metrics_table(df, gem_score, perp_score, overall_winner):
    html = []
    html.append("<div class='model-box'><h4>📊 Metric Comparison</h4>")
    html.append("<table class='metric-table'>")
    html.append("<thead><tr><th>Metric</th><th>Gemini</th><th>Perplexity</th><th>Good Range</th></tr></thead><tbody>")

    for _, row in df.iterrows():

        def try_float(v):
            try:
                return float(v)
            except:
                return None

        g = try_float(row["Gemini"])
        p = try_float(row["Perplexity"])

        if g is None or p is None:
            gcls = pcls = "neutral-cell"
        else:
            if abs(g - p) < 1e-9:
                gcls = pcls = "neutral-cell"
            elif g > p:
                gcls = "winner-cell"
                pcls = "loser-cell"
            else:
                pcls = "winner-cell"
                gcls = "loser-cell"

        html.append(
            f"<tr>"
            f"<td>{row['Metric']}</td>"
            f"<td class='{gcls}'>{row['Gemini']}</td>"
            f"<td class='{pcls}'>{row['Perplexity']}</td>"
            f"<td class='small'>{row['Good Range']}</td>"
            f"</tr>"
        )

    html.append("</tbody></table>")

    if overall_winner:
        html.append(
            f"<div style='margin-top:8px; padding:5px; background:#eef7ff; border-radius:6px;'>"
            f"🏆 <b>Overall Winner:</b> {overall_winner.upper()}</div>"
        )

    html.append("</div>")
    return "\n".join(html)


# =========================================================
#               STREAMLIT SESSION STATE
# =========================================================
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "current_answer" not in st.session_state:
    st.session_state.current_answer = ""
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"user_session_{datetime.now().timestamp()}"

# =========================================================
#               UI LAYOUT
# =========================================================
st.markdown('<div class="main-header">🤖 Adaptive RAG System — Metrics Dashboard</div>', unsafe_allow_html=True)

left, right = st.columns([2, 1])

# =========================================================
#                       LEFT COLUMN
# =========================================================
with left:
    st.subheader("Ask a Question")
    question = st.text_area("Your question:", height=110)

    if st.button("🚀 Run Query", use_container_width=True):

        if not question.strip():
            st.error("Please enter a valid question.")
            st.stop()

        st.session_state.last_result = None
        st.session_state.current_question = question

        from src.workflow.graph import app

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        final_state = None

        try:
            for output in app.stream({"question": question}, config=config):
                for _, v in output.items():
                    final_state = v
        except Exception as e:
            st.error(f"Execution error: {e}")
            st.stop()

        st.session_state.last_result = final_state

        if final_state:
            ans = format_response(final_state)
            st.session_state.query_history.append({"question": question, "answer": ans})
            st.session_state.current_answer = ans

    st.markdown("---")

    # =========================================================
    #                     DISPLAY RESULTS
    # =========================================================
    if st.session_state.last_result:
        result = st.session_state.last_result

        st.markdown("### 🎯 Final Answer")
        model = result.get("chosen_model", "N/A")
        st.success(f"Selected Model: {model}")

        st.markdown(f"<div class='model-box selected-model'>{format_response(result)}</div>", unsafe_allow_html=True)

        # Validation reasoning
        if result.get("validation_reasoning"):
            st.markdown("### 🧠 Validation Reasoning")
            st.markdown(f"<div class='validation-box'>{result['validation_reasoning']}</div>", unsafe_allow_html=True)

        # Retrieved sources
        if result.get("documents"):
            st.markdown("### 📚 Retrieved Sources")
            unique = set()
            for d in result["documents"]:
                try:
                    src = d.metadata.get("source", "Unknown")
                except:
                    src = "Unknown"
                unique.add(src)

            for i, s in enumerate(sorted(unique), 1):
                st.markdown(f"<div class='source-box'>{i}. {s}</div>", unsafe_allow_html=True)

        # =========================================================
        #                METRIC EXTRACTION & DISPLAY
        # =========================================================
        st.markdown("### 📈 Metric Evaluation")

        metrics = extract_metrics(result)

        df = build_metrics_table(
            metrics["gemini_metrics"],
            metrics["perplexity_metrics"],
            metrics["gemini_score"],
            metrics["perplexity_score"]
        )

        winner = None
        if metrics["gemini_score"] is not None and metrics["perplexity_score"] is not None:
            winner = "gemini" if metrics["gemini_score"] >= metrics["perplexity_score"] else "perplexity"

        html = render_html_metrics_table(df, metrics["gemini_score"], metrics["perplexity_score"], winner)
        st.markdown(html, unsafe_allow_html=True)

        with st.expander("Raw Metric Data"):
            st.json(metrics)

# =========================================================
#                      RIGHT COLUMN
# =========================================================
with right:
    st.subheader("Parallel Model Outputs")

    if st.session_state.last_result:
        r = st.session_state.last_result

        st.markdown("#### 🔷 Gemini Output")
        st.markdown(f"<div class='model-box'>{r.get('gemini_answer', 'N/A')}</div>", unsafe_allow_html=True)

        st.markdown("#### 🔶 Perplexity Output")
        st.markdown(f"<div class='model-box'>{r.get('perplexity_answer', 'N/A')}</div>", unsafe_allow_html=True)
    else:
        st.info("Outputs will appear here after running a query.")



