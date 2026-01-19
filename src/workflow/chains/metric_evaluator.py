import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


embedder = SentenceTransformer("all-MiniLM-L6-v2")
smoothie = SmoothingFunction().method4

def normalize(text: str) -> str:
    return text.lower().strip()


def compute_all_metrics(pred: str, gt: str):
    pred_n = normalize(pred)
    gt_n = normalize(gt)

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    rouge = scorer.score(gt_n, pred_n)
    r1 = rouge["rouge1"].fmeasure
    r2 = rouge["rouge2"].fmeasure
    rL = rouge["rougeL"].fmeasure

    # BLEU
    bleu = sentence_bleu([gt_n.split()], pred_n.split(), smoothing_function=smoothie)

    # METEOR
    meteor = meteor_score([gt_n.split()], pred_n.split())

    # BERTScore
    P, R, F1 = bert_score([pred_n], [gt_n], lang='en', rescale_with_baseline=True)
    bert_f1 = float(F1[0])

    # Cosine similarity
    emb_pred = embedder.encode([pred_n])[0]
    emb_gt = embedder.encode([gt_n])[0]
    cos_sim = float(cosine_similarity([emb_pred], [emb_gt])[0][0])

    # Faithfulness & coverage
    pred_tokens = set(pred_n.split())
    gt_tokens = set(gt_n.split())
    common = pred_tokens & gt_tokens

    faithfulness = len(common) / len(gt_tokens) if gt_tokens else 0
    terminology_precision = len(common) / len(pred_tokens) if pred_tokens else 0
    coverage = len(common) / len(gt_tokens) if gt_tokens else 0

    return {
        "rouge1": r1,
        "rouge2": r2,
        "rougeL": rL,
        "bleu": bleu,
        "meteor": meteor,
        "bert_f1": bert_f1,
        "cosine_sim": cos_sim,
        "faithfulness": faithfulness,
        "terminology_precision": terminology_precision,
        "coverage": coverage
    }


def compute_final_score(metrics: dict) -> float:
    # Weighted composite score
    return (
        0.25 * metrics["bert_f1"] +
        0.20 * metrics["rougeL"] +
        0.15 * metrics["rouge2"] +
        0.10 * metrics["bleu"] +
        0.10 * metrics["meteor"] +
        0.10 * metrics["cosine_sim"]
    )


def evaluate_answers(gemini_ans, perplexity_ans, ground_truth):
    m_gemini = compute_all_metrics(gemini_ans, ground_truth)
    m_perp = compute_all_metrics(perplexity_ans, ground_truth)

    score_gemini = compute_final_score(m_gemini)
    score_perp = compute_final_score(m_perp)

    winner = "gemini" if score_gemini >= score_perp else "perplexity"

    return {
        "winner": winner,
        "gemini_score": score_gemini,
        "perplexity_score": score_perp,
        "gemini_metrics": m_gemini,
        "perplexity_metrics": m_perp
    }
