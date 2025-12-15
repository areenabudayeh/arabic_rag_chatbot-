import math
import re
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from sentence_transformers import util


# Exact Match 
def exact_match(pred, true):
    p = re.sub(r'[^\w\s]', '', pred.strip().lower())
    t = re.sub(r'[^\w\s]', '', true.strip().lower())

    if p == t or p in t or t in p:
        return 1.0
    return 0.0


# F1 
def binary_token_f1(preds, refs):
    total_tp = total_fp = total_fn = 0

    for pred, gold in zip(preds, refs):
        pred_tokens = set(word_tokenize(pred.lower()))
        gold_tokens = set(word_tokenize(gold.lower()))

        total_tp += len(pred_tokens & gold_tokens)
        total_fp += len(pred_tokens - gold_tokens)
        total_fn += len(gold_tokens - pred_tokens)

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)

    return 2 * precision * recall / (precision + recall + 1e-8)


# BLEU 
def compute_bleu(preds, refs):
    smooth = SmoothingFunction().method1

    return np.mean([
        sentence_bleu(
            [r.split()],
            p.split(),
            weights=(0.5, 0.5),
            smoothing_function=smooth
        )
        for p, r in zip(preds, refs)
    ])


# NDCG 
def ndcg(relevance, k):
    dcg = sum(
        (1 / math.log2(i + 2))
        for i, r in enumerate(relevance[:k])
        if r == 1
    )

    ideal = sorted(relevance, reverse=True)
    idcg = sum(
        (1 / math.log2(i + 2))
        for i, r in enumerate(ideal[:k])
        if r == 1
    )

    return dcg / idcg if idcg > 0 else 0.0


# Semantic Similarity 
def semantic_similarity(model, preds, refs):
    emb_pred = model.encode(
        preds,
        normalize_embeddings=True,
        convert_to_tensor=True
    )
    emb_ref = model.encode(
        refs,
        normalize_embeddings=True,
        convert_to_tensor=True
    )

    return util.cos_sim(emb_pred, emb_ref).diagonal().mean().item()


# Retrieval Metrics 
def precision_at_k(relevance, k):
    return sum(relevance) / k if k > 0 else 0.0


def recall_at_k(relevance):
    return 1.0 if any(relevance) else 0.0


def f1_at_k(precision, recall):
    return (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )


def average_precision(relevance):
    ap = 0.0
    hit_count = 0

    for i, rel in enumerate(relevance):
        if rel:
            hit_count += 1
            ap += hit_count / (i + 1)

    return ap / hit_count if hit_count > 0 else 0.0


def reciprocal_rank(relevance):
    for i, rel in enumerate(relevance):
        if rel:
            return 1 / (i + 1)
    return 0.0
