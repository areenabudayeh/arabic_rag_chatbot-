import time
import random
import numpy as np
from evaluate import load

from rag.data.text_cleaning import normalize_arabic
from rag.embeddings.embeddings import EmbeddingGenerator
from config.settings_rag import rag_settings

from evaluation.metrics import (
    exact_match,
    binary_token_f1,
    compute_bleu,
    semantic_similarity
)

class GenerationEvaluator:
    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.bleu_metric = load("bleu")

    def safe_generate(self, generate_func, question):
        for attempt in range(rag_settings.MAX_RETRIES):
            try:
                return normalize_arabic(generate_func(question))
            except Exception as e:
                delay = rag_settings.BASE_DELAY * (2 ** attempt)
                time.sleep(delay + random.uniform(0, 2))
        return "ERROR"

    def evaluate(self, df_val, generators_dict):
        df_subset = df_val.head(rag_settings.EVAL_SUBSET_SIZE)
        refs = [normalize_arabic(r) for r in df_subset["answer_text"].tolist()]

        all_preds = {k: [] for k in generators_dict}

        for _, row in df_subset.iterrows():
            for name, gen in generators_dict.items():
                all_preds[name].append(self.safe_generate(gen, row["question"]))
            time.sleep(3)

        results = {}
        for name, preds in all_preds.items():
            results[name] = {
                "bleu2": compute_bleu(preds, refs),
                "f1": binary_token_f1(preds, refs),
                "em": np.mean([exact_match(p, r) for p, r in zip(preds, refs)]),
                "semantic_sim": semantic_similarity(
                    self.embedder.model, preds, refs
                )
            }

        return results
