from tqdm import tqdm
from sentence_transformers import util

from rag.embeddings.embeddings import EmbeddingGenerator
from rag.retrieval.retrieval import Retriever
from config.settings_rag import rag_settings

from evaluation.metrics import (
    ndcg,
    precision_at_k,
    recall_at_k,
    f1_at_k,
    average_precision,
    reciprocal_rank
)


class RetrievalEvaluator:
    def __init__(self, df_val):
        self.df_val = df_val
        self.retriever = Retriever()
        self.embedder = EmbeddingGenerator()

    def evaluate(self):
        total = len(self.df_val)

        soft_hits = 0
        total_sbert_sim = 0.0

        retrieval_precisions = []
        retrieval_recalls = []
        retrieval_f1s = []
        average_precisions = []
        reciprocal_ranks = []
        ndcgs = []

        k = rag_settings.TOP_K_EVALUATION
        threshold = rag_settings.SIMILARITY_THRESHOLD_EVALUATION

        for _, row in tqdm(self.df_val.iterrows(), total=total):

            retrieved = self.retriever.retrieve_similar_context(
                row["question"],
                top_k=k
            )

            if not retrieved:
                retrieval_precisions.append(0.0)
                retrieval_recalls.append(0.0)
                retrieval_f1s.append(0.0)
                average_precisions.append(0.0)
                reciprocal_ranks.append(0.0)
                ndcgs.append(0.0)
                continue

            contexts = [r["context"] for r in retrieved]

            gold_emb = self.embedder.model.encode(
                row["context"],
                convert_to_tensor=True
            )
            ctx_emb = self.embedder.model.encode(
                contexts,
                convert_to_tensor=True
            )

            sims = util.cos_sim(gold_emb, ctx_emb)[0].tolist()
            relevance = [1 if s >= threshold else 0 for s in sims]

            # Soft Recall 
            if any(relevance):
                soft_hits += 1

            total_sbert_sim += max(sims) if sims else 0.0

            # Precision / Recall / F1 
            precision = precision_at_k(relevance, k)
            recall = recall_at_k(relevance)
            f1 = f1_at_k(precision, recall)

            retrieval_precisions.append(precision)
            retrieval_recalls.append(recall)
            retrieval_f1s.append(f1)

            # MAP 
            average_precisions.append(
                average_precision(relevance)
            )

            # MRR 
            reciprocal_ranks.append(
                reciprocal_rank(relevance)
            )

            # NDCG 
            ndcgs.append(
                ndcg(relevance, k)
            )

        return {
            "num_samples": total,
            "soft_recall@k": soft_hits / total,
            "avg_sbert_similarity": total_sbert_sim / total,
            "precision@k": sum(retrieval_precisions) / total,
            "recall@k": sum(retrieval_recalls) / total,
            "f1@k": sum(retrieval_f1s) / total,
            "map": sum(average_precisions) / total,
            "mrr": sum(reciprocal_ranks) / total,
            "ndcg@k": sum(ndcgs) / total
        }
