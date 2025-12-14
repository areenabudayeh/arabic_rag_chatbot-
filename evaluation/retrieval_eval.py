from tqdm import tqdm
from rag.embeddings.embeddings import EmbeddingGenerator
from rag.retrieval.retrieval import Retriever
from config.settings_rag import rag_settings

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    average_precision,
    reciprocal_rank,
    ndcg,
    soft_recall,
    max_sbert_similarity
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

        precisions = []
        recalls = []
        f1s = []
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

            contexts = [r["context"] for r in retrieved]

            # Embeddings
            gold_emb = self.embedder.model.encode(
                row["context"],
                convert_to_tensor=True
            )

            ctx_embs = self.embedder.model.encode(
                contexts,
                convert_to_tensor=True
            )

            # Cosine similarities
            sims = (
                max_sbert_similarity(gold_emb, ctx_embs)
                if len(contexts) > 0
                else 0.0
            )

            all_sims = (
                self.embedder.model.similarity(
                    gold_emb, ctx_embs
                ).tolist()
                if len(contexts) > 0
                else []
            )

            relevance = [1 if s >= threshold else 0 for s in all_sims]

            # Metrics
            precision = precision_at_k(relevance, k)
            recall = recall_at_k(relevance)
            f1 = f1_at_k(precision, recall)

            ap = average_precision(relevance)
            rr = reciprocal_rank(relevance)
            ndcg_k = ndcg(relevance, k)
            soft = soft_recall(relevance)

            # Accumulate
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            average_precisions.append(ap)
            reciprocal_ranks.append(rr)
            ndcgs.append(ndcg_k)

            soft_hits += soft
            total_sbert_sim += sims

        return {
            "num_samples": total,
            "soft_recall@k": soft_hits / total,
            "avg_sbert_similarity": total_sbert_sim / total,
            "precision@k": sum(precisions) / total,
            "recall@k": sum(recalls) / total,
            "f1@k": sum(f1s) / total,
            "map": sum(average_precisions) / total,
            "mrr": sum(reciprocal_ranks) / total,
            "ndcg@k": sum(ndcgs) / total
        }
