from rag.embeddings.embeddings import EmbeddingGenerator
from rag.vector_store.qdrant_store import VectorDB
from config.settings_rag import rag_settings


class Retriever:
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db = VectorDB()
        self.similarity_threshold = rag_settings.SIMILARITY_THRESHOLD

    def retrieve_similar_context(self, query, top_k=None):
        """Retrieve similar contexts for a query"""

        if top_k is None:
            top_k = rag_settings.TOP_K_RETRIEVAL

        query_embedding = self.embedding_generator.generate_single_embedding(query)
        search_results = self.vector_db.client.query_points(
            collection_name=self.vector_db.collection_name,
            query=query_embedding,
            limit=top_k * 3
        )

        unique_contexts = {}
        for result in search_results.points:
            score = result.score
            
            if score < self.similarity_threshold:
                continue
            
            context = result.payload.get("context")
            if context is None:
                context = result.payload.get("text", "")

            if not context:
                continue

            answer = result.payload.get("answer_text", "")

            if context not in unique_contexts or score > unique_contexts[context]["score"]:
                unique_contexts[context] = {
                    "context": context,
                    "answer": answer,
                    "score": score,
                    "payload": result.payload
                }

        sorted_results = sorted(
            unique_contexts.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        top_results = sorted_results[:top_k]
        
        if not top_results:
            print("There are no similar paragraphs for this question.")
            return []

        return top_results
