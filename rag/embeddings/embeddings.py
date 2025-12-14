from sentence_transformers import SentenceTransformer
from config.settings_rag import rag_settings

class EmbeddingGenerator:
    def __init__(self, model_name=rag_settings.EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, texts, batch_size=16):
        """Generate embeddings for a list of texts"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_tensor=True
        )
        return embeddings
    
    def generate_single_embedding(self, text):
        """Generate embedding for a single text (qustion)"""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_tensor=False
        ).astype("float32")
        return embedding
    
    