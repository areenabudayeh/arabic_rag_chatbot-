from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from config.settings_rag import rag_settings


class VectorDB:
    def __init__(self):
        self.client = QdrantClient(
            host=rag_settings.QDRANT_HOST,
            port=rag_settings.QDRANT_PORT
        )
        self.collection_name = rag_settings.COLLECTION_NAME
        self.initialized = False

    def create_collection(self, vector_size):
        """Create a new Qdrant collection"""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        self.initialized = True

    def insert_train_samples(self, samples, embeddings):
        """Insert training QA samples into the vector database"""
        points = [
            PointStruct(
                id=i,
                vector=embeddings[i].cpu().numpy().tolist(),
                payload={
                    "id": samples.iloc[i]["id"],
                    "title": samples.iloc[i]["title"],
                    "context": samples.iloc[i]["context"],
                    "question": samples.iloc[i]["question"],
                    "answer_text": samples.iloc[i].get("answer_text", ""),
                    "split": "train"
                }
            )
            for i in range(len(samples))
        ]

        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points
        )

        return operation_info, len(points)

    def insert_validation_samples(self, samples, embeddings, start_id):
        """Insert validation QA samples into the vector database"""
        points = [
            PointStruct(
                id=start_id + i,
                vector=embeddings[i].cpu().numpy().tolist(),
                payload={
                    "context": samples.iloc[i]["context"],
                    "question": samples.iloc[i]["question"],
                    "split": "val"
                }
            )
            for i in range(len(samples))
        ]

        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points
        )

        return operation_info, len(points)

    def insert_all_samples(self, train_samples, train_embeddings,
                           val_samples, val_embeddings):
        """Insert both training and validation QA samples"""

        train_operation, train_count = self.insert_train_samples(
            train_samples, train_embeddings
        )

        val_operation, val_count = self.insert_validation_samples(
            val_samples, val_embeddings, start_id=train_count
        )

        return {
            "train": {"operation": train_operation, "count": train_count},
            "val": {"operation": val_operation, "count": val_count},
            "total": train_count + val_count
        }

    def search(self, query_vector, limit=rag_settings.TOP_K_RETRIEVAL):
        """Search for similar QA samples"""
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )

    def get_collection_info(self):
        """Check if the collection exists"""
        try:
            collections = self.client.get_collections()
            for collection in collections.collections:
                if collection.name == self.collection_name:
                    return {"name": collection.name, "status": "exists"}
            return {"name": self.collection_name, "status": "not found"}
        except Exception as e:
            return {"error": str(e)}
