import time
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status

from rag.pipeline import RAGPipeline
from rag.embeddings.embeddings import EmbeddingGenerator
from rag.vector_store.qdrant_store import VectorDB
from rag.generation.models_loader import ModelLoader
from qdrant_client.models import PointStruct


# Pipeline Singleton
_pipeline = None

def get_pipeline():
    """
    Lazily initialize and return the RAG pipeline.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
        _pipeline.initialize_pipeline()
    return _pipeline


# Health Check
@api_view(["GET"])
@permission_classes([AllowAny])
def health_check(request):
    """
    Health check endpoint for monitoring.
    Checks Qdrant, Gemini API, and embedding model.
    """
    try:
        # Qdrant check 
        vector_db = VectorDB()
        collection_info = vector_db.get_collection_info()

        # Gemini check 
        try:
            ModelLoader.load_gemini()
            gemini_status = "connected"
        except Exception as e:
            gemini_status = f"error: {str(e)}"

        # Embedding model check 
        try:
            embedding_gen = EmbeddingGenerator()
            embedding_gen.generate_single_embedding("health_check")
            embedding_status = "ready"
        except Exception as e:
            embedding_status = f"error: {str(e)}"

        return Response({
            "status": "healthy",
            "components": {
                "qdrant": collection_info,
                "gemini_api": gemini_status,
                "embedding_model": embedding_status,
            },
            "timestamp": time.time(),
        })

    except Exception as e:
        return Response(
            {
                "status": "unhealthy",
                "error": str(e),
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

# Query Endpoint
@api_view(["POST"])
def query_view(request):
    """
    Submit a user query and return a generated answer.
    """
    question = request.data.get("question", "").strip()

    if not question:
        return Response(
            {"error": "Question is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    pipeline = get_pipeline()

    try:
        answer, retrieved_contexts = pipeline.gemini_generator.generate_with_rag(question)
    except Exception as e:
        return Response(
            {"error": f"Generation failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return Response({
        "question": question,
        "answer": answer,
        "retrieved_contexts": [
            {
                "context": ctx.get("context", ""),
                "relevance_score": ctx.get("score", 0.0),  
            }
            for ctx in retrieved_contexts
        ],
        "context_count": len(retrieved_contexts),
    })


# Ingestion Endpoint
@api_view(["POST"])
def ingest_view(request):
    """
    Ingest new text into the vector database.
    """
    text = request.data.get("text", "").strip()
    metadata = request.data.get("metadata", {})

    if not text:
        return Response(
            {"error": "Text is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    embedding_generator = EmbeddingGenerator()
    embedding = embedding_generator.generate_single_embedding(text)

    vector_db = VectorDB()
    document_id = int(time.time() * 1000)

    point = PointStruct(
        id=document_id,
        vector=embedding.tolist(),
        payload={
            "context": text,
            "metadata": metadata,
            "ingested_at": time.time(),
        },
    )

    vector_db.client.upsert(
        collection_name=vector_db.collection_name,
        points=[point],
        wait=True,
    )

    return Response({
        "status": "success",
        "document_id": document_id,
    })