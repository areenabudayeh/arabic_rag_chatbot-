import gradio as gr
import time
from rag.pipeline import RAGPipeline
from rag.embeddings.embeddings import EmbeddingGenerator
from rag.vector_store.qdrant_store import VectorDB
from qdrant_client.models import PointStruct

# Pipeline singleton
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline()
        pipeline.initialize_pipeline()
    return pipeline

# Ingest text into Qdrant
def ingest_text(text, metadata=None):
    if not text.strip():
        return "Text is required"

    embedding_gen = EmbeddingGenerator()
    embedding = embedding_gen.generate_single_embedding(text)

    vector_db = VectorDB()
    document_id = int(time.time() * 1000)

    point = PointStruct(
        id=document_id,
        vector=embedding.tolist(),
        payload={
            "context": text,
            "metadata": metadata or {},
            "ingested_at": time.time(),
        }
    )

    vector_db.client.upsert(
        collection_name=vector_db.collection_name,
        points=[point],
        wait=True
    )

    return f"Text ingested successfully with ID: {document_id}"

# Ask question (Gemini + RAG only)
def ask_question(question):
    if not question.strip():
        return "Question is required"

    pipeline = get_pipeline()

    try:
        answer, _ = pipeline.generate_answer(
            question=question,
            model="gemini",
            use_rag=True
        )

        if not answer:
            return "No answer available"

        return answer

    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## RAG + Gemini QA Demo")

    with gr.Tab("Ingest Text"):
        ingest_input = gr.Textbox(
            label="Enter text to store",
            lines=5
        )
        ingest_button = gr.Button("Ingest")
        ingest_output = gr.Textbox(
            label="Status",
            lines=2
        )

        ingest_button.click(
            ingest_text,
            inputs=ingest_input,
            outputs=ingest_output
        )

    with gr.Tab("Ask Question"):
        question_input = gr.Textbox(
            label="Enter your question",
            lines=3
        )
        ask_button = gr.Button("Ask")
        answer_output = gr.Textbox(
            label="Answer",
            lines=5
        )

        ask_button.click(
            ask_question,
            inputs=question_input,
            outputs=answer_output
        )

demo.launch()
