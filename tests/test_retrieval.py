from rag.retrieval.retrieval import Retriever
from config.settings_rag import rag_settings

def test_specific_retrieval():
    question = "متى ولد البيرت اينشتاين ؟"
    print(f"Question: {question}\n")

    retriever = Retriever()
    results = retriever.retrieve_similar_context(question)

    print(f"Top-{rag_settings.TOP_K_RETRIEVAL} Retrieved Contexts:\n")

    for i, item in enumerate(results, start=1):
        print(f"{i}. [Score: {item['score']:.4f}]")
        print(f"   Context: {item['context']}\n")

if __name__ == "__main__":
    print("Running retrieval tests...\n")
    test_specific_retrieval()
