from rag.pipeline import RAGPipeline

def test_single_question():
    question = "في أي عام ولد ألبرت أينشتاين؟"

    print("\nInitializing RAG Pipeline")
    pipeline = RAGPipeline().initialize_pipeline()
    print(f"\nQuestion: {question}")

    # GPT-2 with RAG
    ans_gpt2_rag, _ = pipeline.generate_answer(
        question, model="gpt2", use_rag=True
    )
    print("\n[GPT-2 | RAG]")
    print(ans_gpt2_rag)

    # Gemini with RAG
    ans_gemini_rag, _ = pipeline.generate_answer(
        question, model="gemini", use_rag=True
    )
    print("\n[Gemini | RAG]")
    print(ans_gemini_rag)

    # GPT-2 without RAG
    ans_gpt2_no_rag = pipeline.generate_answer(
        question, model="gpt2", use_rag=False
    )
    print("\n[GPT-2 | No-RAG]")
    print(ans_gpt2_no_rag)

    # Gemini without RAG
    ans_gemini_no_rag = pipeline.generate_answer(
        question, model="gemini", use_rag=False
    )
    print("\n[Gemini | No-RAG]")
    print(ans_gemini_no_rag)


if __name__ == "__main__":
    print("\nRunning generation tests...\n")
    test_single_question()
