from rag.pipeline import RAGPipeline
from evaluation.retrieval_eval import RetrievalEvaluator
from evaluation.generation_eval import GenerationEvaluator

pipeline = RAGPipeline().initialize_pipeline()
df_val = pipeline.df_val

# Retrieval
retrieval_eval = RetrievalEvaluator(df_val)
retrieval_results = retrieval_eval.evaluate()

print("\n=== Retrieval Evaluation ===")
for k, v in retrieval_results.items():
    print(f"{k}: {v:.4f}")

# Generation
generation_eval = GenerationEvaluator()

generators = {
    "gpt2_rag": lambda q: pipeline.generate_answer(q, "gpt2", True)[0],
    "gpt2_no_rag": lambda q: pipeline.generate_answer(q, "gpt2", False),
    "gemini_rag": lambda q: pipeline.generate_answer(q, "gemini", True)[0],
    "gemini_no_rag": lambda q: pipeline.generate_answer(q, "gemini", False),
}

gen_results = generation_eval.evaluate(df_val, generators)

print("\n=== Generation Evaluation ===")
for model, metrics in gen_results.items():
    print(f"\n-- {model} --")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
