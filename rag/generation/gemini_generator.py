import google.generativeai as genai
from rag.retrieval.retrieval import Retriever
from rag.generation.prompt import generate_prompt, truncate_answer
from config.settings_rag import rag_settings


class GeminiGenerator:
    def __init__(self, model):
        self.model = model
        self.retriever = Retriever()
    
    def generate_with_rag(self, question):
        """Generate answer using RAG with Gemini """
        try:
            retrieved_contexts = self.retriever.retrieve_similar_context(question)

            if not retrieved_contexts:
                return "There is no available answer", []

            top_context = retrieved_contexts[:1]

            prompt_text = generate_prompt(question, top_context)

            response = self.model.generate_content(
                prompt_text,
                generation_config={
                    "temperature": rag_settings.GEMINI_TEMPERATURE,
                    "top_p": rag_settings.GEMINI_TOP_P,
                    "max_output_tokens": rag_settings.GEMINI_MAX_TOKENS,
                }
            )

            if response.candidates and response.candidates[0].content.parts:
                answer = "".join(
                    [p.text for p in response.candidates[0].content.parts]
                ).strip()
                return truncate_answer(answer), top_context

            return "No response generated", retrieved_contexts
            
        except Exception as e:
            return f"Error: {str(e)}", []


    def generate_without_rag(self, question):
        """Generate answer without RAG"""
        try:
            response = self.model.generate_content(
                question,
                generation_config={
                    "temperature": rag_settings.GEMINI_TEMPERATURE,
                    "top_p": rag_settings.GEMINI_TOP_P,
                    "max_output_tokens": rag_settings.GEMINI_MAX_TOKENS,
                }
            )
            
            if response.candidates and response.candidates[0].content.parts:
                answer = "".join(
                    [p.text for p in response.candidates[0].content.parts]
                ).strip()
                return truncate_answer(answer)
            
            return "No response generated"
            
        except Exception as e:
            return f"Error: {str(e)}"