from transformers import AutoModelForCausalLM, AutoTokenizer
import google.generativeai as genai
from config.settings_rag import rag_settings  

class ModelLoader:
    @staticmethod
    def load_gpt2():
        """Load Arabic GPT-2 model"""
        tokenizer = AutoTokenizer.from_pretrained(rag_settings.GPT2_MODEL)
        model = AutoModelForCausalLM.from_pretrained(rag_settings.GPT2_MODEL)
        return tokenizer, model
    
    @staticmethod
    def load_gemini():
        """Configure Gemini model"""
        genai.configure(api_key=rag_settings.GOOGLE_API_KEY)
        model = genai.GenerativeModel(rag_settings.GEMINI_MODEL)
        return model
