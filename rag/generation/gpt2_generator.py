from rag.retrieval.retrieval import Retriever
from rag.generation.prompt import generate_prompt, truncate_answer
from config.settings_rag import rag_settings


class GPT2Generator:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.retriever = Retriever()

    def generate_with_rag(self, question):
        """Generate answer using RAG """

        retrieved_contexts = self.retriever.retrieve_similar_context(question)
        
        if not retrieved_contexts:
            return "There is no available answer", []
        
        top_context = retrieved_contexts[:1]
        
        prompt_text = generate_prompt(question, top_context)

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        output = self.model.generate(
            **inputs,
            max_new_tokens=rag_settings.GPT2_MAX_TOKENS,
            no_repeat_ngram_size=rag_settings.GPT2_NO_REPEAT_NGRAM_SIZE,
            do_sample=False,
            top_p=rag_settings.GPT2_TOP_P,
            num_beams=rag_settings.GPT2_NUM_BEAMS,
            temperature=rag_settings.GPT2_TEMPERATURE,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        final_answer = answer.replace(prompt_text, "").strip()
        final_answer = truncate_answer(final_answer)

        return final_answer, top_context


    def generate_without_rag(self, question):
        """Generate answer without RAG"""

        inputs = self.tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        output = self.model.generate(
            **inputs,
            max_new_tokens=rag_settings.GPT2_MAX_TOKENS,
            no_repeat_ngram_size=rag_settings.GPT2_NO_REPEAT_NGRAM_SIZE,
            do_sample=False,
            top_p=rag_settings.GPT2_TOP_P,
            num_beams=rag_settings.GPT2_NUM_BEAMS,
            temperature=rag_settings.GPT2_TEMPERATURE,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        final_answer = answer.replace(question, "").strip()
        final_answer = truncate_answer(final_answer)

        return final_answer
