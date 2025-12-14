def generate_prompt(question, retrieved_contexts):
    """Generate prompt """
    prompt = "أجب على السؤال التالي بناءً على المعلومات المتاحة أدناه:\n\n"
    
    if retrieved_contexts:  
        prompt += f"المعلومة: {retrieved_contexts[0]['context']}\n"
    
    prompt += f"\nالسؤال: {question}\nالإجابة:"
    return prompt

def truncate_answer(answer, stop_chars=['.', '،', '?', '!']):
    """Truncate answer at natural stopping points"""
    if not answer:
        return answer
    
    for char in stop_chars:
        if char in answer:
            return answer.split(char)[0] + char
    return answer