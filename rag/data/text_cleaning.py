import re

def normalize_arabic(text):
    """Normalize Arabic text by removing diacritics and unwanted characters"""
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    text = re.sub(r'[^\u0600-\u06FF0-9\s؟.,:؛!]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def clean_dataframe(df):
    """Clean all text columns in a dataframe"""
    df_clean = df.copy()
    
    for col in ['context', 'question']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).apply(normalize_arabic)
    
    # Extract and clean answer text
    if 'answers' in df_clean.columns:
        df_clean['answer_text'] = df_clean['answers'].apply(
            lambda x: x['text'][0] if x['text'] else ""
        )
        df_clean['answer_text'] = df_clean['answer_text'].astype(str).apply(normalize_arabic)
    
    return df_clean