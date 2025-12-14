from .env import GOOGLE_API_KEY, QDRANT_HOST, QDRANT_PORT,COLLECTION_NAME

class RAGSettings:
    # Models 
    EMBEDDING_MODEL = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
    GPT2_MODEL = "aubmindlab/aragpt2-medium"
    GEMINI_MODEL = "gemini-2.5-flash"

    # Qdrant 
    QDRANT_HOST = QDRANT_HOST
    QDRANT_PORT = QDRANT_PORT
    COLLECTION_NAME = COLLECTION_NAME

    # Retrieval 
    TOP_K_RETRIEVAL = 5
    TOP_K_EVALUATION = 3
    SIMILARITY_THRESHOLD_EVALUATION = 0.60
    SIMILARITY_THRESHOLD = 0.3

    # GPT2 
    TOP_K_GENERATION=1
    GPT2_MAX_TOKENS = 30
    GPT2_TEMPERATURE = 0.7
    GPT2_TOP_P = 0.8
    GPT2_NUM_BEAMS = 8
    GPT2_NO_REPEAT_NGRAM_SIZE = 3

    # Gemini 
    GEMINI_MAX_TOKENS = 1024
    GEMINI_TEMPERATURE = 0.1
    GEMINI_TOP_P = 0.9

    # Evaluation 
    EVAL_SUBSET_SIZE = 50
    MAX_RETRIES = 5
    BASE_DELAY = 10

    # API 
    GOOGLE_API_KEY = GOOGLE_API_KEY


rag_settings = RAGSettings()
