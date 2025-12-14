import os
from dotenv import load_dotenv

load_dotenv()

# Secrets
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
