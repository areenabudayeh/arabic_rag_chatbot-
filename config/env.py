import os
from dotenv import load_dotenv

load_dotenv()

# Secrets
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DJANGO_SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")

# Environment
DJANGO_DEBUG = os.getenv("DJANGO_DEBUG", "True").lower() == "true"
DJANGO_ALLOWED_HOSTS = os.getenv("DJANGO_ALLOWED_HOSTS").split(",")

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


# API Security
API_KEYS = os.getenv("API_KEYS").split(",")
API_KEY_HEADER = os.getenv("API_KEY_HEADER")

# CORS
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS").split(",")
