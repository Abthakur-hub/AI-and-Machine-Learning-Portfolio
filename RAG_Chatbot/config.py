import os
from dotenv import load_dotenv

load_dotenv()


GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY"
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)


# Gemini embedding model
EMBEDDING_MODEL = (
    "models/gemini-embedding-001"
)


# Groq model
LLM_MODEL = (
    "llama-3.3-70b-versatile"
)


CHROMA_DB_DIR = (
    "chroma_db"
)


TOP_K = 4


TEMPERATURE = 0