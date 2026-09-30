import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

PDF_PATH = "data/Ebook-Agentic-AI.pdf"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
LLM_MODEL = "gemini-3.8-flash"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 6
MIN_SCORE = 0.62