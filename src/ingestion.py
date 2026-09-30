import time
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.config import (PINECONE_API_KEY, INDEX_NAME, PDF_PATH, EMBED_MODEL,
                        EMBED_DIM, CHUNK_SIZE, CHUNK_OVERLAP)

BATCH_SIZE = 20        # chunks per request group
PAUSE_SECONDS = 15     # 20 chunks / 15s = ~80 per minute, under the 100/min limit
MAX_RETRIES = 5


def ensure_index(pc: Pinecone):
    """Create the Pinecone index if it doesn't exist yet."""
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)


def run_ingestion():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index(pc)

    # 1. Load the PDF (one Document per page)
    docs = PyPDFLoader(PDF_PATH).load()
    print(f"Loaded {len(docs)} pages")

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    # 3. Embed and store in small batches with retry
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        task_type="retrieval_document",
        output_dimensionality=EMBED_DIM,
    )
    vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        ids = [f"chunk-{start + j}" for j in range(len(batch))]  # fixed IDs = no duplicates

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                vector_store.add_documents(batch, ids=ids)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"  Rate limited, waiting 35s (attempt {attempt}/{MAX_RETRIES})...")
                    time.sleep(35)
                else:
                    raise
        else:
            raise RuntimeError("Too many rate-limit failures. Wait a few minutes and rerun.")

        print(f"Uploaded {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}")
        time.sleep(PAUSE_SECONDS)

    print("Ingestion complete.")


if __name__ == "__main__":
    run_ingestion()