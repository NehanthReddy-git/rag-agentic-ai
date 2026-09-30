# RAG-based AI Chatbot: Agentic AI eBook

A Retrieval-Augmented Generation (RAG) chatbot that answers questions **strictly from the Agentic AI eBook**. It uses LangGraph for the workflow, Pinecone as the vector database, Gemini for embeddings and answer generation, and FastAPI for the API.

For every question, the API returns:
- the generated **answer**
- the **retrieved context chunks**
- a **confidence score** (cosine similarity of the best-matching chunk)

## Architecture

```
INGESTION (run once)
PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
    -> Gemini embeddings (gemini-embedding-001, 768-d) -> Pinecone index (cosine)

QUERY (every request)
POST /chat -> FastAPI -> LangGraph:
    START -> retrieve -> generate -> END
    retrieve : embed question, top-k similarity search in Pinecone, record best score
    generate : if best score < threshold, refuse; otherwise ask Gemini to answer
               using ONLY the retrieved chunks
-> { answer, retrieved_chunks, confidence_score }
```

### Components

| File | Purpose |
|---|---|
| `src/config.py` | Loads environment variables and holds constants (chunk size, top-k, threshold, model names) |
| `src/ingestion.py` | Loads the PDF, splits it, creates the Pinecone index, embeds and uploads chunks |
| `src/graph.py` | LangGraph state, `retrieve` and `generate` nodes, compiled workflow |
| `app.py` | FastAPI app exposing `POST /chat` |
| `tests_sample_queries.py` | Sends the 5 sample queries to the API and saves outputs |

## Project Structure

```
rag-agentic-ai/
├── data/
│   └── Ebook-Agentic-AI.pdf
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   └── graph.py
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── sample_outputs.txt
└── tests_sample_queries.py
```

## Setup

**Prerequisites:** Python 3.10+, a free [Google AI Studio](https://aistudio.google.com/apikey) API key, and a free [Pinecone](https://www.pinecone.io) API key.

1. Clone the repository and enter it:
```bash
   git clone <YOUR_REPO_URL>
   cd rag-agentic-ai
```
2. Create and activate a virtual environment:
```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS / Linux
```
3. Install dependencies:
```bash
   pip install -r requirements.txt
```
4. Create your environment file by copying the template, then fill in your keys:
```bash
   copy .env.example .env       # Windows
   cp .env.example .env         # macOS / Linux
```
```
   GOOGLE_API_KEY=your_google_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=agentic-ai-index
```
5. Place the eBook PDF at `data/Ebook-Agentic-AI.pdf`.

## Usage

### 1. Run ingestion (once)

```bash
python -m src.ingestion
```

This creates the Pinecone index if needed (768 dimensions, cosine metric), splits the PDF into chunks, embeds them and uploads them. Chunks get fixed IDs (`chunk-0`, `chunk-1`, ...), so re-running overwrites instead of duplicating. Batches are paced with automatic retries to respect free-tier rate limits.

### 2. Start the API

```bash
uvicorn app:app --reload
```

Interactive docs: http://localhost:8000/docs

### 3. Ask a question

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"query\": \"What is Agentic AI according to the eBook?\"}"
```

Response format:

```json
{
  "answer": "...",
  "retrieved_chunks": ["...", "..."],
  "confidence_score": 0.79
}
```

### 4. Run the sample queries

With the server running, in a second terminal:

```bash
python tests_sample_queries.py
```

Results are also saved to `sample_outputs.txt`.

## Sample Results

| Query | Confidence | Result |
|---|---|---|
| What is Agentic AI according to the eBook? | <SCORE> | <one-line summary of answer> |
| How do AI agents differ from traditional automation systems? | <SCORE> | <one-line summary of answer> |
| What are the core components of an Agentic Architecture? | <SCORE> | <one-line summary of answer> |
| What role does memory play in Agentic AI workflows? | <SCORE> | <one-line summary of answer> |
| Who won the 2022 FIFA World Cup? | 0.496 | Refused: "I cannot answer based on the provided document." |

Full outputs: see `sample_outputs.txt`.

## Design Choices

- **Chunking (1000 chars, 200 overlap):** small enough for precise retrieval, with overlap so ideas are not cut at chunk boundaries.
- **Retrieval (top-k = 6):** enough context to capture definitions and lists that span several chunks.
- **Real confidence score:** the score is the top cosine similarity returned by Pinecone for the retrieved chunks, not a hardcoded number.
- **Two-layer grounding:**
  1. *Retrieval guardrail:* if the best similarity is below `MIN_SCORE` (0.62), the workflow refuses without calling the LLM. Ebook questions scored about 0.75 to 0.79 while the off-topic question scored about 0.50, so the threshold sits between them.
  2. *Prompt guardrail:* the LLM is told to use only the provided context and to refuse when nothing relevant is present. Temperature is 0 for deterministic, grounded answers.
- **Embedding task types:** documents are embedded with `retrieval_document` and queries with `retrieval_query`, as recommended for Gemini embeddings.
- **Gemini instead of OpenAI:** the reference guide suggested OpenAI, but I used Gemini (embeddings and LLM) because of API credit constraints. The architecture is provider-independent, and only the model classes and the index dimension (768) differ.
- **Safe re-ingestion:** fixed chunk IDs prevent duplicate vectors.

## Limitations

- The confidence score is retrieval similarity, not a calibrated probability of answer correctness.
- The free Gemini tier has request quotas, so heavy use can return HTTP 429. The API returns a clear error message in that case.
- Answers are limited to the eBook's content by design.