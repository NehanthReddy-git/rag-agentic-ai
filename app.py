from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.graph import build_rag_graph

app = FastAPI(title="Agentic AI RAG API")
graph = build_rag_graph()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: list[str]
    confidence_score: float


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    try:
        result = graph.invoke(
            {"question": request.query, "context": [], "answer": "", "score": 0.0}
        )
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            raise HTTPException(status_code=429, detail="Model quota exceeded. Try again later.")
        raise HTTPException(status_code=500, detail=msg)

    return QueryResponse(
        answer=result["answer"],
        retrieved_chunks=result["context"],
        confidence_score=result["score"],
    )