from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.config import INDEX_NAME, EMBED_MODEL, EMBED_DIM, LLM_MODEL, TOP_K, MIN_SCORE

REFUSAL = "I cannot answer based on the provided document."


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float


def _to_text(content) -> str:
    """Gemini sometimes returns a list of parts instead of a plain string."""
    if isinstance(content, str):
        return content
    return "".join(
        part if isinstance(part, str) else part.get("text", "") for part in content
    )


def build_rag_graph():
    # Queries use task_type="retrieval_query" (documents were embedded as "retrieval_document")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        task_type="retrieval_query",
        output_dimensionality=EMBED_DIM,
    )
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)

    def retrieve_node(state: AgentState):
        results = vectorstore.similarity_search_with_score(state["question"], k=TOP_K)
        context = [doc.page_content for doc, _ in results]
        top_score = max((s for _, s in results), default=0.0)
        return {"context": context, "score": round(float(top_score), 4)}

    def generate_node(state: AgentState):
        # Guardrail: nothing relevant found, so skip the LLM
        if not state["context"] or state["score"] < MIN_SCORE:
            return {"answer": REFUSAL}

        context_str = "\n\n---\n\n".join(state["context"])
        prompt = f"""You are a strict assistant answering questions about an eBook on Agentic AI.
The context below contains excerpts from that eBook. If the question mentions "the eBook" or "the document", it refers to this context.

Rules:
- Answer using ONLY the context below. Do not use outside knowledge.
- If the context partly answers the question, give the best answer the context supports, and summarize in your own words.
- Only if the context has nothing relevant, reply exactly: "{REFUSAL}"

Context:
{context_str}

Question: {state['question']}

Answer:"""
        response = llm.invoke(prompt)
        return {"answer": _to_text(response.content).strip()}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


if __name__ == "__main__":
    import time
    graph = build_rag_graph()
    questions = [
        "What is Agentic AI according to the eBook?",
        "How do AI agents differ from traditional automation systems?",
        "What are the core components of an Agentic Architecture?",
        "What role does memory play in Agentic AI workflows?",
        "Who won the 2022 FIFA World Cup?",
    ]
    for q in questions:
        r = graph.invoke({"question": q, "context": [], "answer": "", "score": 0.0})
        print(f"\nScore: {r['score']} | Q: {q}\nA: {r['answer'][:200]}")
        time.sleep(5)  # stay under free-tier rate limits