import time
import requests

URL = "http://localhost:8000/chat"

QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?",  # should be refused
]

lines = []
for q in QUERIES:
    r = requests.post(URL, json={"query": q}, timeout=120)
    if r.status_code != 200:
        block = f"Q: {q}\nERROR {r.status_code}: {r.text}\n"
    else:
        d = r.json()
        block = (
            f"Q: {q}\n"
            f"Confidence: {d['confidence_score']}\n"
            f"Chunks retrieved: {len(d['retrieved_chunks'])}\n"
            f"A: {d['answer']}\n"
        )
    print("=" * 80)
    print(block)
    lines.append("=" * 80 + "\n" + block)
    time.sleep(10)  # stay under free-tier per-minute limits

with open("sample_outputs.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nSaved results to sample_outputs.txt")