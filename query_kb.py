import os
from openai import OpenAI
import chromadb

COLLECTION_NAME = "king_of_kb"
OPENAI_EMBED_MODEL = "text-embedding-3-large"

client = OpenAI()
chroma = chromadb.PersistentClient(path="./chroma_db")
col = chroma.get_collection(COLLECTION_NAME)

def embed(text: str) -> list[float]:
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text)
    return resp.data[0].embedding

def search(query: str, k: int = 4) -> list[str]:
    q = embed(query)
    res = col.query(query_embeddings=[q], n_results=k)
    return res["documents"][0]

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY environment variable first.")
    q = input("Ask: ")
    hits = search(q, k=4)
    print("\n--- TOP HITS ---\n")
    for i, h in enumerate(hits, 1):
        print(f"[{i}]\n{h}\n")