import os
import json
from openai import OpenAI
import chromadb

# --- config ---
JSONL_PATH = "knowledge_base.jsonl"
COLLECTION_NAME = "king_of_kb"
OPENAI_EMBED_MODEL = "text-embedding-3-large"  # or "text-embedding-3-small" (cheaper)
# --------------

def _load_local_env() -> None:
    env_path = ".env"
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


_load_local_env()
client = OpenAI()
chroma = chromadb.PersistentClient(path="./chroma_db")

def embed(text: str) -> list[float]:
    resp = client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=text
    )
    return resp.data[0].embedding

def main():
    rebuild = os.getenv("REBUILD_COLLECTION", "true").lower() in {"1", "true", "yes", "y"}
    if rebuild:
        try:
            chroma.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            print(f"No existing collection to delete: {COLLECTION_NAME}")

    col = chroma.get_or_create_collection(COLLECTION_NAME)

    # load jsonl
    records = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    print(f"Loaded {len(records)} chunks from {JSONL_PATH}")

    # upsert in batches
    batch_size = 50
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]

        ids = [r["id"] for r in batch]
        docs = [r["text"] for r in batch]
        metas = [{"source": r.get("source", "")} for r in batch]
        embs = [embed(t) for t in docs]

        col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        print(f"Upserted {start + len(batch)}/{len(records)}")

    print("Done. Chroma DB written to ./chroma_db")

if __name__ == "__main__":
    # expects OPENAI_API_KEY in env
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY environment variable first.")
    main()
