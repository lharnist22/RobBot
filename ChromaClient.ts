import { ChromaClient } from "chromadb";
import OpenAI from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const chroma = new ChromaClient({ path: process.env.CHROMA_URL ?? "http://localhost:8000" });

async function embed(text: string): Promise<number[]> {
  const r = await openai.embeddings.create({
    model: "text-embedding-3-large",
    input: text,
  });
  return r.data[0].embedding as number[];
}

export async function retrieveContext(query: string, k = 4): Promise<string> {
  const collection = await chroma.getOrCreateCollection({ name: "king_of_kb" });
  const qEmb = await embed(query);

  const res = await collection.query({ queryEmbeddings: [qEmb], nResults: k });
  const docs = res.documents?.[0] ?? [];
  return docs.join("\n\n---\n\n");
}