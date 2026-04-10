import dotenv from "dotenv";

dotenv.config({ override: true });

type RetrieveResp = { context?: string; error?: string };

export async function retrieveKbContext(query: string, k = 4): Promise<string> {
  const kbUrl = process.env.KB_URL ?? "http://127.0.0.1:7777";

  const res = await fetch(`${kbUrl}/retrieve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, k }),
  });

  if (!res.ok) throw new Error(`KB service failed: ${res.status} ${await res.text()}`);

  const data = (await res.json()) as RetrieveResp;
  if (data.error) throw new Error(`KB service error: ${data.error}`);

  return (data.context ?? "").trim();
}
