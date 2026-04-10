export async function retrieveKbContext(query: string, k = 4): Promise<string> {
  const res = await fetch("http://127.0.0.1:7777/retrieve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, k }),
  });

  if (!res.ok) {
    throw new Error(`KB retrieve failed: ${res.status} ${await res.text()}`);
  }

  const data = (await res.json()) as { context?: string };
  return data.context ?? "";
}

