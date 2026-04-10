import "dotenv/config";

export async function parlantCreateSession(): Promise<string> {
  const base = process.env.PARLANT_BASE_URL;
  if (!base) throw new Error("PARLANT_BASE_URL missing in .env");

  const allowGreeting = process.env.PARLANT_ALLOW_GREETING ?? "false";

  const res = await fetch(`${base}/sessions?allow_greeting=${allowGreeting}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });

  if (!res.ok) throw new Error(`Parlant create session failed: ${res.status} ${await res.text()}`);

  const data = (await res.json()) as { session_id?: string; id?: string };
  const id = data.session_id ?? data.id;
  if (!id) throw new Error(`Parlant session id missing: ${JSON.stringify(data)}`);
  return id;
}

export async function parlantSendMessage(sessionId: string, text: string, system?: string): Promise<string> {
  const base = process.env.PARLANT_BASE_URL!;
  // NOTE: endpoint shape may differ depending on Parlant deployment/version.
  // We’ll keep this isolated so it’s easy to adjust once you confirm the exact API path.

  const payload: any = { text };
  if (system) payload.system = system;

  const res = await fetch(`${base}/sessions/${sessionId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) throw new Error(`Parlant send message failed: ${res.status} ${await res.text()}`);

  const data = await res.json();

  // Common patterns: data.reply.text or data.message.text etc.
  const reply =
    data?.reply?.text ??
    data?.message?.text ??
    data?.text ??
    "";

  if (!reply) {
    // Don’t hard-fail; show raw response for debugging
    return JSON.stringify(data, null, 2);
  }
  return reply;
}