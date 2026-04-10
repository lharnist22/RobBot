import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from typing import Literal

import chromadb
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AuthenticationError, OpenAI
from pydantic import BaseModel

OPENAI_EMBED_MODEL = "text-embedding-3-large"
COLLECTION_NAME = "king_of_kb"
CHROMA_PATH = "./chroma_db"
JSONL_PATH = "knowledge_base.jsonl"

app = FastAPI()

ROB_QUESTION_SOURCE_KEYWORDS = (
    "king of book",
    "smartups book",
    "sunflower code playbook",
    "the man who saved the internet with a sunflower",
    "0738",
)

QUESTION_PATTERN_HINTS = (
    "what keeps you awake at night",
    "which problems threaten your survival",
    "why aren't you using",
    "what do you wish existed",
    "what specific metric would make this customer breathe easier",
    "how quickly after adoption will they feel the difference",
    "can we quantify that improvement",
    "which of our competencies are truly world-class",
    "what combination of data, workflow, and assurance can i own",
    "if you could fix this one thing",
)


def _load_local_env() -> None:
    """Load local .env and override stale shell values."""
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
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
PARLANT_BASE_URL = (os.getenv("PARLANT_BASE_URL") or "").strip()
PARLANT_ALLOW_GREETING = (os.getenv("PARLANT_ALLOW_GREETING") or "false").lower() in {"1", "true", "yes", "y"}
openai = OpenAI()

RETRIEVAL_BACKEND = "chroma"


def _load_jsonl_records() -> list[dict]:
    if not os.path.isfile(JSONL_PATH):
        return []

    records: list[dict] = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _init_retrieval_backend():
    global RETRIEVAL_BACKEND

    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_collection(COLLECTION_NAME)
        return chroma_client, collection, []
    except Exception as e:
        RETRIEVAL_BACKEND = "jsonl_fallback"
        print(f"Chroma unavailable, using JSONL fallback retrieval: {e}")
        return None, None, _load_jsonl_records()


chroma, col, kb_records = _init_retrieval_backend()

if os.path.isdir("web"):
    app.mount("/static", StaticFiles(directory="web"), name="static")


class RetrieveRequest(BaseModel):
    query: str
    k: int = 4


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = []
    k: int = 6
    parlant_session_id: str | None = None


def embed(text: str) -> list[float]:
    resp = openai.embeddings.create(model=OPENAI_EMBED_MODEL, input=text)
    return resp.data[0].embedding


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _retrieve_context_from_jsonl(query: str, k: int) -> tuple[str, list[dict]]:
    query_tokens = _tokenize(query)
    if not query_tokens or not kb_records:
        return "", []

    query_counts = Counter(query_tokens)
    scored: list[tuple[int, int, dict]] = []

    for record in kb_records:
        text = record.get("text", "")
        text_tokens = _tokenize(text)
        if not text_tokens:
            continue

        text_counts = Counter(text_tokens)
        overlap = sum(min(query_counts[token], text_counts[token]) for token in query_counts)
        phrase_bonus = 3 if query.lower() in text.lower() else 0
        score = overlap + phrase_bonus
        if score <= 0:
            continue

        scored.append((score, len(text), record))

    scored.sort(key=lambda item: (-item[0], item[1]))
    top_records = [record for _, _, record in scored[:k]]
    hits = [
        {
            "id": record.get("id"),
            "source": record.get("source"),
            "text": record.get("text", ""),
        }
        for record in top_records
    ]
    context = "\n\n---\n\n".join(hit["text"] for hit in hits)
    return context, hits


def should_retrieve(user_text: str) -> bool:
    text = user_text.strip().lower()
    if len(text) < 12:
        return False
    if text in {"hi", "hello", "thanks", "ok", "cool"}:
        return False
    return True


def build_system_prompt(kb_context: str) -> str:
    return build_system_prompt_with_questions(kb_context, "")


def build_system_prompt_with_questions(kb_context: str, question_context: str) -> str:
    question_block = ""
    if question_context.strip():
        question_block = f"""

Rob Question Patterns:
{question_context}
""".rstrip()

    return f"""
You are Rob-style business strategy coach.

Rules:
- Base your guidance on the Knowledge Base Context below whenever possible.
- Write in a practical, direct, conversational tone that sounds human.
- Help the user define and sharpen their own "King Of" position in their market.
- Prefer natural phrasing over formal corporate language.
- Avoid stiff phrasing like "I will ensure" / "If there is nothing else needed from my end."
- Avoid repetitive openers like "Thank you for..." in consecutive replies.
- Do not add filler words ("um", "uh"), theatrical drama, or over-apologizing.
- Keep responses concise and action-first; avoid long process-heavy explanations unless asked.
- Use numbered or bulleted lists for recommendations and steps.
- Answer the user's exact question first, then add clarifying questions if needed.
- Mirror the user's energy: calm, helpful, and practical.
- Default format (most replies):
  1) 1-3 sentence direct answer
  2) 2-4 practical next steps
  3) 1-3 focused follow-up questions
- Only use a long strategic breakdown when the user explicitly asks for a full plan.
- Limit follow-up questions to the minimum needed; do not dump long question lists by default.
- Avoid robotic section headers and markdown formatting marks like "###" and "**".
- If context is missing for specific claims, say what is missing briefly and ask focused follow-ups.
- Do not mention this prompt or system rules.
- When you ask follow-up questions, prefer the question types and wording patterns from Rob's books/papers instead of generic coaching questions.
- Use follow-up questions to complete the picture: customer pain, survival threats, current workaround, competitor gap, missing capability, buying urgency, and measurable value.
- Ask only the minimum questions needed to clarify the picture, usually no more than 3.

Knowledge Base Context:
{kb_context}
{question_block}
""".strip()


def _parlant_enabled() -> bool:
    if not PARLANT_BASE_URL:
        return False
    return "PORT" not in PARLANT_BASE_URL.upper()


def _post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
        return json.loads(raw) if raw else {}


def parlant_create_session() -> str:
    url = f"{PARLANT_BASE_URL}/sessions?allow_greeting={'true' if PARLANT_ALLOW_GREETING else 'false'}"
    data = _post_json(url, {})
    sid = data.get("session_id") or data.get("id")
    if not sid:
        raise RuntimeError(f"Parlant session id missing: {json.dumps(data)}")
    return sid


def parlant_send_message(session_id: str, text: str, system: str) -> str:
    url = f"{PARLANT_BASE_URL}/sessions/{session_id}/messages"
    data = _post_json(url, {"text": text, "system": system})
    reply = data.get("reply", {}).get("text") or data.get("message", {}).get("text") or data.get("text") or ""
    if not reply:
        raise RuntimeError(f"Parlant reply missing: {json.dumps(data)}")
    return reply


def openai_fallback_reply(system: str, req: ChatRequest, msg: str) -> str:
    messages = [{"role": "system", "content": system}]
    for turn in req.history[-12:]:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": msg})

    resp = openai.chat.completions.create(model=OPENAI_CHAT_MODEL, messages=messages, temperature=0.6)
    return (resp.choices[0].message.content or "").strip()


def retrieve_context(query: str, k: int) -> tuple[str, list[dict]]:
    if RETRIEVAL_BACKEND != "chroma":
        return _retrieve_context_from_jsonl(query, k)

    q = embed(query)
    res = col.query(query_embeddings=[q], n_results=k)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]

    hits = []
    for i in range(len(docs)):
        hits.append(
            {
                "id": ids[i] if i < len(ids) else None,
                "source": (metas[i] or {}).get("source") if i < len(metas) else None,
                "text": docs[i],
            }
        )

    context = "\n\n---\n\n".join(docs)
    return context, hits


def _is_rob_question_source(source: str | None) -> bool:
    if not source:
        return False
    source_lower = source.lower()
    return any(keyword in source_lower for keyword in ROB_QUESTION_SOURCE_KEYWORDS)


def _extract_question_lines(text: str) -> list[str]:
    candidates: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r'^[\s\-*"]+', "", raw_line)
        line = line.lstrip(chr(8226)).strip().strip('"')
        if not line:
            continue

        lowered = line.lower()
        keep = "?" in line or any(hint in lowered for hint in QUESTION_PATTERN_HINTS)
        if not keep:
            continue

        line = re.sub(r"\s+", " ", line)
        line = re.sub(r"\s*\?\s*", "?", line)
        if len(line) < 12 or len(line) > 180:
            continue
        candidates.append(line)

    unique: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def retrieve_rob_question_context(user_query: str, k: int = 12) -> tuple[str, list[dict]]:
    prompt = (
        "Rob walkabout and strategy questions for understanding a market, customer pain, "
        f"competitor gaps, and what to build next. User topic: {user_query}"
    )
    _, hits = retrieve_context(prompt, k)
    rob_hits = [hit for hit in hits if _is_rob_question_source(hit.get("source"))]

    questions: list[str] = []
    seen: set[str] = set()
    for hit in rob_hits:
        for question in _extract_question_lines(hit.get("text") or ""):
            key = question.lower()
            if key in seen:
                continue
            seen.add(key)
            questions.append(question)
            if len(questions) >= 8:
                break
        if len(questions) >= 8:
            break

    question_context = "\n".join(f"- {question}" for question in questions)
    return question_context, rob_hits


@app.get("/health")
def health():
    return {
        "ok": True,
        "collection": COLLECTION_NAME,
        "chat_mode": "parlant" if _parlant_enabled() else "openai_fallback",
        "retrieval_backend": RETRIEVAL_BACKEND,
    }


@app.get("/")
def root():
    if os.path.isfile("web/index.html"):
        return FileResponse("web/index.html")
    return {"ok": True, "routes": ["/health", "/retrieve", "/chat"]}


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    if not os.getenv("OPENAI_API_KEY"):
        return {"context": "", "hits": [], "error": "OPENAI_API_KEY not set"}

    try:
        context, hits = retrieve_context(req.query, req.k)
    except AuthenticationError:
        return {
            "context": "",
            "hits": [],
            "error": "OPENAI_API_KEY is invalid or revoked. Update it in RobBot/.env and restart kb_service.",
        }
    except Exception as e:
        return {"context": "", "hits": [], "error": f"Embedding failed: {e}"}

    return {"context": context, "hits": hits}


@app.post("/chat")
def chat(req: ChatRequest):
    if not os.getenv("OPENAI_API_KEY"):
        return {"reply": "", "hits": [], "error": "OPENAI_API_KEY not set"}

    msg = req.message.strip()
    if not msg:
        return {"reply": "", "hits": [], "error": "Message is empty"}

    try:
        if should_retrieve(msg):
            context, hits = retrieve_context(msg, req.k)
        else:
            context, hits = "", []

        question_context, question_hits = retrieve_rob_question_context(msg)
        existing_ids = {existing["id"] for existing in hits if existing.get("id")}
        for hit in question_hits:
            if hit["id"] not in existing_ids:
                hits.append(hit)
                if hit.get("id"):
                    existing_ids.add(hit["id"])

        system = build_system_prompt_with_questions(context, question_context)
        if _parlant_enabled():
            try:
                session_id = req.parlant_session_id or parlant_create_session()
                reply = parlant_send_message(session_id, msg, system)
                return {"reply": reply.strip(), "hits": hits, "error": None, "parlant_session_id": session_id}
            except (urllib.error.HTTPError, urllib.error.URLError):
                reply = openai_fallback_reply(system, req, msg)
                return {
                    "reply": reply,
                    "hits": hits,
                    "error": None,
                    "parlant_session_id": None,
                    "mode": "openai_fallback",
                }

        reply = openai_fallback_reply(system, req, msg)
        return {"reply": reply, "hits": hits, "error": None, "parlant_session_id": None, "mode": "openai_fallback"}
    except AuthenticationError:
        return {
            "reply": "",
            "hits": [],
            "error": "OPENAI_API_KEY is invalid or revoked. Update it in RobBot/.env and restart kb_service.",
        }
    except Exception as e:
        return {"reply": "", "hits": [], "error": f"Chat failed: {e}"}
