"""
CadLex — FastAPI backend
Run with: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import json
import os
import sqlite3
from typing import Iterator

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
DB_PATH        = "./legislation.db"
CHROMA_PATH    = "./chroma_db"
COLLECTION     = "legislation"
EMBED_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL   = "gemini-2.5-flash-lite"
TOP_K          = 20
MAX_PER_ACT    = 3
KEY_NORMAS     = ["CONST-charter", "ACT-C-46", "ACT-L-2"]

SYSTEM_PROMPT = """You are a legal research assistant specializing in Canadian federal legislation.

Answer based ONLY on the legal texts provided below.
Always cite the specific section and Act (e.g. "s. 7 of the Canadian Charter of Rights and Freedoms" or "s. 322 of the Criminal Code, R.S.C. 1985, c. C-46").
When the question involves multiple statutes, identify all relevant sources found in the provided texts.
If the answer is not found in the provided texts, clearly state that the information was not found in the available corpus."""

# ── Lazy-loaded globals ────────────────────────────────────────────────────────
_embed_model = None
_collection  = None
_gemini      = None


def load_models():
    global _embed_model, _collection, _gemini
    if _embed_model is not None:
        return _embed_model, _collection, _gemini

    from google import genai
    from sentence_transformers import SentenceTransformer
    import chromadb

    print("Loading embedding model...", flush=True)
    _embed_model = SentenceTransformer(EMBED_MODEL)
    print("Connecting to ChromaDB...", flush=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    _collection = client.get_collection(COLLECTION)
    _gemini = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    print(f"Ready — {_collection.count():,} chunks indexed", flush=True)
    return _embed_model, _collection, _gemini


# ── RAG pipeline (sync — runs in the response thread) ─────────────────────────

def _decompose_query(gemini_client, question: str) -> list[str]:
    resp = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=(
            "You are generating search terms to retrieve sections of Canadian federal legislation "
            "from a vector database.\n"
            "The sections contain text like: "
            "'7 Everyone has the right to life, liberty and security of the person...' "
            "or '322 (1) Every one commits theft who fraudulently and without colour of right takes...'\n\n"
            "For the legal question below, generate 3 to 4 short search phrases containing "
            "KEYWORDS that are likely to appear in the actual text of the relevant sections. "
            "Vary the phrases to cover different statutes that may be involved.\n"
            "Return ONLY a JSON array of strings, with no explanation.\n\n"
            f"Question: {question}"
        ),
    )
    text = resp.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        queries = json.loads(text)
        if isinstance(queries, list) and queries:
            return [str(q) for q in queries]
    except Exception:
        pass
    return [question]


def _retrieve(embed_model, collection, sub_queries: list[str]) -> tuple[list, list]:
    seen_ids:        set  = set()
    seen_identifier: dict = {}
    all_docs, all_metas = [], []

    def add(results, bypass_cap=False):
        for chunk_id, doc, meta in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0]
        ):
            if chunk_id in seen_ids:
                continue
            identifier = meta["identifier"]
            if not bypass_cap and seen_identifier.get(identifier, 0) >= MAX_PER_ACT:
                continue
            seen_ids.add(chunk_id)
            seen_identifier[identifier] = seen_identifier.get(identifier, 0) + 1
            all_docs.append(doc)
            all_metas.append(meta)

    for q in sub_queries:
        vec = embed_model.encode(q).tolist()
        results = collection.query(
            query_embeddings=[vec], n_results=TOP_K, include=["documents", "metadatas"]
        )
        add(results)

    for q in sub_queries:
        vec = embed_model.encode(q).tolist()
        for identifier in KEY_NORMAS:
            try:
                r = collection.query(
                    query_embeddings=[vec],
                    n_results=5,
                    where={"identifier": identifier},
                    include=["documents", "metadatas"],
                )
                add(r, bypass_cap=True)
            except Exception:
                pass

    return all_docs, all_metas


def _stream_rag(question: str) -> Iterator[str]:
    embed_model, collection, gemini_client = load_models()

    # 1. Query decomposition
    sub_queries = _decompose_query(gemini_client, question)
    yield f"data: {json.dumps({'type': 'subqueries', 'queries': sub_queries})}\n\n"

    # 2. Retrieval
    docs, metas = _retrieve(embed_model, collection, sub_queries)

    # 3. Build context block
    context_parts = []
    for doc, meta in zip(docs, metas):
        context_parts.append(
            f"--- {meta['identifier']} ({meta['type']} {meta['act_id']}/{meta['year']}) ---\n{doc}"
        )
    context = "\n\n".join(context_parts)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Retrieved legal texts:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}"
    )

    # 4. Stream answer tokens
    for chunk in gemini_client.models.generate_content_stream(
        model=GEMINI_MODEL, contents=prompt
    ):
        if chunk.text:
            yield f"data: {json.dumps({'type': 'token', 'text': chunk.text})}\n\n"

    # 5. Emit unique sources
    seen_identifiers: dict = {}
    for meta in metas:
        ident = meta["identifier"]
        if ident not in seen_identifiers:
            seen_identifiers[ident] = {
                "identifier": meta["identifier"],
                "type":       meta["type"],
                "act_id":     meta["act_id"],
                "year":       meta["year"],
                "summary":    meta.get("summary", ""),
            }
    yield f"data: {json.dumps({'type': 'sources', 'normas': list(seen_identifiers.values())})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="CadLex API")

_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ORIGINS,   # set ALLOWED_ORIGINS=https://yourdomain.com in production
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        _stream_rag(req.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/norma/{identifier:path}")
async def get_norma(identifier: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT type, act_id, year, identifier, summary, title, url, text FROM legislation WHERE identifier = ?",
        (identifier,),
    ).fetchone()
    conn.close()
    if not row:
        return JSONResponse({"error": "Record not found"}, status_code=404)
    return dict(row)


@app.get("/api/search")
async def search(
    q:    str = Query(..., min_length=1),
    tipo: str = Query("keyword"),
):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    q_clean = q.strip()

    if tipo == "numero":
        # Search by act_id (e.g. "C-46" or "A-1")
        rows = conn.execute(
            """SELECT type, act_id, year, identifier, summary, title
               FROM legislation
               WHERE act_id = ?
                 AND text IS NOT NULL AND text != '[NOT FOUND]'
               ORDER BY year DESC LIMIT 30""",
            (q_clean,),
        ).fetchall()

    elif tipo == "artigo":
        # Search for a specific section number in the full text
        rows = conn.execute(
            """SELECT type, act_id, year, identifier, summary, title
               FROM legislation
               WHERE (text LIKE ? OR text LIKE ?)
                 AND text IS NOT NULL AND text != '[NOT FOUND]'
               ORDER BY year DESC LIMIT 30""",
            (f"% {q_clean} %", f"%s. {q_clean}%"),
        ).fetchall()

    else:  # keyword — search summary and title first (fast), fall back to full text
        rows = conn.execute(
            """SELECT type, act_id, year, identifier, summary, title
               FROM legislation
               WHERE (summary LIKE ? OR title LIKE ?)
                 AND text IS NOT NULL AND text != '[NOT FOUND]'
               ORDER BY year DESC LIMIT 30""",
            (f"%{q_clean}%", f"%{q_clean}%"),
        ).fetchall()
        if not rows:
            rows = conn.execute(
                """SELECT type, act_id, year, identifier, summary, title
                   FROM legislation
                   WHERE text LIKE ?
                     AND text IS NOT NULL AND text != '[NOT FOUND]'
                   ORDER BY year DESC LIMIT 30""",
                (f"%{q_clean}%",),
            ).fetchall()

    conn.close()
    return [dict(r) for r in rows]


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
