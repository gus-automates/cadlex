import os
import sys
import json
import chromadb
from dotenv import load_dotenv; load_dotenv()
from google import genai
from sentence_transformers import SentenceTransformer

# ── Settings ───────────────────────────────────────────────────────────────────

MODEL_NAME    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PATH   = "./chroma_db"
COLLECTION    = "legislation"
TOP_K         = 20   # results per sub-query
MAX_PER_ACT   = 3    # max chunks from any single act in the final context
GEMINI_MODEL  = "gemini-2.5-flash-lite"   # swap to gemini-2.5-pro for production

# Key acts to always inject chunks from, regardless of global top-K ranking.
# This guarantees cross-act retrieval for the most important statutes.
KEY_NORMAS = ["CONST-charter", "ACT-C-46", "ACT-L-2"]

SYSTEM_PROMPT = """You are a legal research assistant specializing in Canadian federal legislation.

Answer based ONLY on the legal texts provided below.
Always cite the specific section and Act (e.g. "s. 7 of the Canadian Charter of Rights and Freedoms" or "s. 322 of the Criminal Code, R.S.C. 1985, c. C-46").
When the question involves multiple statutes, identify all relevant sources found in the provided texts.
If the answer is not found in the provided texts, clearly state that the information was not found in the available corpus."""

# ── Load resources (once, at startup) ─────────────────────────────────────────

print("Loading embedding model...", end=" ", flush=True)
embed_model = SentenceTransformer(MODEL_NAME)
print("OK")

print("Connecting to index...", end=" ", flush=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(COLLECTION)
print(f"OK ({collection.count():,} chunks indexed)\n")

gemini = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# ── Query decomposition ────────────────────────────────────────────────────────

def decompose_query(question: str) -> list[str]:
    """Ask Gemini to break the question into 2-4 focused search sub-queries."""
    resp = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=(
            "You are generating search terms to retrieve sections of Canadian federal legislation "
            "from a vector database.\n"
            "The sections contain text like: "
            "'7 Everyone has the right to life, liberty and security of the person...' "
            "or '322 (1) Every one commits theft who fraudulently and without colour of right takes...'\n\n"
            "For the legal question below, generate 3 to 4 short search phrases containing "
            "KEYWORDS that are likely to appear in the actual text of the relevant sections "
            "(not questions, not references to 'sections about X'). "
            "Vary the phrases to cover different statutes that may be involved.\n"
            "Return ONLY a JSON array of strings, with no explanation.\n\n"
            f"Question: {question}"
        ),
    )
    text = resp.text.strip()
    # Strip markdown code fences if present
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
    return [question]   # fallback: use original question as single query


# ── Retrieval ──────────────────────────────────────────────────────────────────

def _add_results(results, seen_ids, seen_identifier, all_docs, all_metas, bypass_cap=False):
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


def retrieve(question: str, sub_queries: list[str]) -> tuple[list, list]:
    """Run each sub-query, merge results, deduplicate by chunk id, cap per act.

    Also runs per-act vector searches for KEY_NORMAS to guarantee that the
    most important statutes are always represented in the context window,
    even when global top-K results push them out.
    """
    seen_ids:         set[str]  = set()
    seen_identifier:  dict[str, int] = {}
    all_docs: list[str]  = []
    all_metas: list[dict] = []

    # Standard vector search across the entire corpus
    for q in sub_queries:
        vec = embed_model.encode(q).tolist()
        results = collection.query(
            query_embeddings=[vec],
            n_results=TOP_K,
            include=["documents", "metadatas"],
        )
        _add_results(results, seen_ids, seen_identifier, all_docs, all_metas)

    # Per-act injection: force targeted searches within each key statute.
    # Uses the topic-focused sub-queries (not the raw question words).
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
                _add_results(r, seen_ids, seen_identifier, all_docs, all_metas, bypass_cap=True)
            except Exception:
                pass

    return all_docs, all_metas


# ── Main query function ────────────────────────────────────────────────────────

def query(question: str):
    # 1. Decompose into sub-queries
    sub_queries = decompose_query(question)
    print(f"Sub-queries: {sub_queries}\n")

    # 2. Retrieve and merge
    docs, metas = retrieve(question, sub_queries)

    # 3. Build context block
    context_parts = []
    for doc, meta in zip(docs, metas):
        context_parts.append(
            f"--- {meta['identifier']} ({meta['type']} {meta['act_id']}/{meta['year']}) ---\n{doc}"
        )
    context = "\n\n".join(context_parts)

    # 4. Build full prompt
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Retrieved legal texts:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}"
    )

    # 5. Stream answer
    unique_identifiers = list(dict.fromkeys(m["identifier"] for m in metas))
    print(f"Sources consulted: {', '.join(unique_identifiers)}\n")
    print("─" * 60)

    for chunk in gemini.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=prompt,
    ):
        if chunk.text:
            print(chunk.text, end="", flush=True)

    print("\n" + "─" * 60)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        query(question)
    else:
        print("Interactive mode. Type 'exit' to quit.\n")
        while True:
            try:
                question = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                break
            query(question)
            print()


if __name__ == "__main__":
    main()
