import re
import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer

# ── Settings ───────────────────────────────────────────────────────────────────

MODEL_NAME  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PATH = "./chroma_db"
COLLECTION  = "legislation"
BATCH_SIZE  = 64
MAX_CHUNK   = 2000   # chars — split oversized sections further at subdivision boundaries

# Identifiers to force re-index even if already present (deletes old chunks first).
# Set to [] to skip. Use this after fixing the chunking logic for specific acts
# without running a full re-index.
FORCE_REINDEX = []
# e.g. ["CONST-charter", "ACT-C-46", "ACT-L-2"]

# ── Chunking ───────────────────────────────────────────────────────────────────

# Canadian federal legislation uses numbered sections: "1 Short title", "2 Definitions", etc.
# Sections may start after a blank line or at the beginning of a line.
SECTION_PATTERN = re.compile(r"(?m)(?:^|\n)(\s*\d+(?:\.\d+)?\s+[A-Z(])")

# Secondary split boundaries within an oversized section
SUBSECTION_PATTERN = re.compile(r"(?m)(?=^\s*\(\d+\)\s|^\s*\([a-z]\)\s)")

# Division and Part headers used as additional boundary hints
DIVISION_PATTERN = re.compile(
    r"(?m)(?=^\s*(?:PART\s+[IVXLC]+|DIVISION\s+[IVXLC\d]+|SCHEDULE\b))",
    re.IGNORECASE,
)


def chunk(text: str, identifier: str, summary: str) -> list[str]:
    """Split an act's full text into section-level chunks with a metadata header."""
    header = f"[{identifier}] {summary}\n\n"

    # Primary split on numbered sections
    parts = SECTION_PATTERN.split(text)

    # SECTION_PATTERN uses a capturing group, so split() returns
    # [pre-text, section_num+first_word, rest, section_num+first_word, rest, ...]
    # We reassemble: each chunk = section_label + rest
    assembled: list[str] = []

    # The first element is text before the first numbered section (preamble / title block)
    preamble = parts[0].strip()
    if preamble:
        assembled.append(preamble)

    i = 1
    while i < len(parts) - 1:
        label = parts[i]        # e.g. "1 Short "
        body  = parts[i + 1]   # text until next section
        assembled.append((label + body).strip())
        i += 2

    if not assembled:
        # Fallback: no section numbers found — treat whole text as one chunk
        assembled = [text.strip()]

    chunks: list[str] = []

    for part in assembled:
        part = part.strip()
        if len(part) < 40:
            continue

        if len(part) <= MAX_CHUNK:
            chunks.append(header + part)
        else:
            # Oversized section — try to split at subsection / paragraph boundaries
            sec_match = re.match(r"^(\d+(?:\.\d+)?)\s", part)
            sec_label = f"s. {sec_match.group(1)}" if sec_match else ""

            # Try subsection split first, fall back to division split
            subs = SUBSECTION_PATTERN.split(part)
            if len(subs) <= 1:
                subs = DIVISION_PATTERN.split(part)

            buf = ""
            for s in subs:
                if len(buf) + len(s) > MAX_CHUNK and buf:
                    content = buf.strip()
                    if sec_label and not re.match(r"^\d", content):
                        content = f"{sec_label} [cont.]\n{content}"
                    chunks.append(header + content)
                    buf = s
                else:
                    buf += "\n" + s
            if buf.strip():
                content = buf.strip()
                if sec_label and not re.match(r"^\d", content):
                    content = f"{sec_label} [cont.]\n{content}"
                chunks.append(header + content)

    return chunks


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Load embedding model (downloads ~90 MB on first run)
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Delete and re-index specific acts (FORCE_REINDEX) before building the existing set.
    # This lets us re-chunk specific laws after a logic fix without a full re-index.
    if FORCE_REINDEX:
        for identifier in FORCE_REINDEX:
            try:
                ids_to_delete = collection.get(
                    where={"identifier": identifier}, include=[]
                )["ids"]
                if ids_to_delete:
                    collection.delete(ids=ids_to_delete)
                    print(f"Deleted {len(ids_to_delete)} old chunks for {identifier}")
            except Exception as e:
                print(f"Could not delete {identifier}: {e}")

    # Find already-indexed identifiers to allow resuming (paginated to avoid limits)
    existing: set[str] = set()
    total_chunks = collection.count()
    if total_chunks > 0:
        PAGE = 5000
        offset = 0
        while offset < total_chunks:
            batch = collection.get(include=["metadatas"], limit=PAGE, offset=offset)
            for m in batch["metadatas"]:
                existing.add(m["identifier"])
            offset += PAGE
        print(f"Already indexed: {len(existing)} acts — will skip these")

    # Load acts from DB
    conn = sqlite3.connect("legislation.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT identifier, type, act_id, year, summary, text
        FROM legislation
        WHERE text IS NOT NULL
          AND text != '[NOT FOUND]'
        ORDER BY year, identifier
    """)
    rows = cursor.fetchall()
    conn.close()

    total = len(rows)
    print(f"\n{total} acts to process\n")

    chunk_buf_ids:   list[str]  = []
    chunk_buf_texts: list[str]  = []
    chunk_buf_meta:  list[dict] = []
    indexed_acts = 0

    def flush():
        if not chunk_buf_texts:
            return
        embeddings = model.encode(
            chunk_buf_texts, batch_size=BATCH_SIZE, show_progress_bar=False
        ).tolist()
        CHROMA_MAX = 5000
        for i in range(0, len(chunk_buf_texts), CHROMA_MAX):
            collection.add(
                ids=chunk_buf_ids[i:i + CHROMA_MAX],
                embeddings=embeddings[i:i + CHROMA_MAX],
                documents=chunk_buf_texts[i:i + CHROMA_MAX],
                metadatas=chunk_buf_meta[i:i + CHROMA_MAX],
            )
        chunk_buf_ids.clear()
        chunk_buf_texts.clear()
        chunk_buf_meta.clear()

    for i, (identifier, act_type, act_id, year, summary, text) in enumerate(rows, 1):
        if identifier in existing:
            continue

        chunks = chunk(text, identifier, summary or "")
        if not chunks:
            continue

        for n, chunk_text in enumerate(chunks):
            chunk_buf_ids.append(f"{identifier}_sec_{n}")
            chunk_buf_texts.append(chunk_text)
            chunk_buf_meta.append({
                "identifier": identifier,
                "type":       act_type,
                "act_id":     act_id or "",
                "year":       year or 0,
                "summary":    (summary or "")[:200],
            })

        indexed_acts += 1

        # Print progress every 25 acts
        if indexed_acts % 25 == 0:
            print(f"  [{i}/{total}] {identifier} — buffer: {len(chunk_buf_texts)} chunks", flush=True)

        # Flush every 500 acts to keep memory manageable
        if indexed_acts % 500 == 0:
            flush()
            print(f"  Flushed to ChromaDB — {collection.count()} chunks stored total", flush=True)

    flush()
    print(f"\nDone. Total chunks in index: {collection.count()}")


if __name__ == "__main__":
    main()
