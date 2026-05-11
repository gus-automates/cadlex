import re
import sqlite3
import requests
from datetime import datetime
from xml.etree import ElementTree as ET

# ── Settings ──────────────────────────────────────────────────────────────────

LOOKUP_XML = "https://laws-lois.justice.gc.ca/js/lookup_acts_e.xml"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-CA,en;q=0.9",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_url(act_id: str) -> str:
    return f"https://laws-lois.justice.gc.ca/eng/acts/{act_id}/FullText.html"


def extract_year(chapter: str) -> int | None:
    """Extract the most recent 4-digit year from the chapter reference.
    e.g. 'R.S.C., 1985, c. C-46' -> 1985, '2019, c. 10' -> 2019
    """
    years = re.findall(r"\b(1[89]\d{2}|20[0-2]\d)\b", chapter or "")
    return int(years[-1]) if years else None


def build_identifier(act_id: str) -> str:
    return f"ACT-{act_id}"


# ── Collection ────────────────────────────────────────────────────────────────

def collect():
    print("Fetching acts list from Justice Laws XML lookup...")

    resp = requests.get(LOOKUP_XML, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        print(f"  ERROR: HTTP {resp.status_code}")
        return

    # Parse XML (strip UTF-8 BOM if present)
    content = resp.content.decode("utf-8-sig")
    root = ET.fromstring(content)

    # Only include active acts (t="a", no rep="true")
    entries = [
        d for d in root.findall("D")
        if d.get("t") == "a" and d.get("rep") != "true"
    ]
    print(f"  Found {len(entries)} active acts in XML")

    conn = sqlite3.connect("legislation.db")
    cursor = conn.cursor()

    saved = 0
    skipped = 0

    for d in entries:
        act_id  = d.find("C").text.strip()
        chapter = d.find("OC").text.strip()
        title   = d.find("T").text.strip()

        identifier = build_identifier(act_id)
        url        = build_url(act_id)
        year       = extract_year(chapter)

        try:
            cursor.execute("""
                INSERT INTO legislation
                    (act_id, type, title, chapter, year, identifier, summary, url, text, collected_at)
                VALUES (?, 'ACT', ?, ?, ?, ?, ?, ?, NULL, ?)
            """, (
                act_id,
                title,
                chapter,
                year,
                identifier,
                title,  # use title as summary until richer data is available
                url,
                datetime.now().isoformat(),
            ))
            saved += 1

        except sqlite3.IntegrityError:
            skipped += 1

    conn.commit()
    conn.close()

    print(f"  Saved: {saved} | Skipped (already in DB): {skipped}")
    print(f"\nDone! Run 'python fetch_text.py' to download full text for each act.")


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    collect()
