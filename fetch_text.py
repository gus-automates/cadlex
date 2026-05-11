import sqlite3
import requests
import time
from datetime import datetime
from html.parser import HTMLParser

# ── HTML cleaner ───────────────────────────────────────────────────────────────
# Extracts only the visible text from the HTML page,
# ignoring scripts, styles, and markup tags.

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._texts = []
        self._skip = False    # flag to ignore script/style content

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            text = data.strip()
            if text:
                self._texts.append(text)

    def get_text(self):
        return "\n".join(self._texts)


def extract_text_from_html(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    return parser.get_text()


# ── Fetch one URL ──────────────────────────────────────────────────────────────

HEADERS = {
    # Impersonate a browser so the server does not block the request
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-CA,en;q=0.9",
}


def fetch_text(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)

        if resp.status_code == 200:
            # Justice Laws is UTF-8; force correct decoding
            resp.encoding = "utf-8"
            return extract_text_from_html(resp.text)
        else:
            return None

    except Exception as e:
        print(f"    WARNING: {e}")
        return None


# ── Main loop ──────────────────────────────────────────────────────────────────

def fetch_all():
    conn = sqlite3.connect("legislation.db")
    cursor = conn.cursor()

    # Rows that have not yet been fetched and have a URL to use
    cursor.execute("""
        SELECT id, identifier, url
        FROM legislation
        WHERE text IS NULL AND url IS NOT NULL
        ORDER BY year DESC, id ASC
    """)
    pending = cursor.fetchall()

    total = len(pending)
    print(f"{total} acts need full text fetching\n")

    for i, (row_id, identifier, url) in enumerate(pending, 1):
        print(f"  [{i}/{total}] {identifier}")
        print(f"    -> {url}")

        text = fetch_text(url)

        if text:
            cursor.execute("""
                UPDATE legislation
                SET text = ?, collected_at = ?
                WHERE id = ?
            """, (text, datetime.now().isoformat(), row_id))
            conn.commit()
            print(f"    OK  {len(text):,} characters saved")
        else:
            # Mark as attempted but failed so we don't retry endlessly.
            # Run fix_urls.py followed by this script again to retry.
            cursor.execute("""
                UPDATE legislation
                SET text = '[NOT FOUND]', collected_at = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), row_id))
            conn.commit()
            print(f"    FAILED — marked as [NOT FOUND]")

        # 0.5-second rate limit — be polite to the Justice Laws server
        time.sleep(0.5)

    conn.close()
    print(f"\nDone!")


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fetch_all()
