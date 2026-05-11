"""
One-time repair script.

What it does:
  1. Recomputes the Justice Laws full-text URL for every record using the
     standard pattern: https://laws-lois.justice.gc.ca/eng/acts/{act_id}/FullText.html
  2. Resets text = NULL for any record currently marked '[NOT FOUND]' so
     that fetch_text.py will retry them with the corrected URL.

Run this script whenever you suspect URLs are wrong or after fixing collect.py,
then re-run fetch_text.py.
"""

import sqlite3


def build_url(act_id: str) -> str:
    return f"https://laws-lois.justice.gc.ca/eng/acts/{act_id}/FullText.html"


def fix():
    conn = sqlite3.connect("legislation.db")
    cursor = conn.cursor()

    cursor.execute("SELECT id, act_id FROM legislation")
    records = cursor.fetchall()

    updated = 0
    for row_id, act_id in records:
        if not act_id:
            continue
        new_url = build_url(act_id)
        cursor.execute(
            "UPDATE legislation SET url = ? WHERE id = ?",
            (new_url, row_id),
        )
        updated += 1

    # Reset [NOT FOUND] records so fetch_text.py retries them
    cursor.execute("""
        UPDATE legislation
        SET text = NULL, collected_at = NULL
        WHERE text = '[NOT FOUND]'
    """)
    reset = cursor.rowcount

    conn.commit()
    conn.close()

    print(f"URLs recomputed: {updated}")
    print(f"[NOT FOUND] records reset to NULL: {reset}")
    print(f"\nNow run: python fetch_text.py")


if __name__ == "__main__":
    fix()
