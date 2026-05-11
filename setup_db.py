import sqlite3

# Creates 'legislation.db' if it doesn't exist yet.
# If it already exists, this script is a no-op (safe to re-run).
conn = sqlite3.connect("legislation.db")

cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS legislation (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        act_id       TEXT,        -- Justice Laws alphanumeric ID, e.g. C-46
        type         TEXT,        -- ACT, REG, CONST
        title        TEXT,        -- official short title
        chapter      TEXT,        -- e.g. R.S.C., 1985, c. C-46
        year         INTEGER,     -- year extracted from the title/chapter
        identifier   TEXT UNIQUE, -- e.g. ACT-C-46 (no duplicates)
        summary      TEXT,        -- short description
        url          TEXT,        -- full-text URL on laws-lois.justice.gc.ca
        text         TEXT,        -- full text — NULL=not fetched, [NOT FOUND]=failed
        collected_at TEXT         -- ISO timestamp of when the text was fetched
    )
""")

conn.commit()
conn.close()

print("Database created successfully!")
print("File 'legislation.db' is now in your folder.")
