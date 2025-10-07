# run once: python scripts/init_helpers_db.py
import sqlite3, os, json

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "DisasterResponse.db"))

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# create helpers table if not exists
cur.execute("""
CREATE TABLE IF NOT EXISTS helpers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  email TEXT,
  phone TEXT,
  lat REAL,
  lon REAL,
  radius_km REAL DEFAULT 10,
  active INTEGER DEFAULT 1
)
""")

# create reports table if not exists
cur.execute("""
CREATE TABLE IF NOT EXISTS reports (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  message TEXT,
  raw_lat REAL,
  raw_lon REAL,
  resolved_lat REAL,
  resolved_lon REAL,
  location_text TEXT,
  urgency REAL,
  is_emergency INTEGER,
  created_at TEXT,
  assigned_helpers TEXT
)
""")

# insert sample helpers (change emails to real test addresses if you want)
sample = [
    ('Helper A','helpera@example.com',None,9.001,76.905,15,1),
    ('Helper B','helperb@example.com',None,9.010,76.920,10,1),
    ('Rescue Team','rescueteam@example.com',None,9.03,76.89,25,1),
    ('Volunteer X','volx@example.com',None,8.95,76.95,20,1)
]

# avoid duplicating on re-run: check if table empty
cur.execute("SELECT COUNT(*) FROM helpers")
count = cur.fetchone()[0]
if count == 0:
    cur.executemany("INSERT INTO helpers (name,email,phone,lat,lon,radius_km,active) VALUES (?,?,?,?,?,?,?)", sample)
    print("Inserted sample helpers.")
else:
    print("Helpers table already has rows; skipping insert.")

conn.commit()
conn.close()
print("DB initialized at:", DB_PATH)
