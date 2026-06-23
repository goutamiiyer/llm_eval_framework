import sqlite3
from datetime import datetime

DB_PATH = "eval_results.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            timestamp TEXT,
            prompt TEXT,
            expected TEXT,
            response TEXT,
            evaluator TEXT,
            score REAL,
            passed INTEGER,
            reason TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_results(run_id: str, results: list):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    
    for r in results:
        cursor.execute("""
            INSERT INTO results 
            (run_id, timestamp, prompt, expected, response, evaluator, score, passed, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            timestamp,
            r["prompt"],
            r["expected"],
            r["response"],
            r["evaluator"],
            r["score"],
            1 if r["passed"] else 0,
            r.get("reason", "")
        ))
    
    conn.commit()
    conn.close()

def get_last_runs(n: int = 5):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT run_id, timestamp, evaluator, 
               AVG(score) as avg_score, COUNT(*) as total
        FROM results
        GROUP BY run_id, evaluator
        ORDER BY timestamp DESC
        LIMIT ?
    """, (n,))
    rows = cursor.fetchall()
    conn.close()
    return rows