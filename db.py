import sqlite3
from typing import Optional


def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str):
    conn = get_conn(db_path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        type TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS relationships (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity1_id INTEGER NOT NULL,
        entity2_id INTEGER NOT NULL,
        relationship TEXT,
        context TEXT,
        FOREIGN KEY(entity1_id) REFERENCES entities(id) ON DELETE CASCADE,
        FOREIGN KEY(entity2_id) REFERENCES entities(id) ON DELETE CASCADE
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS dialogues (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        speaker_id INTEGER,
        dialogue TEXT,
        context TEXT,
        FOREIGN KEY(speaker_id) REFERENCES entities(id) ON DELETE SET NULL
    )
    """)

    conn.commit()
    conn.close()


def get_or_create_entity(conn: sqlite3.Connection, name: str, type: Optional[str] = None) -> int:
    c = conn.cursor()
    c.execute("SELECT id FROM entities WHERE name = ?", (name,))
    row = c.fetchone()
    if row:
        return row[0]
    c.execute("INSERT INTO entities (name, type) VALUES (?, ?)", (name, type))
    conn.commit()
    return c.lastrowid


def insert_relationship(conn: sqlite3.Connection, entity1: str, entity2: str, relationship: str, context: str = ""):
    e1_id = get_or_create_entity(conn, entity1)
    e2_id = get_or_create_entity(conn, entity2)
    c = conn.cursor()
    c.execute(
        "INSERT INTO relationships (entity1_id, entity2_id, relationship, context) VALUES (?,?,?,?)",
        (e1_id, e2_id, relationship, context),
    )
    conn.commit()


def insert_dialogue(conn: sqlite3.Connection, speaker: Optional[str], dialogue: str, context: str = ""):
    speaker_id = None
    if speaker:
        speaker_id = get_or_create_entity(conn, speaker)
    c = conn.cursor()
    c.execute(
        "INSERT INTO dialogues (speaker_id, dialogue, context) VALUES (?,?,?)",
        (speaker_id, dialogue, context),
    )
    conn.commit()


def query_relationships(conn: sqlite3.Connection, entity_name: Optional[str] = None):
    c = conn.cursor()
    if entity_name:
        c.execute(
            """
            SELECT r.id, e1.name, e2.name, r.relationship, r.context
            FROM relationships r
            JOIN entities e1 ON r.entity1_id = e1.id
            JOIN entities e2 ON r.entity2_id = e2.id
            WHERE e1.name = ? OR e2.name = ?
            """,
            (entity_name, entity_name),
        )
    else:
        c.execute(
            """
            SELECT r.id, e1.name, e2.name, r.relationship, r.context
            FROM relationships r
            JOIN entities e1 ON r.entity1_id = e1.id
            JOIN entities e2 ON r.entity2_id = e2.id
            """
        )
    return c.fetchall()


def query_dialogues(conn: sqlite3.Connection, speaker_name: Optional[str] = None):
    c = conn.cursor()
    if speaker_name:
        c.execute(
            """
            SELECT d.id, e.name, d.dialogue, d.context
            FROM dialogues d
            LEFT JOIN entities e ON d.speaker_id = e.id
            WHERE e.name = ?
            """,
            (speaker_name,)
        )
    else:
        c.execute(
            """
            SELECT d.id, e.name, d.dialogue, d.context
            FROM dialogues d
            LEFT JOIN entities e ON d.speaker_id = e.id
            """
        )
    return c.fetchall()
