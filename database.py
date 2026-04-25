# database.py
import sqlite3
import json
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List

class PaperDatabase:
    def __init__(self, db_path="papers.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    arxiv_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors TEXT,
                    abstract TEXT,
                    pdf_path TEXT,
                    summaries_json TEXT,      -- from ReaderAgent
                    extracted_json TEXT,      -- from KnowledgeExtractor
                    embedding BLOB,           -- numpy array as bytes
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Optional: index on title for faster search
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_title ON papers(title)")
            conn.commit()

    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Convert numpy array to bytes for storage."""
        return embedding.tobytes()

    def _deserialize_embedding(self, blob: bytes) -> np.ndarray:
        """Convert bytes back to numpy array."""
        return np.frombuffer(blob, dtype=np.float32)

    def add_paper(self, arxiv_id: str, title: str, authors: List[str],
                  abstract: str, pdf_path: str, summaries: Dict[str, Any],
                  extracted: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> bool:
        """
        Insert or replace a paper record.
        Returns True if successful.
        """
        try:
            authors_json = json.dumps(authors)
            summaries_json = json.dumps(summaries)
            extracted_json = json.dumps(extracted)
            embedding_blob = self._serialize_embedding(embedding) if embedding is not None else None

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO papers
                    (arxiv_id, title, authors, abstract, pdf_path, summaries_json, extracted_json, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (arxiv_id, title, authors_json, abstract, pdf_path, summaries_json, extracted_json, embedding_blob))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error adding paper {title}: {e}")
            return False

    def get_paper(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single paper by its arXiv ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,))
            row = cursor.fetchone()
            if not row:
                return None
            paper = dict(row)
            paper['authors'] = json.loads(paper['authors'])
            paper['summaries'] = json.loads(paper['summaries_json'])
            paper['extracted'] = json.loads(paper['extracted_json'])
            if paper['embedding']:
                paper['embedding'] = self._deserialize_embedding(paper['embedding'])
            else:
                paper['embedding'] = None
            # Remove the raw JSON fields to keep clean
            del paper['summaries_json']
            del paper['extracted_json']
            return paper

    def get_all_papers(self) -> List[Dict[str, Any]]:
        """Retrieve all papers (without embedding to save memory)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT arxiv_id, title, authors, abstract, pdf_path, summaries_json, extracted_json, created_at FROM papers")
            rows = cursor.fetchall()
            papers = []
            for row in rows:
                paper = dict(row)
                paper['authors'] = json.loads(paper['authors'])
                paper['summaries'] = json.loads(paper['summaries_json'])
                paper['extracted'] = json.loads(paper['extracted_json'])
                del paper['summaries_json']
                del paper['extracted_json']
                papers.append(paper)
            return papers

    def update_extracted(self, arxiv_id: str, extracted: Dict[str, Any]) -> bool:
        """Update only the extracted knowledge for a paper."""
        try:
            extracted_json = json.dumps(extracted)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE papers SET extracted_json = ? WHERE arxiv_id = ?", (extracted_json, arxiv_id))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error updating extracted for {arxiv_id}: {e}")
            return False

    def update_embedding(self, arxiv_id: str, embedding: np.ndarray) -> bool:
        """Update the embedding for a paper."""
        try:
            embedding_blob = self._serialize_embedding(embedding)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE papers SET embedding = ? WHERE arxiv_id = ?", (embedding_blob, arxiv_id))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error updating embedding for {arxiv_id}: {e}")
            return False

    def paper_exists(self, arxiv_id: str) -> bool:
        """Check if a paper is already in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM papers WHERE arxiv_id = ?", (arxiv_id,))
            return cursor.fetchone() is not None

    def delete_paper(self, arxiv_id: str) -> bool:
        """Remove a paper from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM papers WHERE arxiv_id = ?", (arxiv_id,))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting paper {arxiv_id}: {e}")
            return False

    def get_papers_by_topic(self, keyword: str) -> List[Dict[str, Any]]:
        """Search papers by title or abstract keyword."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT arxiv_id, title, authors, abstract, pdf_path, created_at
                FROM papers
                WHERE title LIKE ? OR abstract LIKE ?
                ORDER BY created_at DESC
            """, (f'%{keyword}%', f'%{keyword}%'))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]