# vector_store.py
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from sentence_transformers import SentenceTransformer
from database import PaperDatabase
from typing import List, Dict, Any, Optional, Union

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", db: Optional[PaperDatabase] = None):
        """
        Initialize the embedding model and connect to the database.
        model_name: any sentence-transformers model (small/fast recommended).
        db: PaperDatabase instance (if None, creates a new one).
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.db = db if db is not None else PaperDatabase()
        print("Vector store ready.")

    def _get_text_for_embedding(self, paper: Dict[str, Any]) -> str:
        """
        Combine abstract, methods, and key findings into a single text for embedding.
        paper: dict from database (must contain 'abstract' and 'extracted').
        """
        abstract = paper.get('abstract', '')
        extracted = paper.get('extracted', {})
        methods = ' '.join(extracted.get('methods', []))
        findings = ' '.join(extracted.get('key_findings', []))
        text = f"{abstract}\nMethods: {methods}\nFindings: {findings}"
        return text.strip() or paper.get('title', '')

    def add_paper(self, paper: Dict[str, Any]) -> bool:
        """
        Generate embedding for a paper and store it in the database.
        paper: dict with at least 'arxiv_id', 'abstract', 'extracted', and optionally 'title'.
        Returns True if successful.
        """
        arxiv_id = paper.get('arxiv_id')
        if not arxiv_id:
            print("Warning: paper missing arxiv_id, cannot store embedding.")
            return False
        text = self._get_text_for_embedding(paper)
        embedding = self.model.encode(text, normalize_embeddings=True)  # unit vector for cosine similarity
        success = self.db.update_embedding(arxiv_id, embedding)
        if success:
            print(f"Stored embedding for paper: {paper.get('title', arxiv_id)[:50]}")
        else:
            print(f"Failed to store embedding for {arxiv_id}")
        return success

    def add_all_unembedded(self) -> int:
        """
        Find papers in the database that don't have an embedding and add them.
        Returns number of papers updated.
        """
        all_papers = self.db.get_all_papers()
        count = 0
        for paper in all_papers:
            if paper.get('embedding') is None:
                if self.add_paper(paper):
                    count += 1
        print(f"Added embeddings for {count} papers.")
        return count

    def similar_papers(self, query: Union[str, Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find papers similar to a query string or to a given paper dict.
        If query is a string, embed it directly.
        If query is a paper dict (from db), use its stored embedding if available, else generate.
        Returns list of papers with an added 'similarity' score.
        """
        # Get the query embedding
        if isinstance(query, dict):
            # Query is a paper
            embedding = query.get('embedding')
            if embedding is None:
                # Generate on the fly
                text = self._get_text_for_embedding(query)
                query_emb = self.model.encode(text, normalize_embeddings=True)
            else:
                query_emb = embedding
        else:
            # Query is a string
            query_emb = self.model.encode(query, normalize_embeddings=True)

        # Retrieve all papers with embeddings from database
        all_papers = self.db.get_all_papers()
        similarities = []
        for paper in all_papers:
            emb = paper.get('embedding')
            if emb is None:
                continue
            # Cosine similarity = dot product because vectors are normalized
            sim = np.dot(query_emb, emb)
            similarities.append((paper, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        top = []
        for paper, sim in similarities[:top_k]:
            paper_copy = paper.copy()
            paper_copy['similarity'] = float(sim)
            top.append(paper_copy)
        return top

    def find_by_text(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Convenience method: find papers similar to a free-text query."""
        return self.similar_papers(text, top_k)

    def find_similar_to_paper(self, paper: Dict[str, Any], top_k: int = 5, exclude_self: bool = True) -> List[Dict[str, Any]]:
        """
        Find papers similar to a given paper, optionally excluding the paper itself.
        """
        results = self.similar_papers(paper, top_k + (1 if exclude_self else 0))
        if exclude_self and results:
            arxiv_id = paper.get('arxiv_id')
            results = [r for r in results if r.get('arxiv_id') != arxiv_id]
        return results[:top_k]