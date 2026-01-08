"""ChromaDB RAG store with Gemini embeddings.

Uses the new google-genai SDK (unified SDK).
"""
import os
import chromadb
from google import genai
from pathlib import Path
from typing import List, Dict, Optional
import time

CHROMA_DIR = Path(__file__).parent.parent.parent / "data" / "chroma"


class RAGStore:
    """Vector store for document embeddings using ChromaDB + Gemini."""

    def __init__(self):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize Gemini client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.genai_client = genai.Client(api_key=api_key)
        self._embedding_model = "text-embedding-004"

    def _embed(self, text: str) -> List[float]:
        """Generate embedding using Gemini text-embedding-004."""
        result = self.genai_client.models.embed_content(
            model=self._embedding_model,
            contents=text
        )
        return result.embeddings[0].values

    async def add_document(self, content: str, filename: str, category: str = "general") -> str:
        """Add a document to the vector store.

        Args:
            content: Document text content
            filename: Original filename for reference
            category: Document category for filtering

        Returns:
            Document ID
        """
        # Generate unique ID
        doc_id = f"doc_{int(time.time() * 1000)}_{len(self.collection.get()['ids'])}"

        # Truncate to Gemini's limit and embed
        truncated = content[:8000]
        embedding = self._embed(truncated)

        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[truncated],
            metadatas=[{
                "filename": filename,
                "category": category,
                "chars": len(content),
                "timestamp": int(time.time())
            }]
        )
        return doc_id

    async def query(self, question: str, top_k: int = 3, category: Optional[str] = None) -> List[Dict]:
        """Query documents for relevant content.

        Args:
            question: Natural language query
            top_k: Number of results to return
            category: Optional category filter

        Returns:
            List of matching documents with scores
        """
        if self.collection.count() == 0:
            return []

        q_embedding = self._embed(question)

        where_filter = {"category": category} if category else None
        results = self.collection.query(
            query_embeddings=[q_embedding],
            n_results=min(top_k, self.collection.count()),
            where=where_filter
        )

        if not results['ids'] or not results['ids'][0]:
            return []

        return [
            {
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "score": 1 - results['distances'][0][i]  # cosine distance to similarity
            }
            for i in range(len(results['ids'][0]))
        ]

    def get_stats(self) -> Dict:
        """Get storage statistics."""
        return {
            "total_documents": self.collection.count(),
            "storage_path": str(CHROMA_DIR)
        }

    def list_documents(self, limit: int = 50) -> List[Dict]:
        """List all documents in the store."""
        results = self.collection.get(limit=limit)
        if not results['ids']:
            return []
        return [
            {
                "id": results['ids'][i],
                "metadata": results['metadatas'][i] if results['metadatas'] else {}
            }
            for i in range(len(results['ids']))
        ]


_store: Optional[RAGStore] = None


def get_rag_store() -> RAGStore:
    """Get singleton RAGStore instance."""
    global _store
    if _store is None:
        _store = RAGStore()
    return _store
