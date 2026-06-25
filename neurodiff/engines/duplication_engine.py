"""Code duplication detection engine."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DuplicationFinding:
    """Represents a code duplication finding."""

    source_file: str
    target_file: str
    source_snippet: str
    target_snippet: str
    similarity: float
    severity: str


class DuplicationEngine:
    """Engine for detecting code duplication using ChromaDB and embeddings."""

    SIMILARITY_THRESHOLD = 0.80
    CHROMA_DB_PATH = Path.home() / ".neurodiff" / "chroma_db"

    def __init__(self) -> None:
        """Initialize the DuplicationEngine."""
        self.has_chromadb = self._check_chromadb()
        self.client = None
        self.collection = None

        if self.has_chromadb:
            self._initialize_chromadb()

    def _check_chromadb(self) -> bool:
        """Check if ChromaDB is available.

        Returns:
            True if ChromaDB is installed and available.
        """
        try:
            import chromadb

            return True
        except ImportError:
            return False

    def _initialize_chromadb(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb

            # Create database directory if it doesn't exist
            self.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=str(self.CHROMA_DB_PATH))

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="code_snippets",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            # Graceful degradation
            self.has_chromadb = False

    def analyze(
        self, snippets: list[tuple[str, str]]
    ) -> list[DuplicationFinding]:
        """Analyze snippets for duplication.

        Args:
            snippets: List of (file_path, code_snippet) tuples.

        Returns:
            List of duplication findings.
        """
        findings: list[DuplicationFinding] = []

        if not self.has_chromadb or not self.collection:
            return findings

        try:
            findings.extend(self._find_duplicates_with_chromadb(snippets))
        except Exception:
            # Graceful degradation
            pass

        return findings

    def _find_duplicates_with_chromadb(
        self, snippets: list[tuple[str, str]]
    ) -> list[DuplicationFinding]:
        """Find duplicate code snippets using ChromaDB.

        Args:
            snippets: List of (file_path, code_snippet) tuples.

        Returns:
            List of duplication findings.
        """
        findings: list[DuplicationFinding] = []

        try:
            from sentence_transformers import SentenceTransformer

            # Load embedding model
            model = SentenceTransformer("all-MiniLM-L6-v2")

            # Index current snippets
            for idx, (file_path, snippet) in enumerate(snippets):
                if not snippet.strip():
                    continue

                # Generate embedding
                embedding = model.encode(snippet).tolist()

                # Add to collection
                self.collection.add(
                    ids=[f"{file_path}_{idx}"],
                    embeddings=[embedding],
                    documents=[snippet],
                    metadatas=[
                        {
                            "file_path": file_path,
                            "snippet": snippet[:100],  # Store preview
                        }
                    ],
                )

            # Query for similar snippets
            for idx, (file_path, snippet) in enumerate(snippets):
                if not snippet.strip():
                    continue

                embedding = model.encode(snippet).tolist()

                # Find similar snippets
                results = self.collection.query(
                    query_embeddings=[embedding],
                    n_results=5,  # Get top 5 similar snippets
                )

                if results and results["ids"] and len(results["ids"]) > 0:
                    for result_idx, (result_id, distance) in enumerate(
                        zip(results["ids"][0], results["distances"][0])
                    ):
                        # Skip the exact same snippet
                        if result_id == f"{file_path}_{idx}":
                            continue

                        # Calculate similarity (distance is actually similarity in cosine space)
                        similarity = 1 - distance if distance else 0

                        if similarity >= self.SIMILARITY_THRESHOLD:
                            # Extract metadata
                            metadata = (
                                results["metadatas"][0][result_idx]
                                if results["metadatas"] and result_idx < len(results["metadatas"][0])
                                else {}
                            )

                            finding = DuplicationFinding(
                                source_file=file_path,
                                target_file=metadata.get("file_path", "unknown"),
                                source_snippet=snippet[:200],
                                target_snippet=metadata.get(
                                    "snippet", "..."
                                )[:200],
                                similarity=similarity,
                                severity="MEDIUM"
                                if similarity < 0.95
                                else "HIGH",
                            )
                            findings.append(finding)
        except ImportError:
            # Graceful degradation if sentence-transformers not available
            pass
        except Exception:
            # Graceful degradation
            pass

        return findings

    def store_snapshot(self, snippets: list[tuple[str, str]], run_id: str) -> None:
        """Store a snapshot of analyzed snippets for future comparison.

        Args:
            snippets: List of (file_path, code_snippet) tuples.
            run_id: Identifier for this analysis run.
        """
        try:
            metadata_file = self.CHROMA_DB_PATH / f"snapshot_{run_id}.json"
            snapshot_data = [
                {"file_path": file_path, "snippet": snippet}
                for file_path, snippet in snippets
            ]
            with open(metadata_file, "w") as f:
                json.dump(snapshot_data, f)
        except Exception:
            # Graceful degradation
            pass

    def clear_database(self) -> None:
        """Clear the ChromaDB database."""
        try:
            if self.client and hasattr(self.client, "_system"):
                # Delete and recreate collection
                self.client.delete_collection(name="code_snippets")
                self.collection = self.client.get_or_create_collection(
                    name="code_snippets",
                    metadata={"hnsw:space": "cosine"},
                )
        except Exception:
            # Graceful degradation
            pass
