"""Code duplication detection engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from pathlib import Path

from neurodiff.core.semantic_events import FunctionAdded


@dataclass
class DuplicationFinding:
    """Represents a code duplication finding."""
    new_function: str
    new_file: str
    similar_function: str
    similar_file: str
    similarity_score: float
    severity: Literal["high", "medium"]


class DuplicationEngine:
    """Engine for detecting code duplication using ChromaDB and embeddings."""

    SIMILARITY_THRESHOLD = 0.80
    CHROMA_DB_PATH = Path.home() / ".neurodiff" / "chroma_db"
    EXCLUDED_DIRS = {".git", "node_modules", "__pycache__", "venv", "env", ".venv"}

    def __init__(self) -> None:
        """Initialize the DuplicationEngine."""
        self.has_chromadb = self._check_chromadb()
        self.client = None
        self.collection = None

        if self.has_chromadb:
            self._initialize_chromadb()

    def _check_chromadb(self) -> bool:
        """Check if ChromaDB is available."""
        try:
            import chromadb
            return True
        except ImportError:
            return False

    def _initialize_chromadb(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            self.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.CHROMA_DB_PATH))
            self.collection = self.client.get_or_create_collection(
                name="code_snippets",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            self.has_chromadb = False

    def _get_language_from_ext(self, file_path: Path) -> str | None:
        ext = file_path.suffix.lower()
        if ext == ".py":
            return "python"
        elif ext == ".js":
            return "javascript"
        elif ext == ".ts":
            return "typescript"
        return None

    def index_repo(self, repo_path: Path, ast_engine) -> int:
        """Index all functions in the repository."""
        if not self.has_chromadb or not self.collection:
            raise Exception("ChromaDB is not available. Please install it first.")

        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError as e:
            raise Exception("sentence-transformers not installed.") from e

        count = 0
        repo_path = repo_path.resolve()

        for file_path in repo_path.rglob("*"):
            if file_path.is_dir():
                continue
            
            # Check exclusions
            if any(part in self.EXCLUDED_DIRS for part in file_path.parts):
                continue

            lang = self._get_language_from_ext(file_path)
            if not lang:
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
                tree = ast_engine._parse_code(content, lang)
                functions = ast_engine._extract_functions(tree, lang)
                
                for func_name, info in functions.items():
                    func_body = ast_engine._extract_function_body(
                        content, info["start_line"], info["end_line"]
                    )
                    
                    if not func_body.strip():
                        continue

                    embedding = model.encode(func_body).tolist()
                    rel_path = str(file_path.relative_to(repo_path))
                    
                    doc_id = f"{rel_path}:{func_name}"
                    
                    self.collection.add(
                        ids=[doc_id],
                        embeddings=[embedding],
                        documents=[func_body],
                        metadatas=[
                            {
                                "file": rel_path,
                                "function_name": func_name,
                                "language": lang,
                            }
                        ],
                    )
                    count += 1
            except Exception:
                # Ignore unreadable files or parsing errors
                pass

        return count

    def analyze(
        self, added_functions: list[tuple[FunctionAdded, str]]
    ) -> list[DuplicationFinding]:
        """Analyze newly added functions for duplication.
        
        Args:
            added_functions: List of (FunctionAdded event, function_body code) tuples.
        """
        findings: list[DuplicationFinding] = []

        if not self.has_chromadb or not self.collection:
            return findings

        try:
            findings.extend(self._find_duplicates_with_chromadb(added_functions))
        except Exception:
            pass

        return findings

    def _find_duplicates_with_chromadb(
        self, added_functions: list[tuple[FunctionAdded, str]]
    ) -> list[DuplicationFinding]:
        """Find duplicate code snippets using ChromaDB."""
        findings: list[DuplicationFinding] = []

        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")

            for event, snippet in added_functions:
                if not snippet.strip():
                    continue

                embedding = model.encode(snippet).tolist()

                # pyrefly: ignore [missing-attribute]
                results = self.collection.query(
                    query_embeddings=[embedding],
                    n_results=3,
                )

                if results and results["ids"] and len(results["ids"]) > 0:
                    for result_idx, (result_id, distance) in enumerate(
                        zip(results["ids"][0], results["distances"][0])
                    ):
                        metadata = (
                            results["metadatas"][0][result_idx]
                            if results["metadatas"] and result_idx < len(results["metadatas"][0])
                            else {}
                        )

                        target_file = metadata.get("file", "unknown")
                        target_function = metadata.get("function_name", "unknown")

                        # Skip the exact same function in the same file
                        if target_file == event.file and target_function == event.name:
                            continue

                        similarity = min(1.0, max(0.0, 1.0 - distance)) if distance is not None else 0.0

                        if similarity >= self.SIMILARITY_THRESHOLD:
                            finding = DuplicationFinding(
                                new_function=event.name,
                                new_file=event.file,
                                similar_function=target_function,
                                similar_file=target_file,
                                similarity_score=similarity,
                                severity="high" if similarity > 0.90 else "medium",
                            )
                            findings.append(finding)
        except ImportError:
            pass
        except Exception:
            pass

        return findings
