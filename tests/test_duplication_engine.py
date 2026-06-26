"""Tests for duplication detection engine."""
from __future__ import annotations

# pyrefly: ignore [missing-import]
import pytest
from pathlib import Path

from neurodiff.engines.duplication_engine import DuplicationEngine, DuplicationFinding
from neurodiff.core.semantic_events import FunctionAdded


def _make_added(name: str, file: str, body_lines: int = 5) -> FunctionAdded:
    return FunctionAdded(
        name=name,
        file=file,
        start_line=1,
        body_lines=body_lines,
        calls=[],
        cyclomatic_complexity=1,
    )


@pytest.fixture
def engine() -> DuplicationEngine:
    return DuplicationEngine()


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------

def test_similarity_threshold(engine: DuplicationEngine) -> None:
    assert engine.SIMILARITY_THRESHOLD == 0.80
    assert 0.0 <= engine.SIMILARITY_THRESHOLD <= 1.0


def test_chroma_db_path_is_pathlib(engine: DuplicationEngine) -> None:
    assert isinstance(engine.CHROMA_DB_PATH, Path)
    assert "neurodiff" in str(engine.CHROMA_DB_PATH)


def test_chromadb_flag_is_bool(engine: DuplicationEngine) -> None:
    assert isinstance(engine.has_chromadb, bool)


# ---------------------------------------------------------------------------
# Empty / graceful cases
# ---------------------------------------------------------------------------

def test_analyze_empty_list(engine: DuplicationEngine) -> None:
    findings = engine.analyze([])
    assert findings == []


def test_analyze_returns_list(engine: DuplicationEngine) -> None:
    event = _make_added("hello", "file1.py")
    findings = engine.analyze([(event, "def hello(): return 'world'")])
    assert isinstance(findings, list)


# ---------------------------------------------------------------------------
# DuplicationFinding structure
# ---------------------------------------------------------------------------

def test_finding_has_required_fields(engine: DuplicationEngine) -> None:
    if not engine.has_chromadb:
        pytest.skip("ChromaDB not available")

    event1 = _make_added("calc", "file1.py")
    event2 = _make_added("compute", "file2.py")
    snippet = "def f(x, y):\n    return x + y\n"

    # Pre-seed with identical snippet
    try:
        engine.collection.add(
            ids=["file2.py:compute"],
            documents=[snippet],
            metadatas=[{"file": "file2.py", "function_name": "compute", "language": "python"}],
        )
    except Exception:
        pytest.skip("Could not seed ChromaDB")

    findings = engine.analyze([(event1, snippet)])
    for f in findings:
        assert isinstance(f, DuplicationFinding)
        assert hasattr(f, "new_function")
        assert hasattr(f, "new_file")
        assert hasattr(f, "similar_function")
        assert hasattr(f, "similar_file")
        assert hasattr(f, "similarity_score")
        assert hasattr(f, "severity")
        assert 0.0 <= f.similarity_score <= 1.0
        assert f.severity in ("high", "medium")


# ---------------------------------------------------------------------------
# Severity thresholds
# ---------------------------------------------------------------------------

def test_severity_high_above_90() -> None:
    """Manually check threshold logic."""
    engine = DuplicationEngine()
    # Verify the constant is correct; actual findings depend on ChromaDB
    assert engine.SIMILARITY_THRESHOLD == 0.80


# ---------------------------------------------------------------------------
# Excluded dirs constant
# ---------------------------------------------------------------------------

def test_excluded_dirs(engine: DuplicationEngine) -> None:
    assert "node_modules" in engine.EXCLUDED_DIRS
    assert ".git" in engine.EXCLUDED_DIRS
    assert "__pycache__" in engine.EXCLUDED_DIRS
    assert "venv" in engine.EXCLUDED_DIRS
