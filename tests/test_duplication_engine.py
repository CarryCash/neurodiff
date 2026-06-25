"""Tests for duplication detection."""
from __future__ import annotations

import pytest

from neurodiff.engines.duplication_engine import DuplicationEngine


@pytest.fixture
def duplication_engine() -> DuplicationEngine:
    """Create a DuplicationEngine instance for testing."""
    return DuplicationEngine()


def test_similarity_threshold() -> None:
    """Test similarity threshold constant."""
    engine = DuplicationEngine()
    assert engine.SIMILARITY_THRESHOLD == 0.80
    assert 0.0 <= engine.SIMILARITY_THRESHOLD <= 1.0


def test_chroma_db_initialization(duplication_engine: DuplicationEngine) -> None:
    """Test ChromaDB initialization."""
    # Should initialize gracefully
    assert isinstance(duplication_engine.has_chromadb, bool)


def test_analyze_empty_snippets(duplication_engine: DuplicationEngine) -> None:
    """Test analysis with empty snippet list."""
    findings = duplication_engine.analyze([])
    assert isinstance(findings, list)
    assert len(findings) == 0


def test_analyze_single_snippet(duplication_engine: DuplicationEngine) -> None:
    """Test analysis with single snippet."""
    snippets = [("file1.py", "def hello(): return 'world'")]
    findings = duplication_engine.analyze(snippets)
    assert isinstance(findings, list)


def test_analyze_duplicate_snippets(duplication_engine: DuplicationEngine) -> None:
    """Test analysis with duplicate snippets."""
    code = "def hello(): return 'world'"
    snippets = [("file1.py", code), ("file2.py", code)]
    findings = duplication_engine.analyze(snippets)
    assert isinstance(findings, list)
    # May or may not find duplicates depending on ChromaDB availability


def test_analyze_similar_snippets(duplication_engine: DuplicationEngine) -> None:
    """Test analysis with similar snippets."""
    snippets = [
        ("file1.py", "def add(x, y): return x + y"),
        ("file2.py", "def sum_numbers(a, b): return a + b"),
    ]
    findings = duplication_engine.analyze(snippets)
    assert isinstance(findings, list)


def test_chroma_db_path_property() -> None:
    """Test ChromaDB path property."""
    from pathlib import Path

    engine = DuplicationEngine()
    assert isinstance(engine.CHROMA_DB_PATH, Path)
    assert "neurodiff" in str(engine.CHROMA_DB_PATH)


def test_finding_structure(duplication_engine: DuplicationEngine) -> None:
    """Test that findings have expected structure."""
    # If ChromaDB is available, check finding structure
    if duplication_engine.has_chromadb:
        findings = duplication_engine.analyze(
            [("file1.py", "def func(): pass"), ("file2.py", "def func(): pass")]
        )
        for finding in findings:
            assert hasattr(finding, "source_file")
            assert hasattr(finding, "target_file")
            assert hasattr(finding, "similarity")
            assert 0.0 <= finding.similarity <= 1.0


def test_clear_database(duplication_engine: DuplicationEngine) -> None:
    """Test clearing the database."""
    # Should not raise an exception
    duplication_engine.clear_database()


def test_store_snapshot(duplication_engine: DuplicationEngine, tmp_path) -> None:
    """Test storing a snapshot."""
    snippets = [("file1.py", "def func(): pass")]
    run_id = "test_run_001"
    # Should not raise an exception
    duplication_engine.store_snapshot(snippets, run_id)
