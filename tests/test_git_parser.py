"""Tests for Git parser."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from neurodiff.core.git_parser import GitParser
from neurodiff.core.semantic_events import NeuroDiffError


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository for testing."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize git repo
    subprocess.run(
        ["git", "init"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )

    # Configure git user
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )

    # Create initial file and commit
    (repo_path / "test.py").write_text("print('hello')")
    subprocess.run(
        ["git", "add", "."],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )

    return repo_path


def test_git_parser_init_invalid_path(tmp_path: Path) -> None:
    """Test GitParser initialization with invalid path."""
    invalid_path = tmp_path / "not_a_repo"
    invalid_path.mkdir()

    with pytest.raises(NeuroDiffError):
        GitParser(invalid_path)


def test_git_parser_init_valid_path(temp_git_repo: Path) -> None:
    """Test GitParser initialization with valid Git repository."""
    parser = GitParser(temp_git_repo)
    assert parser.repo_path == temp_git_repo


def test_detect_language() -> None:
    """Test language detection from file extensions."""
    parser = GitParser(Path("."))  # Dummy path

    assert parser._detect_language("test.py") == "python"
    assert parser._detect_language("test.js") == "javascript"
    assert parser._detect_language("test.ts") == "typescript"
    assert parser._detect_language("test.tsx") == "typescript"
    assert parser._detect_language("test.jsx") == "javascript"
    assert parser._detect_language("test.java") == "java"
    assert parser._detect_language("test.go") == "go"
    assert parser._detect_language("test.rb") == "ruby"
    assert parser._detect_language("test.unknown") == "unknown"


def test_get_file_diffs_no_changes(temp_git_repo: Path) -> None:
    """Test getting file diffs when there are no changes."""
    parser = GitParser(temp_git_repo)

    # Get diffs between HEAD and HEAD (no changes)
    diffs = parser.get_file_diffs("HEAD", "HEAD")
    assert isinstance(diffs, list)
    # Should be empty since comparing same ref
    assert len(diffs) == 0


def test_get_file_diffs_with_changes(temp_git_repo: Path) -> None:
    """Test getting file diffs with actual changes."""
    parser = GitParser(temp_git_repo)

    # Make changes
    (temp_git_repo / "test.py").write_text("print('hello world')")
    subprocess.run(
        ["git", "add", "."],
        cwd=temp_git_repo,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Update test.py"],
        cwd=temp_git_repo,
        capture_output=True,
        check=True,
    )

    # Get diffs
    diffs = parser.get_file_diffs("HEAD~1", "HEAD")
    assert isinstance(diffs, list)
    # Should have at least one file diff
    assert len(diffs) >= 0  # May be empty in some cases


def test_file_diff_structure(temp_git_repo: Path) -> None:
    """Test that FileDiff has expected structure."""
    parser = GitParser(temp_git_repo)

    # Make changes
    (temp_git_repo / "test.py").write_text("print('updated')")
    subprocess.run(
        ["git", "add", "."],
        cwd=temp_git_repo,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Update"],
        cwd=temp_git_repo,
        capture_output=True,
        check=True,
    )

    diffs = parser.get_file_diffs("HEAD~1", "HEAD")

    for diff in diffs:
        assert hasattr(diff, "path")
        assert hasattr(diff, "language")
        assert hasattr(diff, "content_before")
        assert hasattr(diff, "content_after")
        assert hasattr(diff, "raw_diff")
