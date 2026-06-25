"""Git parser for extracting diffs."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FileDiff:
    """Represents a file diff between two Git refs."""

    path: str
    language: str
    content_before: str
    content_after: str
    raw_diff: str


class GitParser:
    """Parser for extracting diffs from Git repositories."""

    def __init__(self, repo_path: Path) -> None:
        """Initialize the GitParser.

        Args:
            repo_path: Path to the Git repository.

        Raises:
            NeuroDiffError: If the path is not a valid Git repository.
        """
        from .semantic_events import NeuroDiffError

        self.repo_path = Path(repo_path)
        if not (self.repo_path / ".git").exists():
            raise NeuroDiffError(f"Not a Git repository: {repo_path}")

    def get_file_diffs(self, base_ref: str, head_ref: str) -> list[FileDiff]:
        """Get file diffs between two Git refs.

        Args:
            base_ref: The base Git reference (commit, branch, tag).
            head_ref: The head Git reference.

        Returns:
            List of FileDiff objects for all changed files.

        Raises:
            NeuroDiffError: If Git command fails.
        """
        from .semantic_events import NeuroDiffError

        try:
            # Get list of changed files
            result = subprocess.run(
                ["git", "diff", "--name-only", f"{base_ref}...{head_ref}"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            file_paths = result.stdout.strip().split("\n")
            file_paths = [fp for fp in file_paths if fp]  # Remove empty strings

            diffs = []
            for file_path in file_paths:
                diff = self._get_file_diff(file_path, base_ref, head_ref)
                if diff:
                    diffs.append(diff)

            return diffs
        except subprocess.CalledProcessError as e:
            raise NeuroDiffError(
                f"Failed to get diffs: {e.stderr or e.stdout}"
            ) from e

    def _get_file_diff(
        self, file_path: str, base_ref: str, head_ref: str
    ) -> FileDiff | None:
        """Get the diff for a single file.

        Args:
            file_path: The path to the file relative to the repo root.
            base_ref: The base Git reference.
            head_ref: The head Git reference.

        Returns:
            FileDiff object or None if file cannot be read.
        """
        from .semantic_events import NeuroDiffError

        try:
            # Get raw diff
            result = subprocess.run(
                ["git", "diff", f"{base_ref}...{head_ref}", file_path],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            raw_diff = result.stdout

            # Get content before
            try:
                result_before = subprocess.run(
                    ["git", "show", f"{base_ref}:{file_path}"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                content_before = result_before.stdout
            except subprocess.CalledProcessError:
                content_before = ""

            # Get content after
            try:
                result_after = subprocess.run(
                    ["git", "show", f"{head_ref}:{file_path}"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                content_after = result_after.stdout
            except subprocess.CalledProcessError:
                content_after = ""

            # Determine language from file extension
            language = self._detect_language(file_path)

            return FileDiff(
                path=file_path,
                language=language,
                content_before=content_before,
                content_after=content_after,
                raw_diff=raw_diff,
            )
        except subprocess.CalledProcessError as e:
            return None

    def _detect_language(self, file_path: str) -> str:
        """Detect the programming language from file extension.

        Args:
            file_path: The path to the file.

        Returns:
            The detected language or 'unknown'.
        """
        ext_to_lang = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".go": "go",
            ".rb": "ruby",
            ".php": "php",
        }
        suffix = Path(file_path).suffix.lower()
        return ext_to_lang.get(suffix, "unknown")
