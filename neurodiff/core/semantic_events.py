"""Semantic event dataclasses for code analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union


class NeuroDiffError(Exception):
    """Base exception for NeuroDiff errors."""

    pass


@dataclass
class FunctionAdded:
    """Represents a function that was added."""

    name: str
    language: str
    line_start: int
    line_end: int
    cyclomatic_complexity: int


@dataclass
class FunctionModified:
    """Represents a function that was modified."""

    name: str
    language: str
    line_start_before: int
    line_end_before: int
    line_start_after: int
    line_end_after: int
    complexity_before: int
    complexity_after: int
    changes_summary: str


@dataclass
class FunctionRemoved:
    """Represents a function that was removed."""

    name: str
    language: str
    line_start: int
    line_end: int
    cyclomatic_complexity: int


@dataclass
class ClassAdded:
    """Represents a class that was added."""

    name: str
    language: str
    line_start: int
    line_end: int
    methods: list[str]


@dataclass
class ClassModified:
    """Represents a class that was modified."""

    name: str
    language: str
    line_start_before: int
    line_end_before: int
    line_start_after: int
    line_end_after: int
    methods_added: list[str]
    methods_removed: list[str]
    methods_modified: list[str]


@dataclass
class ImportAdded:
    """Represents an import that was added."""

    module: str
    language: str
    line: int
    full_statement: str


@dataclass
class ImportRemoved:
    """Represents an import that was removed."""

    module: str
    language: str
    line: int
    full_statement: str


# Union type for all semantic events
SemanticEvent = Union[
    FunctionAdded,
    FunctionModified,
    FunctionRemoved,
    ClassAdded,
    ClassModified,
    ImportAdded,
    ImportRemoved,
]
