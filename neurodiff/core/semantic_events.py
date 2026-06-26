"""Semantic event dataclasses for code analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union


class NeuroDiffError(Exception):
    """Base exception for NeuroDiff errors."""
    pass


@dataclass
class FunctionAdded:
    name: str
    file: str
    start_line: int
    body_lines: int
    calls: list[str]
    cyclomatic_complexity: int


@dataclass
class FunctionModified:
    name: str
    file: str
    start_line: int
    lines_before: int
    lines_after: int
    signature_changed: bool
    calls_added: list[str]
    calls_removed: list[str]
    complexity_before: int
    complexity_after: int


@dataclass
class FunctionRemoved:
    name: str
    file: str


@dataclass
class ClassAdded:
    name: str
    file: str
    methods: list[str]
    inherits_from: list[str]


@dataclass
class ClassModified:
    name: str
    file: str
    methods_added: list[str]
    methods_removed: list[str]


@dataclass
class ImportAdded:
    module: str
    file: str
    symbols: list[str]


@dataclass
class ImportRemoved:
    module: str
    file: str


SemanticEvent = Union[
    FunctionAdded,
    FunctionModified,
    FunctionRemoved,
    ClassAdded,
    ClassModified,
    ImportAdded,
    ImportRemoved,
]
