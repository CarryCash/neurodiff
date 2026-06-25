"""NeuroDiff - Semantic git diff analysis CLI tool.

A complete open-source tool for analyzing code changes at the semantic level,
detecting functional changes, security vulnerabilities, and code duplication patterns.
"""
from __future__ import annotations

__version__ = "0.1.0"
__author__ = "NeuroDiff Contributors"

from neurodiff.core import (
    ClassAdded,
    ClassModified,
    FileDiff,
    FunctionAdded,
    FunctionModified,
    FunctionRemoved,
    GitParser,
    ImportAdded,
    ImportRemoved,
    SemanticEvent,
)
from neurodiff.core.semantic_events import NeuroDiffError
from neurodiff.engines import (
    DuplicationEngine,
    DuplicationFinding,
    SecurityEngine,
    SecurityFinding,
)
from neurodiff.output import Reporter

__all__ = [
    "ClassAdded",
    "ClassModified",
    "DuplicationEngine",
    "DuplicationFinding",
    "FileDiff",
    "FunctionAdded",
    "FunctionModified",
    "FunctionRemoved",
    "GitParser",
    "ImportAdded",
    "ImportRemoved",
    "NeuroDiffError",
    "Reporter",
    "SecurityEngine",
    "SecurityFinding",
    "SemanticEvent",
]
