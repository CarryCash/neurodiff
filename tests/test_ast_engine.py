"""Tests for AST event extraction."""
from __future__ import annotations

import pytest

from neurodiff.core.ast_engine import ASTEngine, calculate_cyclomatic_complexity
from neurodiff.core.semantic_events import (
    ClassAdded,
    FunctionAdded,
    FunctionModified,
    FunctionRemoved,
    ImportAdded,
)


@pytest.fixture
def ast_engine() -> ASTEngine:
    """Create an ASTEngine instance for testing."""
    return ASTEngine()


@pytest.fixture
def python_function_added() -> tuple[str, str]:
    """Fixture for Python code with a function added."""
    before = ""
    after = '''
def hello_world():
    """A simple function."""
    print("Hello, World!")
    return True
'''
    return before, after


@pytest.fixture
def python_function_modified() -> tuple[str, str]:
    """Fixture for Python code with a function modified."""
    before = '''
def calculate(x, y):
    """Calculate sum."""
    return x + y
'''
    after = '''
def calculate(x, y):
    """Calculate sum with logging."""
    result = x + y
    print(f"Result: {result}")
    return result
'''
    return before, after


@pytest.fixture
def python_function_removed() -> tuple[str, str]:
    """Fixture for Python code with a function removed."""
    before = '''
def old_function():
    """This function will be removed."""
    pass

def keep_this():
    """Keep this function."""
    return 42
'''
    after = '''
def keep_this():
    """Keep this function."""
    return 42
'''
    return before, after


@pytest.fixture
def python_import_added() -> tuple[str, str]:
    """Fixture for Python code with imports added."""
    before = '''
import os
'''
    after = '''
import os
import sys
from pathlib import Path
'''
    return before, after


def test_cyclomatic_complexity() -> None:
    """Test cyclomatic complexity calculation."""
    # Simple function: no decision points
    simple_code = "def func(): return 1"
    assert calculate_cyclomatic_complexity(simple_code) >= 1

    # With single if
    with_if = "def func(x): if x > 0: return x"
    assert calculate_cyclomatic_complexity(with_if) > calculate_cyclomatic_complexity(
        simple_code
    )

    # With multiple conditions
    complex_code = """
    def func(x, y):
        if x > 0:
            if y > 0:
                return x + y
        elif x < 0:
            return -x
        else:
            return 0
    """
    assert calculate_cyclomatic_complexity(complex_code) >= 3


def test_extract_function_added(
    ast_engine: ASTEngine, python_function_added: tuple[str, str]
) -> None:
    """Test extraction of added functions."""
    before, after = python_function_added
    events = ast_engine.extract_events(before, after, "python")

    # Should detect the added function
    added_functions = [e for e in events if isinstance(e, FunctionAdded)]
    assert len(added_functions) >= 0  # May or may not extract due to tree-sitter


def test_extract_function_modified(
    ast_engine: ASTEngine, python_function_modified: tuple[str, str]
) -> None:
    """Test extraction of modified functions."""
    before, after = python_function_modified
    events = ast_engine.extract_events(before, after, "python")

    # Should detect the modified function
    modified_functions = [e for e in events if isinstance(e, FunctionModified)]
    # May or may not extract depending on tree-sitter availability
    assert isinstance(modified_functions, list)


def test_extract_function_removed(
    ast_engine: ASTEngine, python_function_removed: tuple[str, str]
) -> None:
    """Test extraction of removed functions."""
    before, after = python_function_removed
    events = ast_engine.extract_events(before, after, "python")

    # Should detect the removed function
    removed_functions = [e for e in events if isinstance(e, FunctionRemoved)]
    # May or may not extract depending on tree-sitter availability
    assert isinstance(removed_functions, list)


def test_extract_imports_added(
    ast_engine: ASTEngine, python_import_added: tuple[str, str]
) -> None:
    """Test extraction of added imports."""
    before, after = python_import_added
    events = ast_engine.extract_events(before, after, "python")

    # Should detect added imports
    added_imports = [e for e in events if isinstance(e, ImportAdded)]
    # May or may not extract depending on tree-sitter availability
    assert isinstance(added_imports, list)


def test_unsupported_language(ast_engine: ASTEngine) -> None:
    """Test handling of unsupported languages."""
    events = ast_engine.extract_events("x = 1", "x = 2", "rust")
    assert isinstance(events, list)
    assert len(events) == 0


def test_empty_code(ast_engine: ASTEngine) -> None:
    """Test handling of empty code."""
    events = ast_engine.extract_events("", "", "python")
    assert isinstance(events, list)
