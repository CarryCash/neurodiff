"""Tests for Shadow Developer Engine."""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neurodiff.engines.shadow_engine import (
    DocInventory,
    ShadowEngine,
    ShadowReport,
    _extract_signature,
    _should_skip,
)
from neurodiff.core.semantic_events import FunctionModified, FunctionAdded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_py_file(tmp_path: Path, name: str, content: str) -> Path:
    """Write a .py file into tmp_path and return its path."""
    f = tmp_path / name
    f.write_text(textwrap.dedent(content), encoding="utf-8")
    return f


def _fake_provider() -> MagicMock:
    """Return a mock LLM provider whose complete() is an AsyncMock."""
    p = MagicMock()
    p.name = "test"
    p.complete = AsyncMock(return_value='{"docstring": "A test docstring."}')
    return p


# ---------------------------------------------------------------------------
# _should_skip
# ---------------------------------------------------------------------------

class TestShouldSkip:
    def test_skips_venv(self):
        assert _should_skip(Path("project/venv/lib/foo.py"))

    def test_skips_git(self):
        assert _should_skip(Path(".git/objects/foo"))

    def test_allows_normal_path(self):
        assert not _should_skip(Path("neurodiff/engines/shadow_engine.py"))


# ---------------------------------------------------------------------------
# _extract_signature
# ---------------------------------------------------------------------------

class TestExtractSignature:
    def test_simple_function(self):
        src = "def foo(x, y): pass"
        tree = ast.parse(src)
        node = tree.body[0]
        # pyrefly: ignore [bad-argument-type]
        sig = _extract_signature(node)
        assert sig == "def foo(x, y)"

    def test_async_function(self):
        src = "async def bar(a: int, b: str) -> bool: pass"
        tree = ast.parse(src)
        node = tree.body[0]
        # pyrefly: ignore [bad-argument-type]
        sig = _extract_signature(node)
        assert "async def bar" in sig
        assert "-> bool" in sig

    def test_kwargs(self):
        src = "def baz(*args, **kwargs): pass"
        tree = ast.parse(src)
        node = tree.body[0]
        # pyrefly: ignore [bad-argument-type]
        sig = _extract_signature(node)
        assert "*args" in sig
        assert "**kwargs" in sig


# ---------------------------------------------------------------------------
# DocInventory / scan_repo
# ---------------------------------------------------------------------------

class TestScanRepo:
    def test_generate_mode_no_readme(self, tmp_path: Path):
        """Repo without README → mode = 'generate'."""
        _make_py_file(tmp_path, "app.py", """\
            def hello():
                print("hi")

            def world():
                print("world")
        """)
        engine = ShadowEngine(provider=None)
        inv = engine.scan_repo(tmp_path)
        assert not inv.has_readme
        assert inv.mode == "generate"
        assert inv.functions_without_docstrings >= 2

    def test_update_mode_with_readme_and_docs(self, tmp_path: Path):
        """Repo with README + >40% docstrings → mode = 'update'."""
        (tmp_path / "README.md").write_text("# Hello", encoding="utf-8")
        _make_py_file(tmp_path, "app.py", '''\
            def documented():
                """This is documented."""
                pass

            def also_documented():
                """Also has a docstring."""
                pass

            def undocumented():
                pass
        ''')
        engine = ShadowEngine(provider=None)
        inv = engine.scan_repo(tmp_path)
        assert inv.has_readme
        # 2 documented, 1 undocumented → 33% undocumented → mode="update"
        assert inv.mode == "update"

    def test_generate_mode_mostly_undocumented(self, tmp_path: Path):
        """Repo with README but >60% undocumented → mode = 'generate'."""
        (tmp_path / "README.md").write_text("# Hello", encoding="utf-8")
        _make_py_file(tmp_path, "app.py", """\
            def a(): pass
            def b(): pass
            def c(): pass
            def d():
                \"\"\"Documented.\"\"\"
                pass
        """)
        engine = ShadowEngine(provider=None)
        inv = engine.scan_repo(tmp_path)
        assert inv.has_readme
        # 3 undocumented out of 4 = 75% → generate
        assert inv.mode == "generate"

    def test_skips_private_functions(self, tmp_path: Path):
        """Private functions (single underscore) should not be listed as undocumented."""
        (tmp_path / "README.md").write_text("# Hello", encoding="utf-8")
        _make_py_file(tmp_path, "app.py", """\
            def public():
                \"\"\"Public.\"\"\"
                pass

            def _private():
                pass
        """)
        engine = ShadowEngine(provider=None)
        inv = engine.scan_repo(tmp_path)
        # _private should be skipped → 1 documented, 0 undocumented
        assert inv.functions_without_docstrings == 0

    def test_counts_dunder_init(self, tmp_path: Path):
        """__init__ should be counted."""
        (tmp_path / "README.md").write_text("# Hello", encoding="utf-8")
        _make_py_file(tmp_path, "app.py", """\
            class Foo:
                def __init__(self):
                    pass
        """)
        engine = ShadowEngine(provider=None)
        inv = engine.scan_repo(tmp_path)
        assert inv.functions_without_docstrings >= 1
        names = [f["name"] for f in inv.undocumented_functions]
        assert "__init__" in names


# ---------------------------------------------------------------------------
# Docstring Insertion Logic
# ---------------------------------------------------------------------------

class TestDocstringInsertion:
    def test_insert_single_docstring(self, tmp_path: Path):
        """Test that a docstring is inserted at the correct position."""
        src = textwrap.dedent("""\
            def greet(name):
                print(f"Hello {name}")
        """)
        py_file = tmp_path / "greet.py"
        py_file.write_text(src, encoding="utf-8")

        func_info = {
            "name": "greet",
            "file": "greet.py",
            "abs_file": str(py_file),
            "signature": "def greet(name)",
            "lineno": 1,
            "body_lineno": 2,
            "calls": ["print"],
            "body_summary": '    print(f"Hello {name}")',
            "complexity": 1,
        }

        ShadowEngine._insert_single_docstring(
            py_file, func_info, "Greet someone by name.\n\nArgs:\n    name: The person's name."
        )

        result = py_file.read_text(encoding="utf-8")
        assert '"""Greet someone by name.' in result
        # Backup should exist
        assert (tmp_path / "greet.py.bak").exists()


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

class TestBackup:
    def test_backup_created(self, tmp_path: Path):
        """Backup should be created before modification."""
        f = tmp_path / "README.md"
        f.write_text("# Original", encoding="utf-8")
        ShadowEngine._backup(f)
        bak = tmp_path / "README.md.bak"
        assert bak.exists()
        assert bak.read_text(encoding="utf-8") == "# Original"

    def test_backup_not_overwritten(self, tmp_path: Path):
        """Second call to _backup should not overwrite existing .bak."""
        f = tmp_path / "README.md"
        f.write_text("# Original", encoding="utf-8")
        ShadowEngine._backup(f)

        f.write_text("# Modified", encoding="utf-8")
        ShadowEngine._backup(f)

        bak = tmp_path / "README.md.bak"
        assert bak.read_text(encoding="utf-8") == "# Original"


# ---------------------------------------------------------------------------
# Dry Run
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_write_file_dry_run(self, tmp_path: Path):
        """Dry run should not create the file."""
        target = tmp_path / "README.md"
        report = ShadowReport(mode="generate")
        ShadowEngine._write_file(target, "# Hello", dry_run=True, report=report, generated=True)
        assert not target.exists()
        assert len(report.files_generated) == 1

    def test_write_file_real(self, tmp_path: Path):
        """Real write should create the file."""
        target = tmp_path / "docs" / "api.md"
        report = ShadowReport(mode="generate")
        ShadowEngine._write_file(target, "# API", dry_run=False, report=report, generated=True)
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "# API"


# ---------------------------------------------------------------------------
# Change Detector (Mode 1)
# ---------------------------------------------------------------------------

class TestChangeDetector:
    def test_detects_signature_change(self):
        events = [
            FunctionModified(
                name="foo", file="app.py", start_line=10,
                lines_before=5, lines_after=8,
                signature_changed=True,
                calls_added=[], calls_removed=[],
                complexity_before=2, complexity_after=3,
            )
        ]
        # pyrefly: ignore [bad-argument-type]
        entities = ShadowEngine._detect_changed_entities(events)
        assert len(entities) == 1
        assert entities[0]["name"] == "foo"
        assert entities[0]["signature_changed"] is True

    def test_detects_new_function(self):
        events = [
            FunctionAdded(
                name="new_func", file="app.py",
                start_line=1, body_lines=5,
                calls=[], cyclomatic_complexity=1,
            )
        ]
        # pyrefly: ignore [bad-argument-type]
        entities = ShadowEngine._detect_changed_entities(events)
        assert len(entities) == 1
        assert entities[0]["type"] == "FunctionAdded"

    def test_ignores_non_signature_modification(self):
        events = [
            FunctionModified(
                name="bar", file="app.py", start_line=10,
                lines_before=5, lines_after=8,
                signature_changed=False,
                calls_added=[], calls_removed=[],
                complexity_before=2, complexity_after=3,
            )
        ]
        # pyrefly: ignore [bad-argument-type]
        entities = ShadowEngine._detect_changed_entities(events)
        assert len(entities) == 0


# ---------------------------------------------------------------------------
# Coverage Calculation
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_full_coverage(self):
        inv = DocInventory(
            has_readme=True, has_architecture_md=True,
            markdown_files=[], functions_with_docstrings=10,
            functions_without_docstrings=0, undocumented_functions=[],
            mode="update",
        )
        assert ShadowEngine._calc_coverage(inv) == 100.0

    def test_zero_coverage(self):
        inv = DocInventory(
            has_readme=False, has_architecture_md=False,
            markdown_files=[], functions_with_docstrings=0,
            functions_without_docstrings=5, undocumented_functions=[],
            mode="generate",
        )
        assert ShadowEngine._calc_coverage(inv) == 0.0

    def test_empty_project(self):
        inv = DocInventory(
            has_readme=True, has_architecture_md=True,
            markdown_files=[], functions_with_docstrings=0,
            functions_without_docstrings=0, undocumented_functions=[],
            mode="update",
        )
        assert ShadowEngine._calc_coverage(inv) == 100.0
