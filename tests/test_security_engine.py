"""Tests for security analysis engine."""
from __future__ import annotations

import re

# pyrefly: ignore [missing-import]
import pytest

from neurodiff.engines.security_engine import SecurityEngine, SecurityFinding
from neurodiff.core.semantic_events import FunctionAdded


FILE = "test.py"


def _make_func_added(name: str = "func", start_line: int = 1, body_lines: int = 100) -> FunctionAdded:
    """Helper: create a FunctionAdded event that covers most of the file."""
    return FunctionAdded(
        name=name,
        file=FILE,
        start_line=start_line,
        body_lines=body_lines,
        calls=[],
        cyclomatic_complexity=1,
    )


@pytest.fixture
def engine() -> SecurityEngine:
    return SecurityEngine()


# ---------------------------------------------------------------------------
# Regex / Pattern validation
# ---------------------------------------------------------------------------

def test_fallback_patterns_are_valid_regex(engine: SecurityEngine) -> None:
    for rule_name, rule_config in engine.FALLBACK_PATTERNS.items():
        pattern = rule_config["pattern"]
        try:
            re.compile(pattern)
        except re.error as exc:
            pytest.fail(f"Invalid regex pattern in '{rule_name}': {exc}")


# ---------------------------------------------------------------------------
# Hardcoded secret detection (regex fallback)
# ---------------------------------------------------------------------------

def test_hardcoded_api_key_detected(engine: SecurityEngine) -> None:
    code = 'api_key = "sk_test_1234567890"\n'
    events = [_make_func_added()]
    findings = engine.analyze(code, FILE, events)
    assert any(f.rule_id == "hardcoded_secret" for f in findings)


def test_hardcoded_password_detected(engine: SecurityEngine) -> None:
    code = 'password = "super_secret_password"\n'
    events = [_make_func_added()]
    findings = engine.analyze(code, FILE, events)
    assert any(f.rule_id == "hardcoded_secret" for f in findings)


def test_short_secret_not_flagged(engine: SecurityEngine) -> None:
    """Secrets shorter than 8 chars after the = should NOT be flagged."""
    code = 'token = "abc"\n'
    events = [_make_func_added()]
    findings = engine.analyze(code, FILE, events)
    secret_findings = [f for f in findings if f.rule_id == "hardcoded_secret"]
    assert len(secret_findings) == 0


# ---------------------------------------------------------------------------
# Clean code produces no findings
# ---------------------------------------------------------------------------

def test_clean_code_no_findings(engine: SecurityEngine) -> None:
    code = '''
def greet(name):
    """Greet a person."""
    return f"Hello, {name}!"
'''
    events = [_make_func_added()]
    findings = engine.analyze(code, FILE, events)
    assert isinstance(findings, list)
    # No false positives on clean code
    secret_findings = [f for f in findings if f.rule_id == "hardcoded_secret"]
    assert len(secret_findings) == 0


def test_empty_code_no_findings(engine: SecurityEngine) -> None:
    findings = engine.analyze("", FILE, [])
    assert findings == []


# ---------------------------------------------------------------------------
# Intersection filter: findings outside diff are excluded
# ---------------------------------------------------------------------------

def test_finding_outside_function_excluded(engine: SecurityEngine) -> None:
    """A secret on line 15 should not appear if the diff only covers lines 1-5."""
    code = "\n" * 14 + 'api_key = "sk_test_1234567890"\n'
    # Event only covers lines 1-5
    event = FunctionAdded(name="func", file=FILE, start_line=1, body_lines=5, calls=[], cyclomatic_complexity=1)
    findings = engine.analyze(code, FILE, [event])
    assert findings == []


def test_finding_inside_function_included(engine: SecurityEngine) -> None:
    """A secret on line 2 should appear if the diff covers lines 1-10."""
    code = "\napi_key = \"sk_test_1234567890\"\n"
    event = FunctionAdded(name="func", file=FILE, start_line=1, body_lines=10, calls=[], cyclomatic_complexity=1)
    findings = engine.analyze(code, FILE, [event])
    assert any(f.rule_id == "hardcoded_secret" for f in findings)


# ---------------------------------------------------------------------------
# SecurityFinding structure
# ---------------------------------------------------------------------------

def test_finding_has_required_fields(engine: SecurityEngine) -> None:
    code = 'secret = "my_super_secret_key"\n'
    events = [_make_func_added()]
    findings = engine.analyze(code, FILE, events)
    for f in findings:
        assert isinstance(f, SecurityFinding)
        assert hasattr(f, "severity")
        assert hasattr(f, "category")
        assert hasattr(f, "file")
        assert hasattr(f, "line")
        assert hasattr(f, "function_name")
        assert hasattr(f, "description")
        assert hasattr(f, "rule_id")
        assert f.file == FILE
        assert f.severity in ("critical", "high", "medium", "low")


# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------

def test_severity_mapping(engine: SecurityEngine) -> None:
    assert engine._map_severity("error") == "critical"
    assert engine._map_severity("warning") == "high"
    assert engine._map_severity("info") == "low"
    assert engine._map_severity("unknown_level") == "medium"
