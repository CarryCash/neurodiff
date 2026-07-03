"""Tests for Algorithm Performance Reviewer (Feature 4)."""
from __future__ import annotations

import textwrap
from pathlib import Path

from neurodiff.engines.perf_engine import (
    analyze_function,
    BenchmarkGenerator,
    AlgorithmRewrite,
    AlgorithmFinding,
)

# ---------------------------------------------------------------------------
# Pattern Detection Tests (Layer 1)
# ---------------------------------------------------------------------------

def test_nested_loop_detected():
    source = textwrap.dedent("""
        def find_duplicates(data):
            for i in range(len(data)):
                for j in range(i + 1, len(data)):
                    if data[i] == data[j]:
                        print("Dup")
    """)
    findings = analyze_function(source, "find_duplicates", "app.py")
    assert len(findings) == 1
    assert findings[0].pattern_type == "NESTED_LOOP"
    assert findings[0].complexity_before == "O(n²)"
    assert findings[0].severity == "critical"
    assert findings[0].nesting_depth == 2

def test_linear_search_in_loop():
    source = textwrap.dedent("""
        def check_whitelist(users, whitelist):
            for user in users:
                if user in whitelist:
                    print("Allowed")
    """)
    findings = analyze_function(source, "check_whitelist", "app.py")
    # Should flag whitelist because it sounds like a list
    assert any(f.pattern_type == "LINEAR_SEARCH_IN_LOOP" for f in findings)
    assert findings[0].complexity_before == "O(n²)"

def test_no_false_positive_for_set_lookup():
    source = textwrap.dedent("""
        def check_whitelist_set(users, whitelist_set):
            for user in users:
                if user in whitelist_set:
                    print("Allowed")
    """)
    findings = analyze_function(source, "check_whitelist_set", "app.py")
    # Since variable name contains "set", heuristic should skip
    assert not any(f.pattern_type == "LINEAR_SEARCH_IN_LOOP" for f in findings)

def test_count_in_loop():
    source = textwrap.dedent("""
        def get_counts(items):
            res = []
            for x in items:
                res.append(items.count(x))
            return res
    """)
    findings = analyze_function(source, "get_counts", "app.py")
    assert findings[0].pattern_type == "COUNT_IN_LOOP"
    assert findings[0].severity == "high"

def test_remove_in_loop():
    source = textwrap.dedent("""
        def filter_bad(items):
            for bad in bad_items:
                items.remove(bad)
    """)
    findings = analyze_function(source, "filter_bad", "app.py")
    assert findings[0].pattern_type == "REMOVE_IN_LOOP"

def test_string_concat_in_loop():
    source = textwrap.dedent("""
        def build_string(words):
            s = ""
            for w in words:
                s += w
            return s
    """)
    findings = analyze_function(source, "build_string", "app.py")
    assert findings[0].pattern_type == "STRING_CONCAT_LOOP"
    assert findings[0].improvement_class == "memory"

def test_sort_for_minmax():
    source = textwrap.dedent("""
        def get_max(items):
            return sorted(items)[-1]
    """)
    findings = analyze_function(source, "get_max", "app.py")
    assert findings[0].pattern_type == "SORT_FOR_MINMAX"
    assert findings[0].complexity_before == "O(n log n)"

def test_len_in_loop_condition():
    source = textwrap.dedent("""
        def pop_all(items):
            while len(items) > 0:
                items.pop()
    """)
    findings = analyze_function(source, "pop_all", "app.py")
    assert findings[0].pattern_type == "LEN_IN_LOOP_COND"

# ---------------------------------------------------------------------------
# Benchmark Generation
# ---------------------------------------------------------------------------

def test_benchmark_generator(tmp_path: Path):
    finding = AlgorithmFinding(
        function_name="slow_func",
        file="app.py",
        line=1,
        pattern_type="NESTED_LOOP",
        complexity_before="O(n²)",
        complexity_after="O(n)",
        nesting_depth=2,
        offending_lines=[1],
        offending_code="for i in x:",
        full_function_source="def slow_func(x):\\n  pass",
        severity="critical",
        confidence=0.9,
        improvement_class="time",
        description="Slow."
    )
    
    rewrite = AlgorithmRewrite(
        finding=finding,
        rewritten_function="def slow_func(x):\\n  pass # fast",
        why_slow="Because.",
        why_fast="Magic.",
        big_o_proof="n^2 -> n",
        memory_note="None",
        semantic_equivalence="CONFIRMED",
        benchmark_code="",
        estimated_speedup="100x",
        provider_used="test",
        patch=""
    )
    
    out = tmp_path / "perf_benchmark.py"
    BenchmarkGenerator.generate([rewrite], out)
    
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "def _run_benchmark_slow_func():" in content
    assert "timeit.timeit(" in content
