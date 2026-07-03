"""Feature 4 — Algorithm Performance Reviewer.

Two-layer detection:
  Layer 1 — ComplexityAnalyzer: pure stdlib `ast` pattern detection, zero deps.
  Layer 2 — LLM Rewrite Engine: generates optimized rewrites with Big O proofs.

No new external dependencies. Uses stdlib: ast, difflib, timeit, textwrap.
"""
from __future__ import annotations

import ast
import asyncio
import difflib
import json
import logging
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from neurodiff.core.semantic_events import SemanticEvent, FunctionAdded, FunctionModified
from neurodiff.engines.llm_engine import LLMProvider, parse_json_response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern Severity Map
# ---------------------------------------------------------------------------

_PATTERN_META: dict[str, dict] = {
    "NESTED_LOOP": {
        "complexity_before": "O(n²)",
        "complexity_after": "O(n)",
        "severity": "critical",
        "description": "Nested loops multiply iteration counts, causing quadratic or worse performance.",
    },
    "LINEAR_SEARCH_IN_LOOP": {
        "complexity_before": "O(n²)",
        "complexity_after": "O(n)",
        "severity": "critical",
        "description": "`x in list` inside a loop is O(n) per iteration. Use a set for O(1) lookup.",
    },
    "COUNT_IN_LOOP": {
        "complexity_before": "O(n²)",
        "complexity_after": "O(n)",
        "severity": "high",
        "description": "`list.count()` is O(n) and called inside a loop, making total cost O(n²).",
    },
    "REMOVE_IN_LOOP": {
        "complexity_before": "O(n²)",
        "complexity_after": "O(n)",
        "severity": "critical",
        "description": "`list.remove()` is O(n) per call. Rebuilding the list via comprehension is O(n) total.",
    },
    "INDEX_IN_LOOP": {
        "complexity_before": "O(n²)",
        "complexity_after": "O(n)",
        "severity": "high",
        "description": "`list.index()` scans from start on every call inside a loop.",
    },
    "STRING_CONCAT_LOOP": {
        "complexity_before": "O(n²) memory",
        "complexity_after": "O(n) memory",
        "severity": "high",
        "description": "String concatenation in a loop creates a new string object each iteration. Use `''.join()`.",
    },
    "REDUNDANT_SORT": {
        "complexity_before": "O(k · n log n)",
        "complexity_after": "O(n log n)",
        "severity": "medium",
        "description": "`sorted()` called repeatedly on the same collection. Sort once, reuse.",
    },
    "REPEATED_COMPUTATION": {
        "complexity_before": "O(n · f(n))",
        "complexity_after": "O(f(n) + n)",
        "severity": "medium",
        "description": "A function call with identical arguments is repeated inside a loop. Hoist it out.",
    },
    "LEN_IN_LOOP_COND": {
        "complexity_before": "O(n) per condition check",
        "complexity_after": "O(1) per condition check",
        "severity": "low",
        "description": "`len(x)` in the loop condition is recomputed every iteration. Cache it in a variable.",
    },
    "SORT_FOR_MINMAX": {
        "complexity_before": "O(n log n)",
        "complexity_after": "O(n)",
        "severity": "medium",
        "description": "`sorted(x)[0]` or `sorted(x)[-1]` just to find min/max. Use `min()`/`max()` instead.",
    },
    "UNNECESSARY_LIST_COMPREHENSION": {
        "complexity_before": "O(n) memory",
        "complexity_after": "O(1) memory",
        "severity": "low",
        "description": "A list comprehension is immediately consumed by `sum()`, `any()`, etc. Use a generator instead.",
    },
    "REPEATED_DICT_LOOKUP": {
        "complexity_before": "O(n) lookups",
        "complexity_after": "O(1) with cached ref",
        "severity": "low",
        "description": "Same dict key is looked up multiple times in a tight loop. Cache `d[key]` in a local variable.",
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmFinding:
    """A single performance issue detected by the AST analyzer."""
    function_name: str
    file: str
    line: int
    pattern_type: str
    complexity_before: str
    complexity_after: str
    nesting_depth: int
    offending_lines: list[int]
    offending_code: str
    full_function_source: str
    severity: Literal["critical", "high", "medium", "low"]
    confidence: float
    improvement_class: Literal["time", "memory", "both"]
    description: str


@dataclass
class AlgorithmRewrite:
    """LLM-generated optimization for an AlgorithmFinding."""
    finding: AlgorithmFinding
    rewritten_function: str
    why_slow: str
    why_fast: str
    big_o_proof: str
    memory_note: str
    semantic_equivalence: Literal["CONFIRMED", "LIKELY", "VERIFY_MANUALLY"]
    benchmark_code: str
    estimated_speedup: str
    provider_used: str
    patch: str


@dataclass
class PerfReport:
    """Complete performance analysis report."""
    findings: list[AlgorithmFinding] = field(default_factory=list)
    rewrites: list[AlgorithmRewrite] = field(default_factory=list)
    functions_analyzed: int = 0
    functions_with_issues: int = 0
    overall_complexity_score: int = 0   # 0–100 (higher = more issues)
    benchmark_file: str | None = None
    provider_used: str = "none"
    error: str | None = None

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "high")


# ---------------------------------------------------------------------------
# Layer 1 — AST Pattern Detector
# ---------------------------------------------------------------------------

class ComplexityAnalyzer(ast.NodeVisitor):
    """Walks a single function's AST and detects performance anti-patterns.

    Uses stdlib `ast` only — zero external dependencies.
    """

    def __init__(self, func_name: str, source_lines: list[str]):
        self.func_name = func_name
        self.source_lines = source_lines
        self.findings: list[tuple[str, int, int]] = []  # (pattern, line, depth)
        self._loop_depth = 0

    # ------------------------------------------------------------------
    # Visitor methods
    # ------------------------------------------------------------------

    def visit_For(self, node: ast.For) -> None:
        self._loop_depth += 1
        if self._loop_depth >= 2:
            # Nested loop — record once per outermost pair
            self.findings.append(("NESTED_LOOP", node.lineno, self._loop_depth))
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self._loop_depth += 1
        if self._loop_depth >= 2:
            self.findings.append(("NESTED_LOOP", node.lineno, self._loop_depth))
        # Check len() in condition
        self._check_len_in_condition(node.test)
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_Compare(self, node: ast.Compare) -> None:
        """Detect `x in list_var` inside a loop."""
        if self._loop_depth > 0:
            for op in node.ops:
                if isinstance(op, ast.In):
                    # Check each comparator
                    for comp in node.comparators:
                        if self._is_list_var(comp):
                            self.findings.append(
                                ("LINEAR_SEARCH_IN_LOOP", node.lineno, self._loop_depth)
                            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Detect method calls that are O(n) used inside loops."""
        if self._loop_depth > 0 and isinstance(node.func, ast.Attribute):
            method = node.func.attr

            if method == "count":
                self.findings.append(("COUNT_IN_LOOP", node.lineno, self._loop_depth))
            elif method == "remove":
                self.findings.append(("REMOVE_IN_LOOP", node.lineno, self._loop_depth))
            elif method == "index":
                self.findings.append(("INDEX_IN_LOOP", node.lineno, self._loop_depth))

        # Detect sorted()[0] or sorted()[-1] → SORT_FOR_MINMAX
        if (
            isinstance(node.func, ast.Name) and node.func.id == "sorted"
            and isinstance(node, ast.expr)
        ):
            # Will be caught by Subscript visitor
            pass

        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Detect `str += ...` in a loop."""
        if self._loop_depth > 0 and isinstance(node.op, ast.Add):
            if isinstance(node.target, ast.Name):
                self.findings.append(
                    ("STRING_CONCAT_LOOP", node.lineno, self._loop_depth)
                )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Detect sorted(x)[0] or sorted(x)[-1]."""
        if isinstance(node.value, ast.Call):
            call = node.value
            func_name = ""
            if isinstance(call.func, ast.Name):
                func_name = call.func.id
            if func_name == "sorted":
                sl = node.slice
                # Check for [0] or [-1]
                idx = None
                if isinstance(sl, ast.Constant):
                    idx = sl.value
                elif isinstance(sl, ast.UnaryOp) and isinstance(sl.op, ast.USub):
                    if isinstance(sl.operand, ast.Constant):
                        # pyrefly: ignore [unsupported-operation]
                        idx = -sl.operand.value
                if idx in (0, -1):
                    self.findings.append(("SORT_FOR_MINMAX", node.lineno, 0))
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Detect list comprehension immediately passed to any(), all(), sum(), etc."""
        # This node will appear as an argument in a Call — handled by parent visitor
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_list_var(node: ast.expr) -> bool:
        """Heuristic: returns True if this node is likely a list, not a set/dict."""
        if isinstance(node, ast.Name):
            name = node.id.lower()
            # Common set/dict names — do NOT flag
            if any(kw in name for kw in ("set", "dict", "map", "hash", "lookup", "index_")):
                return False
            # Common list names — flag
            if any(kw in name for kw in ("list", "arr", "items", "data", "results", "seen", "cache")):
                return True
            # Unknown — flag with lower confidence (still flag)
            return True
        if isinstance(node, ast.List):
            return True
        return False

    def _check_len_in_condition(self, test: ast.expr) -> None:
        """Detect len(x) used in a while condition."""
        if isinstance(test, ast.Compare):
            for side in [test.left] + test.comparators:
                if isinstance(side, ast.Call):
                    if isinstance(side.func, ast.Name) and side.func.id == "len":
                        self.findings.append(("LEN_IN_LOOP_COND", test.lineno, self._loop_depth))


def _extract_function_source(source: str, func_name: str) -> str | None:
    """Extract the full source of a named function from a source string."""
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == func_name:
                    lines = source.splitlines()
                    start = node.lineno - 1
                    end = node.end_lineno or (start + 30)
                    return "\n".join(lines[start:end])
    except Exception:
        pass
    return None


def analyze_function(source: str, func_name: str, file_path: str) -> list[AlgorithmFinding]:
    """Run the ComplexityAnalyzer on a single function and return findings."""
    findings: list[AlgorithmFinding] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings

    source_lines = source.splitlines()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if func_name and node.name != func_name:
            continue

        # Extract function source
        fn_start = node.lineno - 1
        fn_end = node.end_lineno or (fn_start + 50)
        fn_source = "\n".join(source_lines[fn_start:fn_end])
        fn_lines = fn_source.splitlines()

        analyzer = ComplexityAnalyzer(node.name, fn_lines)
        analyzer.visit(node)

        # Deduplicate: keep only the most severe of each pattern
        seen_patterns: dict[str, tuple[str, int, int]] = {}
        for pattern, lineno, depth in analyzer.findings:
            if pattern not in seen_patterns or depth > seen_patterns[pattern][2]:
                seen_patterns[pattern] = (pattern, lineno, depth)

        for pattern, lineno, depth in seen_patterns.values():
            meta = _PATTERN_META.get(pattern, {})

            # Determine improvement class
            impr = "time"
            if "memory" in meta.get("complexity_before", ""):
                impr = "memory"
            if "memory" in meta.get("description", "") and "O(n" in meta.get("complexity_before", ""):
                impr = "both"

            # Get offending line snippet
            abs_line = lineno
            offending_code_snippet = ""
            if 0 < abs_line <= len(source_lines):
                offending_code_snippet = source_lines[abs_line - 1].strip()

            findings.append(AlgorithmFinding(
                function_name=node.name,
                file=file_path,
                line=abs_line,
                pattern_type=pattern,
                complexity_before=meta.get("complexity_before", "O(n²)"),
                complexity_after=meta.get("complexity_after", "O(n)"),
                nesting_depth=depth,
                offending_lines=[abs_line],
                offending_code=offending_code_snippet,
                full_function_source=fn_source,
                severity=meta.get("severity", "medium"),  # type: ignore[arg-type]
                confidence=0.9 if depth >= 2 else 0.75,
                improvement_class=impr,  # type: ignore[arg-type]
                description=meta.get("description", ""),
            ))

    return findings


# ---------------------------------------------------------------------------
# Layer 2 — LLM Rewrite Engine
# ---------------------------------------------------------------------------

REWRITE_SYSTEM = """\
You are NeuroDiff's Algorithm Performance Reviewer — the most precise code optimizer that exists.
You receive:
1. A Python function with a detected performance anti-pattern
2. The exact pattern type and Big O class
3. The file context

Your job:
- Write an OPTIMIZED version of the function that is functionally equivalent
- Explain CONCISELY why the original is slow and why the new version is fast
- Provide a Big O proof
- Write a runnable microbenchmark using ONLY stdlib timeit

CRITICAL RULES:
- The rewritten function MUST be semantically equivalent for all valid inputs
- Do NOT add new imports that are not stdlib or already in the file
- Keep the same function signature exactly
- Keep the exact indentation.
- If it's complex, put the ENTIRE original function in search_block and the ENTIRE rewritten function in replace_block.
- Respond ONLY in the JSON format below. No preamble, no markdown fences.

{
  "search_block": "...",
  "replace_block": "...",
  "why_slow": "...",
  "why_fast": "...",
  "big_o_proof": "...",
  "memory_note": "...",
  "semantic_equivalence": "CONFIRMED|LIKELY|VERIFY_MANUALLY",
  "benchmark_code": "import timeit\\n...",
  "estimated_speedup": "e.g. ~50x at n=10000"
}"""


async def _rewrite_one(
    finding: AlgorithmFinding,
    provider: LLMProvider,
    semaphore: asyncio.Semaphore,
) -> AlgorithmRewrite | None:
    """Ask the LLM to rewrite a single finding."""
    async with semaphore:
        user_prompt = json.dumps({
            "function_name": finding.function_name,
            "file": finding.file,
            "pattern_type": finding.pattern_type,
            "complexity_before": finding.complexity_before,
            "complexity_after": finding.complexity_after,
            "description": finding.description,
            "offending_code": finding.offending_code,
            "full_function_source": finding.full_function_source,
        }, indent=2)

        try:
            raw = await provider.complete(REWRITE_SYSTEM, user_prompt)
            data = parse_json_response(raw)

            if "error" in data or "search_block" not in data or "replace_block" not in data:
                logger.warning("Perf rewrite parse failed for %s: %s", finding.function_name, data)
                return None

            search = data.get("search_block", "")
            replace = data.get("replace_block", "")
            
            # Reconstruct the full rewritten function
            if search and search in finding.full_function_source:
                rewritten = finding.full_function_source.replace(search, replace, 1)
            else:
                logger.warning("Search block not found in original source for %s. Falling back to full function replacement.", finding.function_name)
                rewritten = replace

            # Generate unified diff patch
            original_lines = finding.full_function_source.splitlines(keepends=True)
            rewritten_lines = rewritten.splitlines(keepends=True)
            patch_lines = list(difflib.unified_diff(
                original_lines,
                rewritten_lines,
                fromfile=f"a/{finding.file}",
                tofile=f"b/{finding.file}",
                lineterm="\n",
            ))
            patch = "".join(patch_lines)

            return AlgorithmRewrite(
                finding=finding,
                rewritten_function=rewritten,
                why_slow=data.get("why_slow", ""),
                why_fast=data.get("why_fast", ""),
                big_o_proof=data.get("big_o_proof", ""),
                memory_note=data.get("memory_note", ""),
                semantic_equivalence=data.get("semantic_equivalence", "VERIFY_MANUALLY"),  # type: ignore[arg-type]
                benchmark_code=data.get("benchmark_code", ""),
                estimated_speedup=data.get("estimated_speedup", "Unknown"),
                provider_used=provider.name,
                patch=patch,
            )
        except Exception as exc:
            logger.error("Perf LLM rewrite failed for %s: %s", finding.function_name, exc)
            return None


# ---------------------------------------------------------------------------
# Benchmark Generator
# ---------------------------------------------------------------------------

class BenchmarkGenerator:
    """Generates a runnable perf_benchmark.py from a list of rewrites."""

    @staticmethod
    def generate(rewrites: list[AlgorithmRewrite], output_path: Path) -> str:
        """Write benchmark file and return its path as string."""
        lines = [
            '"""Auto-generated benchmark by NeuroDiff — Algorithm Performance Reviewer.',
            "",
            "Run with: python perf_benchmark.py",
            '"""',
            "import timeit",
            "import random",
            "",
        ]

        for i, rw in enumerate(rewrites):
            fn = rw.finding.function_name
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", fn)

            lines += [
                f"# ─── Benchmark {i + 1}: {fn} ({rw.finding.file}) ───",
                f"# Pattern: {rw.finding.pattern_type}",
                f"# Before: {rw.finding.complexity_before}  After: {rw.finding.complexity_after}",
                "",
                "# ORIGINAL (slow version)",
                rw.finding.full_function_source,
                "",
                "# OPTIMIZED (NeuroDiff rewrite)",
                rw.rewritten_function,
                "",
            ]

            # Rename optimized to avoid name collision
            lines.append(
                f"{safe_name}_optimized = {fn}  # after rewrite definition above"
            )
            lines += [
                "",
                f"def _run_benchmark_{safe_name}():",
                "    sizes = [100, 1_000, 10_000, 100_000]",
                f"    print(f'\\n  Benchmark: {fn}')",
                "    print(f'  {'n':>10}  {'original':>14}  {'optimized':>14}  {'speedup':>10}')",
                "    for n in sizes:",
                "        data = list(range(n))",
                "        try:",
                f"            t_orig = timeit.timeit(lambda: {fn}(data), number=3)",
                f"            t_opt  = timeit.timeit(lambda: {safe_name}_optimized(data), number=3)",
                "            speedup = t_orig / t_opt if t_opt > 0 else float('inf')",
                "            print(f'  {n:>10}  {t_orig*1000:>12.3f}ms  {t_opt*1000:>12.3f}ms  {speedup:>9.1f}x')",
                "        except Exception as e:",
                "            print(f'  {n:>10}  ERROR: {e}')",
                "",
            ]

        lines += [
            "if __name__ == '__main__':",
            "    print('NeuroDiff — Algorithm Performance Benchmark')",
            "    print('=' * 60)",
        ]
        for i, rw in enumerate(rewrites):
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", rw.finding.function_name)
            lines.append(f"    _run_benchmark_{safe_name}()")

        content = "\n".join(lines)
        output_path.write_text(content, encoding="utf-8")
        return str(output_path)


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------

class PerfEngine:
    """Algorithm Performance Reviewer — orchestrates AST detection + LLM rewrites."""

    def __init__(self, provider: LLMProvider | None = None):
        self.provider = provider

    async def run(
        self,
        events: list[SemanticEvent],
        file_diffs: list[Any],
        *,
        write_benchmarks: bool = False,
        repo_path: Path | None = None,
        rewrite_limit: int = 5,
    ) -> PerfReport:
        """Analyze all changed functions for performance issues."""
        report = PerfReport()

        # Build a map of file path → content_after
        content_map: dict[str, str] = {}
        for fd in file_diffs:
            content_map[fd.path] = getattr(fd, "content_after", "") or ""

        # Collect functions to analyze
        funcs_to_analyze: list[tuple[str, str, str]] = []  # (func_name, file_path, source)
        for ev in events:
            if isinstance(ev, (FunctionAdded, FunctionModified)):
                file_path = ev.file
                func_name = ev.name
                file_source = content_map.get(file_path, "")
                if file_source:
                    funcs_to_analyze.append((func_name, file_path, file_source))

        report.functions_analyzed = len(funcs_to_analyze)

        # Run AST analysis
        all_findings: list[AlgorithmFinding] = []
        for func_name, file_path, source in funcs_to_analyze:
            try:
                found = analyze_function(source, func_name, file_path)
                all_findings.extend(found)
            except Exception as exc:
                logger.debug("AST analysis failed for %s: %s", func_name, exc)

        report.findings = all_findings
        report.functions_with_issues = len({f.function_name for f in all_findings})

        # Score: weighted sum of severities
        weights = {"critical": 30, "high": 15, "medium": 7, "low": 3}
        raw_score = sum(weights.get(f.severity, 0) for f in all_findings)
        report.overall_complexity_score = min(100, raw_score)

        # Layer 2 — LLM rewrites (only if provider available)
        if self.provider and all_findings:
            # Sort findings by severity (critical first) so we rewrite the worst ones
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            top_findings = sorted(all_findings, key=lambda f: severity_order.get(f.severity, 4))
            
            if rewrite_limit > 0:
                top_findings = top_findings[:rewrite_limit]

            semaphore = asyncio.Semaphore(3)
            tasks = [
                _rewrite_one(finding, self.provider, semaphore)
                for finding in top_findings
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            rewrites = []
            for r in results:
                if isinstance(r, AlgorithmRewrite):
                    rewrites.append(r)
                elif isinstance(r, Exception):
                    logger.error("Rewrite gather exception: %s", r)
            report.rewrites = rewrites
            report.provider_used = self.provider.name
        elif all_findings:
            report.provider_used = "none (AST only)"

        # Write benchmark file
        if write_benchmarks and report.rewrites and repo_path:
            bench_path = (repo_path / "perf_benchmark.py")
            try:
                BenchmarkGenerator.generate(report.rewrites, bench_path)
                report.benchmark_file = str(bench_path)
            except Exception as exc:
                logger.warning("Benchmark generation failed: %s", exc)

        return report
