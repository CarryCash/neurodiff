"""Phase 6 — Cognitive Load Score Engine.

Quantifies the cognitive burden a diff places on a human reviewer, with
special focus on detecting AI-generated code patterns. Zero new dependencies
— uses only stdlib (re, ast, statistics) + networkx (Phase 3).
"""
from __future__ import annotations

import ast
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AIGeneratedScore:
    probability: float                    # 0.0 to 1.0
    confidence: Literal["high", "medium", "low"]
    signals_triggered: list[str]
    dominant_signal: str
    estimated_ai_lines: int
    human_lines: int
    signal_breakdown: dict[str, float]   # signal_name → contribution


@dataclass
class CognitiveFatigueIndex:
    total_score: float                    # 0–100
    grade: Literal["A", "B", "C", "D", "F"]
    component_scores: dict[str, float]   # A–F component breakdown
    review_time_estimate: str            # "~15 min", "~45 min", etc.
    reviewer_recommendation: str         # "1 reviewer", "2 reviewers", etc.


@dataclass
class BlastRadiusReport:
    epicenter_files: list[str]
    first_ring: list[str]
    second_ring: list[str]
    third_ring: list[str]
    total_affected: int
    risk_score: Literal["contained", "moderate", "wide", "critical"]
    hotspot: str | None
    hotspot_dependents: int | None


@dataclass
class CommitAntiPattern:
    name: str
    severity: Literal["high", "medium", "low"]
    description: str
    evidence: str


@dataclass
class CommitPatternReport:
    anti_patterns: list[CommitAntiPattern]
    commit_health_score: float            # 0.0–1.0 (1.0 = perfectly healthy)


@dataclass
class CognitiveReport:
    ai_generated: AIGeneratedScore
    fatigue_index: CognitiveFatigueIndex
    blast_radius: BlastRadiusReport
    commit_patterns: CommitPatternReport
    overall_verdict: Literal["safe", "caution", "danger"]


# ---------------------------------------------------------------------------
# AI-Generated Code Detector
# ---------------------------------------------------------------------------

# Common AI boilerplate phrases
_BOILERPLATE_PATTERNS = [
    r"This function\s",
    r"This method\s",
    r"This class\s",
    r"TODO:\s*Add error handling",
    r"TODO:\s*Add tests",
    r"#\s*Step \d+:",
    r"#\s*\d+\.",          # numbered step comments
    r"Returns:$",
    r"Args:$",
    r"Raises:$",
    r"Note:$",
    r"Example:$",
]
_BOILERPLATE_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _BOILERPLATE_PATTERNS]


def _extract_function_bodies(code: str) -> list[str]:
    """Return list of function body strings from source code."""
    bodies: list[str] = []
    try:
        tree = ast.parse(code)
        lines = code.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno - 1
                end = node.end_lineno or (start + 1)
                bodies.append("\n".join(lines[start:end]))
    except SyntaxError:
        pass
    return bodies


def _count_comment_lines(code: str) -> tuple[int, int]:
    """Return (comment_lines, total_lines) for a code block."""
    lines = code.splitlines()
    comment_lines = sum(
        1 for ln in lines
        if ln.strip().startswith("#") or ln.strip().startswith('"""') or ln.strip().startswith("'''")
    )
    return comment_lines, max(1, len(lines))


def _has_toplevel_try(func_body: str) -> bool:
    """Check if the function body is primarily a top-level try/except block."""
    try:
        tree = ast.parse(func_body)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = node.body
                # Ignore leading docstring
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                    body = body[1:]
                if body and isinstance(body[0], ast.Try):
                    return True
    except SyntaxError:
        pass
    return False


def _avg_identifier_length(code: str) -> float:
    """Return the average length of all identifiers in the code."""
    try:
        tree = ast.parse(code)
        names = [
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id not in ("None", "True", "False")
        ]
        if not names:
            return 0.0
        return sum(len(n) for n in names) / len(names)
    except SyntaxError:
        return 0.0


class AIDetector:
    """Heuristic AI-generated code classifier. No LLM calls."""

    WEIGHTS = {
        "density_spike": 0.25,
        "naming_uniformity": 0.20,
        "comment_density": 0.20,
        "boilerplate_patterns": 0.15,
        "structural_uniformity": 0.10,
        "try_except_wrapping": 0.10,
    }

    def analyze(
        self,
        new_code_blocks: list[str],   # added function bodies
        repo_baseline_lines_per_fn: float = 25.0,
        repo_baseline_identifier_len: float = 8.0,
    ) -> AIGeneratedScore:
        if not new_code_blocks:
            return AIGeneratedScore(
                probability=0.0, confidence="low", signals_triggered=[],
                dominant_signal="none", estimated_ai_lines=0, human_lines=0,
                signal_breakdown={},
            )

        contributions: dict[str, float] = {}

        # Signal 1 — Density Spike
        fn_line_counts = [len(b.splitlines()) for b in new_code_blocks]
        avg_new_lines = statistics.mean(fn_line_counts) if fn_line_counts else 0
        density_ratio = avg_new_lines / max(1, repo_baseline_lines_per_fn)
        contributions["density_spike"] = min(1.0, max(0.0, (density_ratio - 1.0) / 1.5))

        # Signal 2 — Naming Uniformity
        avg_id_lens = [_avg_identifier_length(b) for b in new_code_blocks]
        avg_id_len = statistics.mean(avg_id_lens) if avg_id_lens else 0
        # Score: long names (>18) = 1.0, baseline (~8) = 0.0
        contributions["naming_uniformity"] = min(1.0, max(0.0, (avg_id_len - repo_baseline_identifier_len) / 12.0))

        # Signal 3 — Comment Density
        comment_ratios = []
        for b in new_code_blocks:
            c, t = _count_comment_lines(b)
            comment_ratios.append(c / t)
        avg_comment_ratio = statistics.mean(comment_ratios) if comment_ratios else 0
        contributions["comment_density"] = min(1.0, max(0.0, (avg_comment_ratio - 0.10) / 0.20))

        # Signal 4 — Boilerplate Patterns
        all_code = "\n".join(new_code_blocks)
        boilerplate_hits = sum(bool(p.search(all_code)) for p in _BOILERPLATE_RE)
        contributions["boilerplate_patterns"] = min(1.0, boilerplate_hits / 5.0)

        # Signal 5 — Structural Uniformity (low std dev = AI signal)
        if len(fn_line_counts) >= 3:
            std_dev = statistics.stdev(fn_line_counts)
            # Low std dev (< 5) = high AI signal
            contributions["structural_uniformity"] = min(1.0, max(0.0, 1.0 - (std_dev / 15.0)))
        else:
            contributions["structural_uniformity"] = 0.0

        # Signal 6 — Try/Except Wrapping
        try_wrapped = sum(1 for b in new_code_blocks if _has_toplevel_try(b))
        try_ratio = try_wrapped / max(1, len(new_code_blocks))
        contributions["try_except_wrapping"] = min(1.0, max(0.0, (try_ratio - 0.3) / 0.4))

        # Compute weighted probability
        probability = sum(
            contributions[sig] * self.WEIGHTS[sig]
            for sig in self.WEIGHTS
        )
        probability = min(1.0, max(0.0, probability))

        signals_triggered = [s for s, v in contributions.items() if v > 0.4]
        dominant_signal = max(contributions, key=lambda s: contributions[s] * self.WEIGHTS[s])

        num_signals = len(signals_triggered)
        confidence: Literal["high", "medium", "low"] = (
            "high" if num_signals >= 3 else "medium" if num_signals >= 1 else "low"
        )

        total_lines = sum(len(b.splitlines()) for b in new_code_blocks)
        ai_lines = round(total_lines * probability)
        human_lines = total_lines - ai_lines

        return AIGeneratedScore(
            probability=round(probability, 3),
            confidence=confidence,
            signals_triggered=signals_triggered,
            dominant_signal=dominant_signal,
            estimated_ai_lines=ai_lines,
            human_lines=human_lines,
            signal_breakdown={k: round(v, 3) for k, v in contributions.items()},
        )


# ---------------------------------------------------------------------------
# Cognitive Fatigue Index (CFI)
# ---------------------------------------------------------------------------

_CONCERN_TYPES = {
    "auth": re.compile(r"auth|login|oauth|jwt|token|password|credential", re.I),
    "database": re.compile(r"db|database|model|migration|orm|sql|query|repo", re.I),
    "api": re.compile(r"api|router|endpoint|view|handler|request|response|http", re.I),
    "ui": re.compile(r"ui|template|render|component|widget|page|frontend", re.I),
    "config": re.compile(r"config|setting|env|environ|constant", re.I),
    "tests": re.compile(r"test_|_test|spec|fixture|mock", re.I),
    "migrations": re.compile(r"migration|alembic|schema|migrate", re.I),
    "utils": re.compile(r"util|helper|tool|common|shared|base", re.I),
    "models": re.compile(r"model|entity|domain|schema", re.I),
}


def _compute_grade(score: float) -> Literal["A", "B", "C", "D", "F"]:
    if score <= 20:
        return "A"
    if score <= 40:
        return "B"
    if score <= 60:
        return "C"
    if score <= 80:
        return "D"
    return "F"


def _compute_review_time(total_score: float, ai_prob: float) -> str:
    base_minutes = total_score * 1.5
    if ai_prob > 0.7:
        base_minutes *= 1.4
    if base_minutes < 20:
        return "~15 min"
    if base_minutes < 35:
        return f"~{round(base_minutes / 5) * 5} min"
    if base_minutes < 90:
        return f"~{round(base_minutes / 15) * 15} min"
    hours = base_minutes / 60
    if hours < 2:
        return f"~{round(hours * 2) / 2:.1f}h"
    return f"~{round(hours)}h"


def _compute_reviewer_recommendation(total_score: float) -> str:
    if total_score <= 30:
        return "1 reviewer"
    if total_score <= 60:
        return "2 reviewers"
    return "team review (3+ reviewers)"


def _path_distance(path_a: str, path_b: str) -> float:
    """Fallback if ChromaDB is not available -> calculate distance by path components."""
    parts_a = set(Path(path_a).parts)
    parts_b = set(Path(path_b).parts)
    
    union = parts_a | parts_b
    if not union:
        return 0.0
        
    intersection = parts_a & parts_b
    return 1.0 - (len(intersection) / len(union))

class CFICalculator:
    """Computes the Cognitive Fatigue Index (0–100)."""

    def compute(
        self,
        total_lines_changed: int,
        file_diffs: list[Any],         # FileDiff objects
        semantic_events: list[Any],    # SemanticEvent objects
        ai_score: AIGeneratedScore,
        chroma_available: bool = False,
    ) -> CognitiveFatigueIndex:
        components: dict[str, float] = {}

        # A. Volume Score (max 25)
        components["A_volume"] = min(25.0, total_lines_changed / 40.0)

        # B. Spread Score (max 20)
        dirs_touched = {
            str(Path(fd.path).parent)
            for fd in file_diffs
            if hasattr(fd, "path")
        }
        components["B_spread"] = min(20.0, len(dirs_touched) * 3.0)

        # C. Conceptual Distance Score (max 20)
        # Si ChromaDB no disponible -> distancia por ruta (fallback heurístico)
        paths_touched = list({getattr(fd, "path", "") for fd in file_diffs if hasattr(fd, "path")})
        
        if len(paths_touched) > 1:
            pairs = 0
            total_dist = 0.0
            for i in range(len(paths_touched)):
                for j in range(i + 1, len(paths_touched)):
                    dist = _path_distance(paths_touched[i], paths_touched[j])
                    total_dist += dist
                    pairs += 1
            
            avg_dist = total_dist / max(1, pairs)
            components["C_conceptual_distance"] = min(20.0, avg_dist * 20.0)
        else:
            components["C_conceptual_distance"] = 0.0
            components["C_conceptual_distance"] = 0.0

        # D. Context Switch Score (max 15)
        all_paths = " ".join(
            getattr(fd, "path", "") for fd in file_diffs
        )
        concerns_detected = sum(
            1 for pattern in _CONCERN_TYPES.values()
            if pattern.search(all_paths)
        )
        components["D_context_switches"] = min(15.0, concerns_detected * 3.0)

        # E. Complexity Delta Score (max 10)
        from neurodiff.core.semantic_events import FunctionAdded, FunctionModified
        total_complexity = 0
        for event in semantic_events:
            if isinstance(event, FunctionAdded):
                total_complexity += getattr(event, "cyclomatic_complexity", 1)
            elif isinstance(event, FunctionModified):
                after = getattr(event, "complexity_after", 0)
                before = getattr(event, "complexity_before", 0)
                total_complexity += max(0, after - before)
        components["E_complexity_delta"] = min(10.0, total_complexity / 10.0)

        # F. AI Amplification Score (max 10)
        components["F_ai_amplification"] = ai_score.probability * 10.0

        raw_total = sum(components.values())

        # AI code amplification multiplier
        if ai_score.probability > 0.7:
            raw_total = min(100.0, raw_total * 1.3)

        total_score = round(min(100.0, raw_total), 1)
        grade = _compute_grade(total_score)
        review_time = _compute_review_time(total_score, ai_score.probability)
        recommendation = _compute_reviewer_recommendation(total_score)

        return CognitiveFatigueIndex(
            total_score=total_score,
            grade=grade,
            component_scores={k: round(v, 1) for k, v in components.items()},
            review_time_estimate=review_time,
            reviewer_recommendation=recommendation,
        )


# ---------------------------------------------------------------------------
# Blast Radius Formalizer
# ---------------------------------------------------------------------------

def _risk_score_from_count(n: int) -> Literal["contained", "moderate", "wide", "critical"]:
    if n < 3:
        return "contained"
    if n <= 8:
        return "moderate"
    if n <= 20:
        return "wide"
    return "critical"


class BlastRadiusFormalizer:
    """Formalizes blast radius into structured concentric rings."""

    def compute(
        self,
        epicenter_files: list[str],
        graph: Any,   # nx.DiGraph where edge A→B means A imports B
    ) -> BlastRadiusReport:
        try:
            import networkx as nx
        except ImportError:
            return BlastRadiusReport(
                epicenter_files=epicenter_files, first_ring=[], second_ring=[],
                third_ring=[], total_affected=0, risk_score="contained",
                hotspot=None, hotspot_dependents=None,
            )

        # Convert file paths to module-like keys for graph lookup
        def path_to_mod(p: str) -> str:
            return p.replace("/", ".").replace("\\", ".").removesuffix(".py")

        epicenter_mods = {path_to_mod(f) for f in epicenter_files}

        first_ring: set[str] = set()
        second_ring: set[str] = set()
        third_ring: set[str] = set()

        for mod in epicenter_mods:
            if mod not in graph:
                continue
            # Direct importers of the epicenter module
            ring1 = set(graph.predecessors(mod)) - epicenter_mods
            first_ring.update(ring1)

        for mod in first_ring:
            if mod not in graph:
                continue
            ring2 = set(graph.predecessors(mod)) - epicenter_mods - first_ring
            second_ring.update(ring2)

        for mod in second_ring:
            if mod not in graph:
                continue
            ring3 = set(graph.predecessors(mod)) - epicenter_mods - first_ring - second_ring
            third_ring.update(ring3)

        total_affected = len(first_ring) + len(second_ring) + len(third_ring)
        risk_score = _risk_score_from_count(total_affected)

        # Find hotspot: the epicenter file with the most first-ring dependents
        hotspot: str | None = None
        hotspot_dependents: int | None = None
        for mod in epicenter_mods:
            if mod in graph:
                n = len(list(graph.predecessors(mod)))
                if hotspot_dependents is None or n > hotspot_dependents:
                    hotspot = mod
                    hotspot_dependents = n

        return BlastRadiusReport(
            epicenter_files=epicenter_files,
            first_ring=sorted(first_ring),
            second_ring=sorted(second_ring),
            third_ring=sorted(third_ring),
            total_affected=total_affected,
            risk_score=risk_score,
            hotspot=hotspot,
            hotspot_dependents=hotspot_dependents,
        )


# ---------------------------------------------------------------------------
# Commit Pattern Analyzer
# ---------------------------------------------------------------------------

_TEST_FILE_PATTERN = re.compile(r"test_|_test\.py$|/tests?/|spec[/_]", re.I)
_CORE_FILE_PATTERN = re.compile(r"/core/|/models/|/domain/", re.I)


class CommitPatternAnalyzer:
    """Detects anti-patterns in the commit structure."""

    def analyze(
        self,
        semantic_events: list[Any],
        file_diffs: list[Any],
        ai_score: AIGeneratedScore,
        context_switch_score: float,
        blast_radius: BlastRadiusReport,
    ) -> CommitPatternReport:
        from neurodiff.core.semantic_events import (
            FunctionAdded, FunctionModified, ImportAdded,
        )

        anti_patterns: list[CommitAntiPattern] = []
        total_lines_added = sum(
            getattr(fd, "added_lines", 0) for fd in file_diffs
        )
        total_lines_removed = sum(
            getattr(fd, "removed_lines", 0) for fd in file_diffs
        )

        fn_added = [e for e in semantic_events if isinstance(e, FunctionAdded)]
        fn_modified = [e for e in semantic_events if isinstance(e, FunctionModified)]
        imports_added = [e for e in semantic_events if isinstance(e, ImportAdded)]
        test_files_changed = [
            fd for fd in file_diffs
            if _TEST_FILE_PATTERN.search(getattr(fd, "path", ""))
        ]
        core_files_changed = [
            fd for fd in file_diffs
            if _CORE_FILE_PATTERN.search(getattr(fd, "path", ""))
        ]

        # Anti-pattern: "The Dump"
        if total_lines_added > 500 and ai_score.probability > 0.7:
            anti_patterns.append(CommitAntiPattern(
                name="The Dump",
                severity="high",
                description="Large AI-generated code drop with no iterative refinement",
                evidence=f"{total_lines_added} lines added, AI probability {ai_score.probability:.0%}",
            ))

        # Anti-pattern: "The Franken-commit"
        if context_switch_score > 9:
            concerns = [
                k for k, v in _CONCERN_TYPES.items()
                if v.search(" ".join(getattr(fd, "path", "") for fd in file_diffs))
            ]
            anti_patterns.append(CommitAntiPattern(
                name="The Franken-commit",
                severity="medium",
                description="Unrelated concerns bundled in a single commit",
                evidence=f"Detected concerns: {', '.join(concerns)}",
            ))

        # Anti-pattern: "The Ghost Refactor"
        sig_unchanged_mods = [
            e for e in fn_modified
            if not getattr(e, "signature_changed", True)
        ]
        if (len(sig_unchanged_mods) > 3 and total_lines_added > 100
                and len(fn_added) == 0):
            ratio = total_lines_added / max(1, len(sig_unchanged_mods))
            anti_patterns.append(CommitAntiPattern(
                name="The Ghost Refactor",
                severity="medium",
                description="High churn, low semantic change — possible cosmetic AI refactor",
                evidence=f"{len(sig_unchanged_mods)} functions modified without signature changes, {total_lines_added} lines added",
            ))

        # Anti-pattern: "The Dependency Creep"
        if len(imports_added) > 5:
            import_names = [getattr(e, "module", "?") for e in imports_added[:8]]
            anti_patterns.append(CommitAntiPattern(
                name="The Dependency Creep",
                severity="medium",
                description="Too many new imports introduced in a single diff",
                evidence=f"{len(imports_added)} new imports: {', '.join(import_names)}",
            ))

        # Anti-pattern: "The Test Desert"
        if len(fn_added) > 3 and not test_files_changed:
            anti_patterns.append(CommitAntiPattern(
                name="The Test Desert",
                severity="high",
                description="Significant code added with no corresponding test coverage",
                evidence=f"{len(fn_added)} functions added, 0 test files modified",
            ))

        # Anti-pattern: "The Depth Bomb"
        if core_files_changed and blast_radius.total_affected > 10:
            anti_patterns.append(CommitAntiPattern(
                name="The Depth Bomb",
                severity="high",
                description="Core module modification with extensive downstream impact",
                evidence=f"{len(core_files_changed)} core files changed, {blast_radius.total_affected} downstream modules affected",
            ))

        # Health score: deduct based on severity
        deductions = sum(
            0.25 if ap.severity == "high" else 0.12 if ap.severity == "medium" else 0.06
            for ap in anti_patterns
        )
        health_score = max(0.0, round(1.0 - deductions, 2))

        return CommitPatternReport(
            anti_patterns=anti_patterns,
            commit_health_score=health_score,
        )


# ---------------------------------------------------------------------------
# Cognitive Engine Runner
# ---------------------------------------------------------------------------

class CognitiveEngine:
    """Orchestrates Phase 6 Cognitive Load analysis."""

    def run(
        self,
        semantic_events: list[Any],
        file_diffs: list[Any],
        arch_report: Any | None = None,
        dependency_graph: Any | None = None,
    ) -> CognitiveReport:
        # Gather code blocks from added functions
        from neurodiff.core.semantic_events import FunctionAdded
        new_code_blocks = []
        for event in semantic_events:
            if isinstance(event, FunctionAdded):
                body = getattr(event, "body", None)
                if body:
                    new_code_blocks.append(body)

        # 1. AI Detector
        ai_detector = AIDetector()
        ai_score = ai_detector.analyze(new_code_blocks)

        # 2. CFI
        total_lines_changed = sum(
            getattr(fd, "added_lines", 0) + getattr(fd, "removed_lines", 0)
            for fd in file_diffs
        )
        cfi_calc = CFICalculator()
        cfi = cfi_calc.compute(
            total_lines_changed=total_lines_changed,
            file_diffs=file_diffs,
            semantic_events=semantic_events,
            ai_score=ai_score,
        )

        # 3. Blast Radius
        epicenter_files = [getattr(fd, "path", "") for fd in file_diffs]
        br_formalizer = BlastRadiusFormalizer()
        blast_radius = br_formalizer.compute(epicenter_files, dependency_graph or _empty_graph())

        # 4. Commit Pattern Analyzer
        cp_analyzer = CommitPatternAnalyzer()
        commit_patterns = cp_analyzer.analyze(
            semantic_events=semantic_events,
            file_diffs=file_diffs,
            ai_score=ai_score,
            context_switch_score=cfi.component_scores.get("D_context_switches", 0),
            blast_radius=blast_radius,
        )

        # 5. Overall verdict
        high_anti_patterns = any(ap.severity == "high" for ap in commit_patterns.anti_patterns)
        medium_anti_patterns = any(ap.severity == "medium" for ap in commit_patterns.anti_patterns)

        if cfi.total_score > 70 or high_anti_patterns or ai_score.probability > 0.85:
            verdict: Literal["safe", "caution", "danger"] = "danger"
        elif cfi.total_score > 40 or medium_anti_patterns or ai_score.probability > 0.5:
            verdict = "caution"
        else:
            verdict = "safe"

        return CognitiveReport(
            ai_generated=ai_score,
            fatigue_index=cfi,
            blast_radius=blast_radius,
            commit_patterns=commit_patterns,
            overall_verdict=verdict,
        )


def _empty_graph() -> Any:
    """Return an empty networkx DiGraph, gracefully."""
    try:
        import networkx as nx
        return nx.DiGraph()
    except ImportError:
        return None
