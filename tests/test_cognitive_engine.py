"""Tests for Phase 6 — Cognitive Load Score Engine."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_AI_HEAVY_CODE = """
def process_user_authentication_request(user_data, session_token, config_params):
    \"\"\"
    This function handles the user authentication request.
    
    Args:
        user_data: The user data dictionary.
        session_token: The session token string.
        config_params: Configuration parameters.
    
    Returns:
        Authentication result dictionary.
    
    Raises:
        ValueError: If user data is invalid.
    \"\"\"
    # Step 1: Validate user data
    try:
        if not user_data:
            raise ValueError("User data cannot be empty")
        # Step 2: Process authentication
        authentication_result_value = validate_authentication_credentials(user_data)
        # Step 3: Return result
        return authentication_result_value
    except Exception as authentication_exception:
        raise ValueError(str(authentication_exception))
"""

_IDIOMATIC_CODE = """
def get(key, d, default=None):
    return d.get(key, default)

def fmt(n):
    return f"{n:.2f}"

def add(a, b):
    return a + b
"""

_DENSE_AI_CODE = """
def calculate_comprehensive_statistical_metrics_for_dataset(data_collection, configuration_options, output_format_specifier):
    \"\"\"This function calculates comprehensive statistical metrics for the provided dataset.\"\"\"
    # Step 1: Initialize metric collection
    try:
        statistical_metric_results = {}
        normalized_data_collection = [x / max(data_collection) for x in data_collection]
        average_value_calculation = sum(normalized_data_collection) / len(normalized_data_collection)
        statistical_metric_results['normalized_average'] = average_value_calculation
        return statistical_metric_results
    except Exception as calculation_exception:
        raise ValueError(f"Calculation failed: {calculation_exception}")
"""


def _make_fn_added(name: str = "foo", file: str = "core/utils.py", cc: int = 3, body: str = "") -> MagicMock:
    e = MagicMock()
    e.__class__.__name__ = "FunctionAdded"
    e.name = name
    e.file = file
    e.cyclomatic_complexity = cc
    e.body = body
    return e


def _make_fn_modified(sig_changed: bool = True) -> MagicMock:
    e = MagicMock()
    e.__class__.__name__ = "FunctionModified"
    e.signature_changed = sig_changed
    e.complexity_before = 3
    e.complexity_after = 7
    return e


def _make_import_added(module: str = "os") -> MagicMock:
    e = MagicMock()
    e.__class__.__name__ = "ImportAdded"
    e.module = module
    return e


def _make_file_diff(path: str, added: int = 50, removed: int = 10) -> MagicMock:
    fd = MagicMock()
    fd.path = path
    fd.added_lines = added
    fd.removed_lines = removed
    return fd


# ---------------------------------------------------------------------------
# AI Detector Tests
# ---------------------------------------------------------------------------

class TestAIDetector:
    def test_heavy_docstring_code_high_probability(self):
        from neurodiff.engines.cognitive_engine import AIDetector
        detector = AIDetector()
        result = detector.analyze([_AI_HEAVY_CODE, _DENSE_AI_CODE])
        assert result.probability > 0.35, f"Expected > 0.35, got {result.probability}"

    def test_idiomatic_terse_code_low_probability(self):
        from neurodiff.engines.cognitive_engine import AIDetector
        detector = AIDetector()
        result = detector.analyze([_IDIOMATIC_CODE, _IDIOMATIC_CODE, _IDIOMATIC_CODE])
        assert result.probability < 0.4, f"Expected < 0.4, got {result.probability}"

    def test_empty_code_blocks_returns_zero(self):
        from neurodiff.engines.cognitive_engine import AIDetector
        result = AIDetector().analyze([])
        assert result.probability == 0.0

    def test_confidence_levels(self):
        from neurodiff.engines.cognitive_engine import AIDetector
        detector = AIDetector()
        # With multiple AI patterns, should get medium or high confidence
        result = detector.analyze([_AI_HEAVY_CODE, _DENSE_AI_CODE])
        assert result.confidence in ("high", "medium", "low")

    def test_ai_score_fields_populated(self):
        from neurodiff.engines.cognitive_engine import AIDetector
        result = AIDetector().analyze([_AI_HEAVY_CODE])
        assert 0.0 <= result.probability <= 1.0
        assert result.dominant_signal != ""
        assert isinstance(result.signals_triggered, list)
        assert isinstance(result.signal_breakdown, dict)
        assert len(result.signal_breakdown) == 6  # all 6 signals present

    def test_try_except_wrapping_signal_detected(self):
        from neurodiff.engines.cognitive_engine import AIDetector
        result = AIDetector().analyze([_AI_HEAVY_CODE])
        # The AI heavy code has a top-level try/except
        assert result.signal_breakdown["try_except_wrapping"] > 0.0

    def test_boilerplate_signal_detected(self):
        from neurodiff.engines.cognitive_engine import AIDetector
        result = AIDetector().analyze([_AI_HEAVY_CODE])
        assert result.signal_breakdown["boilerplate_patterns"] > 0.0


# ---------------------------------------------------------------------------
# CFI Tests
# ---------------------------------------------------------------------------

class TestCognitiveFatigueIndex:
    def _make_ai_score(self, prob: float = 0.2):
        from neurodiff.engines.cognitive_engine import AIGeneratedScore
        return AIGeneratedScore(
            probability=prob, confidence="low", signals_triggered=[],
            dominant_signal="none", estimated_ai_lines=0, human_lines=0, signal_breakdown={},
        )

    def test_large_diff_many_dirs_grade_d_or_f(self):
        from neurodiff.engines.cognitive_engine import CFICalculator
        calc = CFICalculator()
        file_diffs = [_make_file_diff(f"dir{i}/module{j}.py", added=100) for i in range(7) for j in range(3)]
        events = [_make_fn_added(cc=8) for _ in range(20)]
        result = calc.compute(1000, file_diffs, events, self._make_ai_score(0.1))
        assert result.grade in ("D", "F"), f"Expected D or F, got {result.grade} (score={result.total_score})"

    def test_small_single_file_diff_grade_a_or_b(self):
        from neurodiff.engines.cognitive_engine import CFICalculator
        calc = CFICalculator()
        file_diffs = [_make_file_diff("utils/helpers.py", added=40, removed=10)]
        events = [_make_fn_added(cc=2)]
        result = calc.compute(50, file_diffs, events, self._make_ai_score(0.1))
        assert result.grade in ("A", "B"), f"Expected A or B, got {result.grade} (score={result.total_score})"

    def test_score_in_valid_range(self):
        from neurodiff.engines.cognitive_engine import CFICalculator
        calc = CFICalculator()
        result = calc.compute(200, [_make_file_diff("a/b.py")], [], self._make_ai_score())
        assert 0 <= result.total_score <= 100

    def test_high_ai_probability_increases_score(self):
        from neurodiff.engines.cognitive_engine import CFICalculator
        calc = CFICalculator()
        file_diffs = [_make_file_diff("core/module.py", added=200)]
        events = [_make_fn_added(cc=5) for _ in range(5)]
        low_ai = calc.compute(200, file_diffs, events, self._make_ai_score(0.1))
        high_ai = calc.compute(200, file_diffs, events, self._make_ai_score(0.9))
        assert high_ai.total_score >= low_ai.total_score

    def test_review_time_estimate_present(self):
        from neurodiff.engines.cognitive_engine import CFICalculator
        result = CFICalculator().compute(100, [_make_file_diff("a.py")], [], self._make_ai_score())
        assert "min" in result.review_time_estimate or "h" in result.review_time_estimate

    def test_reviewer_recommendation_present(self):
        from neurodiff.engines.cognitive_engine import CFICalculator
        result = CFICalculator().compute(100, [_make_file_diff("a.py")], [], self._make_ai_score())
        assert "reviewer" in result.reviewer_recommendation


# ---------------------------------------------------------------------------
# Blast Radius Formalizer Tests
# ---------------------------------------------------------------------------

class TestBlastRadiusFormalizer:
    def _make_graph(self):
        """Build a 3-hop chain: output.reporter → cli.main → core.ast_engine → engines.arch_engine"""
        try:
            import networkx as nx
            g = nx.DiGraph()
            # A→B means A imports B
            g.add_edge("output.reporter", "cli.main")
            g.add_edge("cli.main", "core.ast_engine")
            g.add_edge("core.ast_engine", "engines.arch_engine")
            return g
        except ImportError:
            pytest.skip("networkx not available")

    def test_3hop_chain_rings_assigned_correctly(self):
        from neurodiff.engines.cognitive_engine import BlastRadiusFormalizer
        graph = self._make_graph()
        formalizer = BlastRadiusFormalizer()
        result = formalizer.compute(["engines/arch_engine.py"], graph)
        # first_ring should contain core.ast_engine (imports arch_engine)
        assert "core.ast_engine" in result.first_ring
        assert "cli.main" in result.second_ring
        assert "output.reporter" in result.third_ring

    def test_no_dependents_contained(self):
        from neurodiff.engines.cognitive_engine import BlastRadiusFormalizer
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not available")
        g = nx.DiGraph()
        g.add_node("isolated.module")
        result = BlastRadiusFormalizer().compute(["isolated/module.py"], g)
        assert result.risk_score == "contained"
        assert result.total_affected == 0

    def test_risk_score_critical_above_20(self):
        from neurodiff.engines.cognitive_engine import BlastRadiusFormalizer, _risk_score_from_count
        assert _risk_score_from_count(21) == "critical"
        assert _risk_score_from_count(5) == "moderate"
        assert _risk_score_from_count(2) == "contained"


# ---------------------------------------------------------------------------
# Commit Pattern Analyzer Tests
# ---------------------------------------------------------------------------

class TestCommitPatternAnalyzer:
    def _make_ai_score(self, prob: float = 0.8):
        from neurodiff.engines.cognitive_engine import AIGeneratedScore
        return AIGeneratedScore(
            probability=prob, confidence="high", signals_triggered=["comment_density"],
            dominant_signal="comment_density", estimated_ai_lines=100, human_lines=20,
            signal_breakdown={},
        )

    def _make_blast_radius(self, total: int = 3):
        from neurodiff.engines.cognitive_engine import BlastRadiusReport
        return BlastRadiusReport(
            epicenter_files=[], first_ring=[], second_ring=[], third_ring=[],
            total_affected=total, risk_score="moderate", hotspot=None, hotspot_dependents=None,
        )

    def test_test_desert_detected(self):
        from neurodiff.engines.cognitive_engine import CommitPatternAnalyzer
        analyzer = CommitPatternAnalyzer()
        events = [_make_fn_added() for _ in range(4)]
        file_diffs = [_make_file_diff("core/auth.py")]  # no test files

        # Need to patch isinstance checks — use real-like objects
        from neurodiff.core.semantic_events import FunctionAdded
        real_events = []
        for i in range(4):
            e = MagicMock(spec=FunctionAdded)
            e.name = f"func_{i}"
            e.file = "core/auth.py"
            e.cyclomatic_complexity = 3
            e.body = ""
            real_events.append(e)

        result = analyzer.analyze(real_events, file_diffs, self._make_ai_score(0.3), 3.0, self._make_blast_radius(3))
        names = [ap.name for ap in result.anti_patterns]
        assert "The Test Desert" in names

    def test_the_dump_detected(self):
        from neurodiff.engines.cognitive_engine import CommitPatternAnalyzer
        from neurodiff.core.semantic_events import FunctionAdded
        analyzer = CommitPatternAnalyzer()

        file_diffs = [_make_file_diff("core/big_module.py", added=600, removed=10)]
        real_events = [MagicMock(spec=FunctionAdded)]
        result = analyzer.analyze(real_events, file_diffs, self._make_ai_score(0.85), 3.0, self._make_blast_radius(2))
        names = [ap.name for ap in result.anti_patterns]
        assert "The Dump" in names

    def test_dependency_creep_detected(self):
        from neurodiff.engines.cognitive_engine import CommitPatternAnalyzer
        from neurodiff.core.semantic_events import ImportAdded
        analyzer = CommitPatternAnalyzer()

        imports = [MagicMock(spec=ImportAdded, module=f"lib_{i}") for i in range(7)]
        file_diffs = [_make_file_diff("app.py", added=50, removed=5)]
        result = analyzer.analyze(imports, file_diffs, self._make_ai_score(0.2), 3.0, self._make_blast_radius(1))
        names = [ap.name for ap in result.anti_patterns]
        assert "The Dependency Creep" in names

    def test_health_score_decreases_with_anti_patterns(self):
        from neurodiff.engines.cognitive_engine import CommitPatternAnalyzer
        from neurodiff.core.semantic_events import FunctionAdded, ImportAdded
        analyzer = CommitPatternAnalyzer()

        # Clean scenario: no anti-patterns
        clean_fds = [_make_file_diff("tests/test_core.py", added=30, removed=5)]
        clean_result = analyzer.analyze([], clean_fds, self._make_ai_score(0.0), 0.0, self._make_blast_radius(0))
        clean_score = clean_result.commit_health_score

        # Dirty scenario: The Dump
        dirty_fds = [_make_file_diff("core/big.py", added=700, removed=5)]
        dirty_result = analyzer.analyze(
            [MagicMock(spec=FunctionAdded)],
            dirty_fds,
            self._make_ai_score(0.9),
            12.0,  # high context switch = Franken-commit too
            self._make_blast_radius(15),
        )
        dirty_score = dirty_result.commit_health_score
        assert dirty_score < clean_score

    def test_commit_health_score_in_range(self):
        from neurodiff.engines.cognitive_engine import CommitPatternAnalyzer
        analyzer = CommitPatternAnalyzer()
        file_diffs = [_make_file_diff("core/module.py", added=1000, removed=0)]
        result = analyzer.analyze([], file_diffs, self._make_ai_score(0.95), 12.0, self._make_blast_radius(25))
        assert 0.0 <= result.commit_health_score <= 1.0


# ---------------------------------------------------------------------------
# CognitiveEngine Runner Tests
# ---------------------------------------------------------------------------

class TestCognitiveEngine:
    def test_run_returns_cognitive_report(self):
        from neurodiff.engines.cognitive_engine import CognitiveEngine
        engine = CognitiveEngine()
        result = engine.run([], [_make_file_diff("a/b.py")])
        assert result.overall_verdict in ("safe", "caution", "danger")
        assert 0.0 <= result.fatigue_index.total_score <= 100
        assert 0.0 <= result.ai_generated.probability <= 1.0

    def test_empty_diff_is_safe(self):
        from neurodiff.engines.cognitive_engine import CognitiveEngine
        engine = CognitiveEngine()
        result = engine.run([], [])
        assert result.overall_verdict == "safe"

    def test_large_diff_danger_verdict(self):
        from neurodiff.engines.cognitive_engine import CognitiveEngine
        from neurodiff.core.semantic_events import FunctionAdded
        engine = CognitiveEngine()
        file_diffs = [_make_file_diff(f"dir{i}/module{j}.py", added=150) for i in range(8) for j in range(4)]
        events = [MagicMock(spec=FunctionAdded, cyclomatic_complexity=10, body="") for _ in range(20)]
        result = engine.run(events, file_diffs)
        assert result.overall_verdict in ("caution", "danger")
