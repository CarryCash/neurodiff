"""Tests for Phase 3 Architecture Engine."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from neurodiff.engines.arch_engine import (
    ArchEngine,
    ArchFinding,
    ArchReport,
    SolidFinding,
    DEFAULT_LAYERS,
)
from neurodiff.core.semantic_events import (
    ClassAdded,
    ClassModified,
    FunctionAdded,
    ImportAdded,
    ImportRemoved,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(files: dict[str, str], tmp_path: Path) -> Path:
    """Create a fake repo directory with given file contents."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for rel, content in files.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return repo


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

class TestGraphConstruction:
    def test_builds_nodes_for_py_files(self, tmp_path: Path) -> None:
        repo = _make_repo({"core/parser.py": "import os\n"}, tmp_path)
        engine = ArchEngine(repo)
        g = engine._get_graph()
        assert "core.parser" in g.nodes

    def test_builds_edge_for_import(self, tmp_path: Path) -> None:
        repo = _make_repo(
            {
                "core/parser.py": "import os\nfrom core import events\n",
                "core/events.py": "",
            },
            tmp_path,
        )
        engine = ArchEngine(repo)
        g = engine._get_graph()
        # Should have an edge from core.parser → core
        assert g.has_edge("core.parser", "core")

    def test_skips_excluded_dirs(self, tmp_path: Path) -> None:
        repo = _make_repo(
            {"__pycache__/cache.py": "x = 1\n", "core/real.py": ""},
            tmp_path,
        )
        engine = ArchEngine(repo)
        g = engine._get_graph()
        assert "__pycache__.cache" not in g.nodes
        assert "core.real" in g.nodes

    def test_invalid_python_skipped_gracefully(self, tmp_path: Path) -> None:
        repo = _make_repo({"core/bad.py": "def f(\n"}, tmp_path)
        engine = ArchEngine(repo)
        # Should not raise
        g = engine._get_graph()
        assert isinstance(g.nodes, object)


# ---------------------------------------------------------------------------
# Circular Dependency Detection
# ---------------------------------------------------------------------------

class TestCircularDeps:
    def test_simple_cycle_detected(self, tmp_path: Path) -> None:
        repo = _make_repo(
            {
                "core/a.py": "from core import b\n",
                "core/b.py": "from core import a\n",
            },
            tmp_path,
        )
        engine = ArchEngine(repo)
        g = engine._get_graph()
        findings = engine._detect_circular_deps(g, diff_modules=["core.a"])
        assert len(findings) >= 1
        assert findings[0].category == "circular_dep"
        assert findings[0].severity == "critical"
        assert "core.a" in findings[0].involved_modules or "core.b" in findings[0].involved_modules

    def test_no_cycle_no_finding(self, tmp_path: Path) -> None:
        repo = _make_repo(
            {
                "core/a.py": "from core import b\n",
                "core/b.py": "",
            },
            tmp_path,
        )
        engine = ArchEngine(repo)
        g = engine._get_graph()
        findings = engine._detect_circular_deps(g, diff_modules=["core.a"])
        assert findings == []

    def test_cycle_not_touching_diff_is_ignored(self, tmp_path: Path) -> None:
        """Cycles that don't involve diff files should NOT be reported."""
        repo = _make_repo(
            {
                "core/a.py": "from core import b\n",
                "core/b.py": "from core import a\n",
                "core/unrelated.py": "",
            },
            tmp_path,
        )
        engine = ArchEngine(repo)
        g = engine._get_graph()
        # Only unrelated in diff
        findings = engine._detect_circular_deps(g, diff_modules=["core.unrelated"])
        assert findings == []


# ---------------------------------------------------------------------------
# Layer Violation Detection
# ---------------------------------------------------------------------------

class TestLayerViolations:
    def test_core_importing_engines_is_violation(self, tmp_path: Path) -> None:
        repo = _make_repo(
            {
                "core/parser.py": "from engines import security\n",
                "engines/security.py": "",
            },
            tmp_path,
        )
        engine = ArchEngine(repo)
        g = engine._get_graph()
        # Manually add the edge using the module keys the engine will assign
        g.add_edge("core.parser", "engines.security")
        findings = engine._detect_layer_violations(g, diff_modules=["core.parser"])
        assert any(f.category == "layer_violation" for f in findings)

    def test_cli_importing_core_is_allowed(self, tmp_path: Path) -> None:
        """cli (idx 0) → core (idx 2) is valid: higher layers CAN import lower."""
        repo = _make_repo({"cli/main.py": ""}, tmp_path)
        engine = ArchEngine(repo)
        g = engine._get_graph()
        g.add_edge("cli.main", "core.parser")
        findings = engine._detect_layer_violations(g, diff_modules=["cli.main"])
        assert all(f.involved_modules != ["cli.main", "core.parser"] for f in findings)

    def test_no_violations_when_no_diff(self, tmp_path: Path) -> None:
        repo = _make_repo({"core/a.py": "from engines import b\n", "engines/b.py": ""}, tmp_path)
        engine = ArchEngine(repo)
        g = engine._get_graph()
        g.add_edge("core.a", "engines.b")
        findings = engine._detect_layer_violations(g, diff_modules=[])
        assert findings == []


# ---------------------------------------------------------------------------
# SOLID Heuristics
# ---------------------------------------------------------------------------

class TestSolidViolations:
    def _engine(self, tmp_path: Path) -> ArchEngine:
        repo = _make_repo({"cli/main.py": ""}, tmp_path)
        return ArchEngine(repo)

    def test_isп_large_class_added(self, tmp_path: Path) -> None:
        engine = self._engine(tmp_path)
        event = ClassAdded(
            name="MegaService",
            file="core/service.py",
            methods=[f"method_{i}" for i in range(15)],
            inherits_from=[],
        )
        findings = engine._detect_solid_violations([event])
        isp = [f for f in findings if f.principle == "ISP"]
        assert len(isp) >= 1
        assert isp[0].entity == "MegaService"

    def test_srp_high_complexity_function(self, tmp_path: Path) -> None:
        engine = self._engine(tmp_path)
        event = FunctionAdded(
            name="do_everything",
            file="core/handler.py",
            start_line=1,
            body_lines=80,
            calls=[],
            cyclomatic_complexity=20,
        )
        findings = engine._detect_solid_violations([event])
        srp = [f for f in findings if f.principle == "SRP"]
        assert len(srp) >= 1

    def test_ocp_base_class_modified(self, tmp_path: Path) -> None:
        engine = self._engine(tmp_path)
        event = ClassModified(
            name="BaseRepository",
            file="core/repo.py",
            methods_added=["new_query", "bulk_save"],
            methods_removed=[],
        )
        findings = engine._detect_solid_violations([event])
        ocp = [f for f in findings if f.principle == "OCP"]
        assert len(ocp) >= 1

    def test_srp_class_too_many_new_methods(self, tmp_path: Path) -> None:
        engine = self._engine(tmp_path)
        event = ClassModified(
            name="UserService",
            file="core/user.py",
            methods_added=[f"m{i}" for i in range(7)],
            methods_removed=[],
        )
        findings = engine._detect_solid_violations([event])
        srp = [f for f in findings if f.principle == "SRP"]
        assert len(srp) >= 1

    def test_clean_events_no_solid_findings(self, tmp_path: Path) -> None:
        engine = self._engine(tmp_path)
        event = FunctionAdded(
            name="parse",
            file="core/parser.py",
            start_line=1,
            body_lines=10,
            calls=["re.match"],
            cyclomatic_complexity=2,
        )
        findings = engine._detect_solid_violations([event])
        assert findings == []


# ---------------------------------------------------------------------------
# Blast Radius
# ---------------------------------------------------------------------------

class TestBlastRadius:
    def test_3hop_chain(self, tmp_path: Path) -> None:
        """A → B → C → D: modifying A should affect B, C, D."""
        import networkx as nx

        repo = _make_repo({"core/a.py": ""}, tmp_path)
        engine = ArchEngine(repo)
        engine._graph = nx.DiGraph()
        g = engine._graph
        g.add_edges_from([("b", "a"), ("c", "b"), ("d", "c")])

        result = engine._compute_blast_radius(["a"], g)
        assert result["total_affected"] >= 3
        assert "b" in result["affected_modules"]
        assert "c" in result["affected_modules"]
        assert "d" in result["affected_modules"]

    def test_no_dependents(self, tmp_path: Path) -> None:
        import networkx as nx

        repo = _make_repo({"core/leaf.py": ""}, tmp_path)
        engine = ArchEngine(repo)
        engine._graph = nx.DiGraph()
        engine._graph.add_node("core.leaf")

        result = engine._compute_blast_radius(["core.leaf"], engine._graph)
        assert result["total_affected"] == 0
        assert result["score"] == "low"

    def test_score_thresholds(self, tmp_path: Path) -> None:
        import networkx as nx

        repo = _make_repo({"core/hub.py": ""}, tmp_path)
        engine = ArchEngine(repo)
        engine._graph = nx.DiGraph()
        g = engine._graph
        hub = "core.hub"
        # 16 dependents → critical
        for i in range(16):
            g.add_edge(f"mod_{i}", hub)

        result = engine._compute_blast_radius([hub], g)
        assert result["score"] == "critical"


# ---------------------------------------------------------------------------
# ArchReport dataclass
# ---------------------------------------------------------------------------

def test_arch_report_defaults() -> None:
    report = ArchReport()
    assert report.layer_violations == []
    assert report.circular_deps == []
    assert report.solid_findings == []
    assert isinstance(report.blast_radius, dict)


# ---------------------------------------------------------------------------
# arch-rules.yaml loading
# ---------------------------------------------------------------------------

class TestArchRulesLoading:
    def test_custom_rules_loaded(self, tmp_path: Path) -> None:
        rules = tmp_path / "arch-rules.yaml"
        rules.write_text(
            "layers:\n  - name: api\n    patterns: ['api/*']\n"
            "forbidden_imports: []\n",
            encoding="utf-8",
        )
        repo = _make_repo({"api/main.py": ""}, tmp_path)
        engine = ArchEngine(repo, arch_rules_path=rules)
        assert engine._layers[0]["name"] == "api"

    def test_missing_rules_uses_defaults(self, tmp_path: Path) -> None:
        repo = _make_repo({"core/a.py": ""}, tmp_path)
        engine = ArchEngine(repo, arch_rules_path=None)
        assert engine._layers == DEFAULT_LAYERS

    def test_bad_yaml_raises_valueerror(self, tmp_path: Path) -> None:
        rules = tmp_path / "arch-rules.yaml"
        rules.write_text(": bad: {\n", encoding="utf-8")
        repo = _make_repo({"core/a.py": ""}, tmp_path)
        with pytest.raises(ValueError, match="arch-rules.yaml syntax error"):
            ArchEngine(repo, arch_rules_path=rules)
