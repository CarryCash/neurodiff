"""Phase 3 — Architectural Impact Engine.

Builds a module dependency graph from static import analysis and detects:
  - Layer violations (strict hierarchy)
  - Circular dependencies (filtered to diff-affected subgraph)
  - SOLID principle heuristics
  - Blast radius (transitive impact of modified files)

Uses stdlib `ast` for import analysis (not tree-sitter) and `networkx` for graph ops.
"""
from __future__ import annotations

import ast
import fnmatch
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ArchFinding:
    """A layer-level or structural architecture finding."""
    severity: Literal["critical", "high", "medium", "low"]
    category: Literal["layer_violation", "circular_dep", "solid_violation", "coupling"]
    file: str
    description: str
    involved_modules: list[str]


@dataclass
class SolidFinding:
    """A SOLID-principle heuristic finding."""
    principle: Literal["SRP", "OCP", "LSP", "ISP", "DIP"]
    severity: Literal["high", "medium", "low"]
    file: str
    entity: str
    description: str


@dataclass
class ArchReport:
    """Complete architectural analysis result."""
    layer_violations: list[ArchFinding] = field(default_factory=list)
    circular_deps: list[ArchFinding] = field(default_factory=list)
    solid_findings: list[SolidFinding] = field(default_factory=list)
    blast_radius: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default layer configuration
# ---------------------------------------------------------------------------

DEFAULT_LAYERS: list[dict] = [
    {"name": "cli",     "patterns": ["cli/*", "main.py"]},
    {"name": "engines", "patterns": ["engines/*"]},
    {"name": "core",    "patterns": ["core/*"]},
    {"name": "output",  "patterns": ["output/*"]},
    {"name": "models",  "patterns": ["models/*", "*/semantic_events.py"]},
]

DEFAULT_FORBIDDEN: list[dict] = [
    {
        "from": "core",
        "to": "engines",
        "reason": "Core must not depend on engines (DIP violation)",
    },
    {
        "from": "core",
        "to": "cli",
        "reason": "Core logic must not know about presentation layer",
    },
    {
        "from": "models",
        "to": "core",
        "reason": "Models/events must remain framework-agnostic",
    },
]

_EXCLUDED_DIRS = {".git", "__pycache__", "venv", "env", ".venv", "node_modules", "dist", "build"}


# ---------------------------------------------------------------------------
# Architecture Engine
# ---------------------------------------------------------------------------

class ArchEngine:
    """Architectural analysis engine.

    Usage::

        engine = ArchEngine(repo_path, arch_rules_path)
        report = engine.run(diff_files, semantic_events)
    """

    def __init__(
        self,
        repo_path: Path,
        arch_rules_path: Path | None = None,
    ) -> None:
        self.repo_path = repo_path.resolve()
        self._graph: Any = None  # networkx.DiGraph, lazy
        self._layers: list[dict] = []
        self._forbidden: list[dict] = []
        self._load_arch_rules(arch_rules_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def graph(self) -> Any:
        """Get the dependency graph."""
        return self._get_graph()

    def run(
        self,
        diff_files: list[str],
        semantic_events: list,
    ) -> ArchReport:
        """Run all architectural checks and return a consolidated report."""
        graph = self._get_graph()

        # Normalise diff_files to module-path keys used in the graph
        diff_modules = self._files_to_modules(diff_files)

        report = ArchReport()
        report.layer_violations = self._detect_layer_violations(graph, diff_modules)
        report.circular_deps = self._detect_circular_deps(graph, diff_modules)
        report.solid_findings = self._detect_solid_violations(semantic_events)
        report.blast_radius = self._compute_blast_radius(diff_modules, graph)
        return report

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_arch_rules(self, path: Path | None) -> None:
        """Load arch-rules.yaml; fall back silently to defaults."""
        if path is None:
            # Auto-discover in repo root
            candidate = self.repo_path / "arch-rules.yaml"
            if candidate.exists():
                path = candidate

        if path is not None and path.exists():
            try:
                import yaml  # pyyaml
            except ImportError:
                logger.warning("pyyaml not installed; using default arch rules.")
                self._layers = DEFAULT_LAYERS
                self._forbidden = DEFAULT_FORBIDDEN
                return

            try:
                with open(path, encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                self._layers = data.get("layers", DEFAULT_LAYERS)
                self._forbidden = data.get("forbidden_imports", DEFAULT_FORBIDDEN)
                logger.debug("Loaded arch rules from %s", path)
                return
            except Exception as exc:
                raise ValueError(
                    f"arch-rules.yaml syntax error in {path}: {exc}"
                ) from exc

        self._layers = DEFAULT_LAYERS
        self._forbidden = DEFAULT_FORBIDDEN

    # ------------------------------------------------------------------
    # Graph construction (lazy, cached)
    # ------------------------------------------------------------------

    def _get_graph(self) -> Any:
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _build_graph(self) -> Any:
        """Walk all .py files and build a directed import graph."""
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "networkx is required for architecture analysis. "
                "Install with: pip install networkx"
            ) from exc

        g: Any = nx.DiGraph()

        for py_file in self.repo_path.rglob("*.py"):
            # Skip excluded directories
            if any(part in _EXCLUDED_DIRS for part in py_file.parts):
                continue

            mod = self._file_to_module(py_file)
            g.add_node(mod)

            imports = self._extract_imports_from_file(py_file)
            for imp in imports:
                g.add_edge(mod, imp)

        return g

    def _extract_imports_from_file(self, path: Path) -> list[str]:
        """Extract all imported module names from a Python file using stdlib ast."""
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            return []
        except Exception:
            return []

        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")

        # Resolve relative imports to absolute module paths where possible
        resolved: list[str] = []
        pkg = self._file_to_module(path).rsplit(".", 1)[0] if "." in self._file_to_module(path) else ""
        for imp in imports:
            # Keep only imports that look internal (no dots starting = external pkg)
            if imp.startswith("."):
                if pkg:
                    resolved.append(f"{pkg}.{imp.lstrip('.')}")
            else:
                resolved.append(imp)

        return resolved

    def _file_to_module(self, path: Path) -> str:
        """Convert an absolute file path to a dotted module string relative to repo root."""
        try:
            rel = path.relative_to(self.repo_path)
        except ValueError:
            rel = path
        parts = list(rel.parts)
        if parts and parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    def _files_to_modules(self, files: list[str]) -> list[str]:
        mods = []
        for f in files:
            p = Path(f)
            if not p.is_absolute():
                p = self.repo_path / p
            mods.append(self._file_to_module(p))
        return mods

    # ------------------------------------------------------------------
    # Layer detection helpers
    # ------------------------------------------------------------------

    def _get_layer(self, module: str) -> tuple[int, str] | None:
        """Return (layer_index, layer_name) for a module, or None if unclassified."""
        # Convert dotted module to a path-like string for fnmatch
        path_like = module.replace(".", "/")
        for idx, layer in enumerate(self._layers):
            for pattern in layer.get("patterns", []):
                if fnmatch.fnmatch(path_like, pattern):
                    return idx, layer["name"]
        return None

    # ------------------------------------------------------------------
    # Layer violation detection
    # ------------------------------------------------------------------

    def _detect_layer_violations(
        self, graph: Any, diff_modules: list[str]
    ) -> list[ArchFinding]:
        findings: list[ArchFinding] = []
        diff_set = set(diff_modules)

        for src, dst in graph.edges():
            # Only flag edges where at least one node is in the diff
            if src not in diff_set and dst not in diff_set:
                continue

            src_layer = self._get_layer(src)
            dst_layer = self._get_layer(dst)

            if src_layer is None or dst_layer is None:
                continue

            src_idx, src_name = src_layer
            dst_idx, dst_name = dst_layer

            # Check if lower-level layer (higher index) imports higher-level layer (lower index)
            # Example: core (idx 2) importing engines (idx 1) is a violation.
            if dst_idx < src_idx:
                findings.append(ArchFinding(
                    severity="critical",
                    category="layer_violation",
                    file=src.replace(".", "/") + ".py",
                    description=(
                        f"Layer violation: `{src}` ({src_name}) imports `{dst}` ({dst_name}). "
                        f"Layer '{src_name}' (index {src_idx}) must not depend on "
                        f"'{dst_name}' (index {dst_idx}) — lower layers should be more abstract."
                    ),
                    involved_modules=[src, dst],
                ))
                continue

            # Check forbidden import rules
            for rule in self._forbidden:
                if rule.get("from") == src_name and rule.get("to") == dst_name:
                    findings.append(ArchFinding(
                        severity="high",
                        category="layer_violation",
                        file=src.replace(".", "/") + ".py",
                        description=(
                            f"Forbidden import: `{src}` → `{dst}`. "
                            f"Reason: {rule.get('reason', 'Architectural rule violation')}"
                        ),
                        involved_modules=[src, dst],
                    ))

        return findings

    # ------------------------------------------------------------------
    # Circular dependency detection
    # ------------------------------------------------------------------

    def _detect_circular_deps(
        self, graph: Any, diff_modules: list[str]
    ) -> list[ArchFinding]:
        try:
            import networkx as nx
        except ImportError:
            return []

        findings: list[ArchFinding] = []
        diff_set = set(diff_modules)

        if not diff_set:
            return []

        # Build 2-hop subgraph around diff nodes for performance
        subgraph_nodes: set[str] = set(diff_set)
        for mod in diff_set:
            if mod in graph:
                subgraph_nodes.update(graph.successors(mod))
                subgraph_nodes.update(graph.predecessors(mod))
                for neighbor in list(graph.successors(mod)):
                    subgraph_nodes.update(graph.successors(neighbor))
                for neighbor in list(graph.predecessors(mod)):
                    subgraph_nodes.update(graph.predecessors(neighbor))

        subgraph = graph.subgraph(subgraph_nodes)

        try:
            cycles = list(nx.simple_cycles(subgraph))
        except Exception:
            return []

        seen: set[frozenset] = set()
        for cycle in cycles:
            # Only include cycles that touch the diff
            if not any(m in diff_set for m in cycle):
                continue

            key = frozenset(cycle)
            if key in seen:
                continue
            seen.add(key)

            cycle_str = " → ".join(cycle) + f" → {cycle[0]}"
            findings.append(ArchFinding(
                severity="critical",
                category="circular_dep",
                file=cycle[0].replace(".", "/") + ".py",
                description=f"Circular dependency detected: {cycle_str}",
                involved_modules=list(cycle),
            ))

        return findings

    # ------------------------------------------------------------------
    # SOLID heuristics
    # ------------------------------------------------------------------

    def _detect_solid_violations(self, semantic_events: list) -> list[SolidFinding]:
        """Analyze SemanticEvents for SOLID principle heuristics."""
        from neurodiff.core.semantic_events import (
            ClassAdded,
            ClassModified,
            FunctionAdded,
            FunctionModified,
            ImportAdded,
        )

        findings: list[SolidFinding] = []

        for event in semantic_events:
            # ---- SRP: functions doing too much ----
            if isinstance(event, FunctionAdded):
                if event.cyclomatic_complexity > 15:
                    findings.append(SolidFinding(
                        principle="SRP",
                        severity="high",
                        file=event.file,
                        entity=event.name,
                        description=(
                            f"(heuristic) Function `{event.name}` has cyclomatic complexity "
                            f"{event.cyclomatic_complexity} > 15. May be taking on too many "
                            "responsibilities. Consider decomposing."
                        ),
                    ))
                if event.body_lines > 50:
                    findings.append(SolidFinding(
                        principle="SRP",
                        severity="medium",
                        file=event.file,
                        entity=event.name,
                        description=(
                            f"(heuristic) Function `{event.name}` is {event.body_lines} lines long. "
                            "Long functions often handle multiple concerns."
                        ),
                    ))

            # ---- SRP: class receiving too many new methods ----
            if isinstance(event, ClassModified) and len(event.methods_added) > 5:
                findings.append(SolidFinding(
                    principle="SRP",
                    severity="high",
                    file=event.file,
                    entity=event.name,
                    description=(
                        f"(heuristic) Class `{event.name}` gained {len(event.methods_added)} new "
                        "methods in a single diff. May be taking on too many responsibilities."
                    ),
                ))

            # ---- OCP: modifying base/abstract class ----
            if isinstance(event, ClassModified):
                is_abstract = any(
                    kw in event.name
                    for kw in ("Base", "Abstract", "Interface", "Mixin", "Protocol")
                )
                if is_abstract and event.methods_added:
                    findings.append(SolidFinding(
                        principle="OCP",
                        severity="high",
                        file=event.file,
                        entity=event.name,
                        description=(
                            f"(heuristic) Abstract/base class `{event.name}` was modified "
                            f"(added: {', '.join(event.methods_added[:3])}). "
                            "Prefer extension over modification of base classes."
                        ),
                    ))

            # ---- ISP: large class added ----
            if isinstance(event, ClassAdded) and len(event.methods) > 10:
                findings.append(SolidFinding(
                    principle="ISP",
                    severity="high" if len(event.methods) > 20 else "medium",
                    file=event.file,
                    entity=event.name,
                    description=(
                        f"(heuristic) Class `{event.name}` was added with {len(event.methods)} "
                        "methods. Large interfaces force clients to depend on methods they "
                        "don't use. Consider splitting."
                    ),
                ))

            # ---- DIP: core importing engines ----
            if isinstance(event, ImportAdded):
                file_layer = self._get_layer(
                    event.file.replace("/", ".").replace("\\", ".").removesuffix(".py")
                )
                mod_layer = self._get_layer(event.module.replace("/", "."))

                if (
                    file_layer is not None
                    and mod_layer is not None
                    and file_layer[0] > mod_layer[0]
                ):
                    findings.append(SolidFinding(
                        principle="DIP",
                        severity="high",
                        file=event.file,
                        entity=event.module,
                        description=(
                            f"(heuristic) Module in layer '{file_layer[1]}' imports from "
                            f"layer '{mod_layer[1]}' (`{event.module}`). "
                            "Higher layers should depend on abstractions, not concrete "
                            "lower-level implementations."
                        ),
                    ))

        return findings

    # ------------------------------------------------------------------
    # Blast radius
    # ------------------------------------------------------------------

    def _compute_blast_radius(
        self, diff_modules: list[str], graph: Any
    ) -> dict[str, Any]:
        """Compute transitive impact of modified files."""
        try:
            import networkx as nx
        except ImportError:
            return {"total_affected": 0, "affected_modules": [], "score": "low", "error": "networkx not installed"}

        affected: set[str] = set()
        per_module_counts: dict[str, int] = {}

        for mod in diff_modules:
            if mod not in graph:
                continue
            try:
                # Ancestors in the dependency graph = modules that import `mod` (directly or indirectly)
                ancestors = nx.ancestors(graph, mod)
                per_module_counts[mod] = len(ancestors)
                affected.update(ancestors)
            except Exception:
                per_module_counts[mod] = 0

        total = len(affected)

        if total >= 15:
            score = "critical"
        elif total >= 8:
            score = "high"
        elif total >= 3:
            score = "medium"
        else:
            score = "low"

        # Most impacted = modules most widely imported by others
        hotspots = sorted(per_module_counts.items(), key=lambda x: -x[1])

        return {
            "total_affected": total,
            "affected_modules": sorted(affected),
            "score": score,
            "hotspots": [
                {"module": m, "reverse_deps": c} for m, c in hotspots[:5]
            ],
        }
