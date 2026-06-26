"""Phase 7 — Global Project Context Engine.

Builds a rich context payload combining RAG + architectural map so the LLM
understands the full codebase before suggesting active corrections.
"""
from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ArchitecturalMap:
    project_name: str
    detected_framework: str | None
    detected_patterns: list[str]
    module_tree: dict
    dependency_edges: list[tuple[str, str]]
    entry_points: list[str]
    test_coverage_map: dict[str, list[str]]
    total_functions: int
    total_classes: int
    avg_complexity: float


@dataclass
class GlobalContext:
    arch_map: ArchitecturalMap
    relevant_existing_code: dict[str, list[dict]] = field(default_factory=dict)
    project_conventions: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Context Engine
# ---------------------------------------------------------------------------

class ContextEngine:
    """Builds the global context for the active correction engine."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path.resolve()

    def build_context(
        self,
        dependency_graph: Any | None = None,
        layer_names: list[str] | None = None,
        duplication_engine: Any | None = None,
        findings_to_query: list[Any] | None = None,
    ) -> GlobalContext:
        """
        Builds the complete global context.
        
        Args:
            dependency_graph: nx.DiGraph from ArchEngine
            layer_names: List of architectural layers detected
            duplication_engine: Instance of DuplicationEngine (for RAG)
            findings_to_query: List of findings (Security, Arch) to run RAG against
        """
        arch_map, conventions = self._build_arch_map_and_conventions(
            dependency_graph, layer_names or []
        )
        
        rag_context = {}
        if duplication_engine and duplication_engine.has_chromadb and findings_to_query:
            rag_context = self._retrieve_rag_context(duplication_engine, findings_to_query)

        return GlobalContext(
            arch_map=arch_map,
            relevant_existing_code=rag_context,
            project_conventions=conventions,
        )

    # -----------------------------------------------------------------------
    # Internal Builders
    # -----------------------------------------------------------------------

    def _build_arch_map_and_conventions(
        self, graph: Any | None, layers: list[str]
    ) -> tuple[ArchitecturalMap, dict]:
        
        total_funcs = 0
        total_classes = 0
        total_cc = 0
        
        module_tree: dict = {}
        entry_points: list[str] = []
        
        frameworks_found = set()
        
        naming_styles = {"snake_case": 0, "camelCase": 0, "PascalCase": 0}
        docstring_types = {"google": 0, "numpy": 0, "sphinx": 0, "plain": 0}
        
        test_files = []
        source_files = []

        # Walk the repo
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "venv", ".venv", "node_modules", "dist", "build"}]
            
            for file in files:
                if not file.endswith((".py", ".js", ".ts")):
                    continue
                    
                filepath = Path(root) / file
                rel_path = str(filepath.relative_to(self.repo_path)).replace("\\", "/")
                
                if "/test" in rel_path or rel_path.startswith("test_") or "_test" in rel_path:
                    test_files.append(rel_path)
                else:
                    source_files.append(rel_path)
                
                try:
                    content = filepath.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                
                # Basic framework heuristic
                if "fastapi" in content.lower():
                    frameworks_found.add("fastapi")
                if "django" in content.lower():
                    frameworks_found.add("django")
                if "flask" in content.lower():
                    frameworks_found.add("flask")
                    
                if "__main__" in content or "FastAPI()" in content or "Flask(__name__)" in content:
                    entry_points.append(rel_path)
                
                # Parse AST to count and detect conventions
                if file.endswith(".py"):
                    try:
                        tree = ast.parse(content)
                        # Build module tree shape
                        parts = rel_path.split("/")
                        curr = module_tree
                        for p in parts[:-1]:
                            curr = curr.setdefault(p, {})
                        curr[parts[-1]] = {"functions": 0, "classes": 0}
                        
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                total_funcs += 1
                                curr[parts[-1]]["functions"] += 1
                                
                                # Estimate CC crudely for avg
                                total_cc += 1 + sum(1 for n in ast.walk(node) if isinstance(n, (ast.If, ast.For, ast.While, ast.And, ast.Or)))
                                
                                # Check naming
                                name = node.name
                                if not name.startswith("__"):
                                    if "_" in name and name.islower():
                                        naming_styles["snake_case"] += 1
                                    elif name[0].islower() and any(c.isupper() for c in name):
                                        naming_styles["camelCase"] += 1
                                
                                # Check docstring
                                doc = ast.get_docstring(node)
                                if doc:
                                    if "Args:" in doc or "Returns:" in doc:
                                        docstring_types["google"] += 1
                                    elif ":param" in doc or ":return:" in doc:
                                        docstring_types["sphinx"] += 1
                                    elif "Parameters\n" in doc and "----------" in doc:
                                        docstring_types["numpy"] += 1
                                    else:
                                        docstring_types["plain"] += 1
                                        
                            elif isinstance(node, ast.ClassDef):
                                total_classes += 1
                                curr[parts[-1]]["classes"] += 1
                                if node.name[0].isupper() and "_" not in node.name:
                                    naming_styles["PascalCase"] += 1
                    except SyntaxError:
                        pass
                        
        framework = "plain"
        if "fastapi" in frameworks_found: framework = "fastapi"
        elif "django" in frameworks_found: framework = "django"
        elif "flask" in frameworks_found: framework = "flask"
        
        # Determine dominant conventions
        dom_naming = max(naming_styles, key=lambda k: naming_styles[k]) if any(naming_styles.values()) else "unknown"
        dom_doc = max(docstring_types, key=lambda k: docstring_types[k]) if any(docstring_types.values()) else "unknown"
        
        conventions = {
            "naming_style": dom_naming,
            "docstring_style": dom_doc,
        }
        
        # Test coverage map heuristically
        test_coverage_map = {}
        for src in source_files:
            src_name = Path(src).stem
            matched_tests = []
            for tst in test_files:
                if src_name in tst or src_name.replace("_", "") in tst:
                    matched_tests.append(tst)
            if matched_tests:
                test_coverage_map[src] = matched_tests

        # Edges from networkx
        edges = []
        if graph:
            edges = list(graph.edges())
            
        avg_cc = total_cc / total_funcs if total_funcs > 0 else 1.0
        
        detected_patterns = []
        if layers:
            detected_patterns.append("layered")
            
        arch_map = ArchitecturalMap(
            project_name=self.repo_path.name,
            detected_framework=framework,
            detected_patterns=detected_patterns,
            module_tree=module_tree,
            dependency_edges=edges,
            entry_points=entry_points,
            test_coverage_map=test_coverage_map,
            total_functions=total_funcs,
            total_classes=total_classes,
            avg_complexity=round(avg_cc, 2),
        )
        
        return arch_map, conventions

    def _retrieve_rag_context(self, duplication_engine: Any, findings: list[Any]) -> dict[str, list[dict]]:
        rag_context: dict[str, list[dict]] = {}
        if not duplication_engine.has_chromadb:
            return rag_context
            
        for finding in findings:
            # Create a query string from the finding
            finding_id = getattr(finding, "rule_id", getattr(finding, "id", None))
            if not finding_id:
                # generate a pseudo id
                finding_id = f"{getattr(finding, 'severity', 'high')}_{getattr(finding, 'file', 'unknown')}"
                
            desc = getattr(finding, "description", "")
            if not desc:
                continue
                
            try:
                # Query chromadb collection
                # Find up to 3 most relevant functions in the codebase
                results = duplication_engine.collection.query(
                    query_texts=[desc],
                    n_results=3,
                )
                if results and results["documents"] and results["documents"][0]:
                    docs = results["documents"][0]
                    metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                    
                    matched = []
                    for doc, meta in zip(docs, metas):
                        matched.append({
                            "code": doc,
                            "file": meta.get("file", "unknown")
                        })
                    rag_context[finding_id] = matched
            except Exception as e:
                logger.warning(f"RAG retrieval failed for {finding_id}: {e}")
                
        return rag_context
