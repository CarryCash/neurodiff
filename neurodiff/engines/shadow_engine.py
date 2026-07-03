"""Shadow Developer Engine — Automatic documentation generation and updating.

Two modes:
  - GENERATE: Creates README.md, ARCHITECTURE.md, docs/api.md, and inserts docstrings.
  - UPDATE: Detects signature changes from SemanticEvents and patches existing docs.
"""
from __future__ import annotations

import ast
import asyncio
import difflib
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import typer
from rich.console import Console

from neurodiff.engines.llm_engine import LLMProvider, parse_json_response
from neurodiff.core.semantic_events import (
    SemanticEvent,
    FunctionAdded,
    FunctionModified,
    # pyrefly: ignore [missing-module-attribute]
    ClassModified,
)

logger = logging.getLogger(__name__)
console = Console()

# Directories / file names to skip when walking the repo
_SKIP_DIRS = {
    ".git", ".tox", ".mypy_cache", "__pycache__", "venv", ".venv",
    "env", "node_modules", ".eggs", "dist", "build", "site-packages",
}

# -----------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------

@dataclass
class DocInventory:
    """Result of scanning a repository for documentation state."""
    has_readme: bool
    has_architecture_md: bool
    markdown_files: list[Path]
    functions_with_docstrings: int
    functions_without_docstrings: int
    undocumented_functions: list[dict]
    mode: Literal["update", "generate"]


@dataclass
class DocMention:
    """A reference to a code entity found inside a Markdown file."""
    file: Path
    line: int
    context: str
    entity_name: str
    needs_update: bool


@dataclass
class ShadowReport:
    """Report summarising Shadow Developer actions."""
    mode: Literal["update", "generate"]
    files_generated: list[str] = field(default_factory=list)
    files_updated: list[str] = field(default_factory=list)
    docstrings_added: int = 0
    docstrings_updated: int = 0
    functions_still_undocumented: int = 0
    coverage_before: float = 0.0
    coverage_after: float = 0.0
    error: str | None = None


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _should_skip(path: Path) -> bool:
    """Return True if any component of *path* is in the skip list."""
    return bool(set(path.parts) & _SKIP_DIRS)


def _extract_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build a human-readable signature string from an AST node."""
    args_parts: list[str] = []
    all_args = node.args

    # positional / normal args
    for arg in all_args.args:
        ann = ""
        if arg.annotation:
            ann = f": {ast.unparse(arg.annotation)}"
        args_parts.append(f"{arg.arg}{ann}")

    # *args
    if all_args.vararg:
        args_parts.append(f"*{all_args.vararg.arg}")

    # keyword-only
    for arg in all_args.kwonlyargs:
        ann = ""
        if arg.annotation:
            ann = f": {ast.unparse(arg.annotation)}"
        args_parts.append(f"{arg.arg}{ann}")

    # **kwargs
    if all_args.kwarg:
        args_parts.append(f"**{all_args.kwarg.arg}")

    ret = ""
    if node.returns:
        ret = f" -> {ast.unparse(node.returns)}"

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(args_parts)}){ret}"


def _func_body_summary(source_lines: list[str], node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Return a short textual summary of the function body (first ~15 lines)."""
    start = node.lineno  # 1-indexed, body starts after def line
    end = node.end_lineno or start + 15
    body_lines = source_lines[start: min(end, start + 15)]
    return "\n".join(body_lines)


def _extract_calls(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return names of all functions/methods called inside *node*."""
    calls: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                calls.append(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                calls.append(child.func.attr)
    return calls


# -----------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------

class ShadowEngine:
    """Shadow Developer — automatic documentation generator / updater."""

    def __init__(self, provider: LLMProvider | None = None):
        self.provider = provider

    # ------------------------------------------------------------------
    # 1. Repository Scanner
    # ------------------------------------------------------------------

    def scan_repo(self, repo_path: Path) -> DocInventory:
        """Walk the repository and build a DocInventory."""
        has_readme = (repo_path / "README.md").exists()
        has_architecture_md = (repo_path / "ARCHITECTURE.md").exists()

        md_files: list[Path] = []
        for md in repo_path.rglob("*.md"):
            if not _should_skip(md):
                md_files.append(md)

        funcs_with = 0
        funcs_without = 0
        undoc_funcs: list[dict] = []

        for py_file in repo_path.rglob("*.py"):
            if _should_skip(py_file):
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(content, filename=str(py_file))
                source_lines = content.splitlines()

                for node in ast.walk(tree):
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    # Skip private helpers (keep __init__ and other dunders)
                    if node.name.startswith("_") and not node.name.startswith("__"):
                        continue

                    docstring = ast.get_docstring(node)
                    if docstring:
                        funcs_with += 1
                    else:
                        funcs_without += 1
                        sig = _extract_signature(node)
                        calls = _extract_calls(node)
                        body_summary = _func_body_summary(source_lines, node)
                        undoc_funcs.append({
                            "name": node.name,
                            "file": str(py_file.relative_to(repo_path)),
                            "abs_file": str(py_file),
                            "signature": sig,
                            "lineno": node.lineno,
                            "body_lineno": node.body[0].lineno if node.body else node.lineno + 1,
                            "calls": calls,
                            "body_summary": body_summary,
                            "complexity": self._estimate_complexity(node),
                        })
            except Exception as exc:
                logger.debug("Failed to parse %s: %s", py_file, exc)

        total_funcs = funcs_with + funcs_without
        mode: Literal["update", "generate"] = "update"
        if not has_readme or (total_funcs > 0 and (funcs_without / total_funcs) > 0.6):
            mode = "generate"

        return DocInventory(
            has_readme=has_readme,
            has_architecture_md=has_architecture_md,
            markdown_files=md_files,
            functions_with_docstrings=funcs_with,
            functions_without_docstrings=funcs_without,
            undocumented_functions=undoc_funcs,
            mode=mode,
        )

    @staticmethod
    def _estimate_complexity(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        """Rough cyclomatic-complexity estimator via AST node counting."""
        count = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                count += 1
            elif isinstance(child, ast.BoolOp):
                count += len(child.values) - 1
        return count

    # ------------------------------------------------------------------
    # 2. Mode 2 — GENERATE
    # ------------------------------------------------------------------

    async def run_generate(
        self,
        repo_path: Path,
        inventory: DocInventory,
        *,
        dry_run: bool = False,
        no_inline: bool = False,
        arch_report: Any = None,
    ) -> ShadowReport:
        """Generate documentation from scratch."""
        report = ShadowReport(
            mode="generate",
            coverage_before=self._calc_coverage(inventory),
        )

        if not self.provider:
            report.error = "No LLM provider configured"
            return report

        # ---- README.md ----
        readme_content = await self._generate_readme(repo_path, inventory, arch_report)
        if readme_content:
            self._write_file(
                repo_path / "README.md", readme_content, dry_run=dry_run, report=report, generated=True,
            )

        # ---- ARCHITECTURE.md ----
        arch_content = await self._generate_architecture_md(repo_path, inventory, arch_report)
        if arch_content:
            self._write_file(
                repo_path / "ARCHITECTURE.md", arch_content, dry_run=dry_run, report=report, generated=True,
            )

        # ---- docs/api.md ----
        api_content = await self._generate_api_md(repo_path, inventory)
        if api_content:
            self._write_file(
                repo_path / "docs" / "api.md", api_content, dry_run=dry_run, report=report, generated=True,
            )

        # ---- Docstring insertion ----
        if not no_inline:
            added, failed = await self._generate_docstrings(repo_path, inventory, dry_run=dry_run)
            report.docstrings_added = added
            report.functions_still_undocumented = failed

        # Recalculate coverage
        if not dry_run:
            new_inv = self.scan_repo(repo_path)
            report.coverage_after = self._calc_coverage(new_inv)
        else:
            estimated_after_with = inventory.functions_with_docstrings + report.docstrings_added
            estimated_total = estimated_after_with + report.functions_still_undocumented
            report.coverage_after = (estimated_after_with / estimated_total * 100) if estimated_total else 0

        return report

    # ------------------------------------------------------------------
    # 2A. README Generator
    # ------------------------------------------------------------------

    async def _generate_readme(
        self, repo_path: Path, inventory: DocInventory, arch_report: Any,
    ) -> str | None:
        """Ask LLM to write a README.md from the project structure."""
        if not self.provider:
            return None

        # Gather pyproject / setup info
        pyproject_text = ""
        pyproject_path = repo_path / "pyproject.toml"
        if pyproject_path.exists():
            pyproject_text = pyproject_path.read_text(encoding="utf-8", errors="replace")[:3000]

        # Build project structure tree (top-level dirs + key files)
        tree_lines: list[str] = []
        for item in sorted(repo_path.iterdir()):
            if item.name.startswith(".") or item.name in _SKIP_DIRS:
                continue
            if item.is_dir():
                sub_count = sum(1 for _ in item.rglob("*.py"))
                tree_lines.append(f"  {item.name}/  ({sub_count} .py files)")
            else:
                tree_lines.append(f"  {item.name}")
        tree_str = "\n".join(tree_lines[:40])

        arch_info = ""
        if arch_report:
            try:
                arch_info = json.dumps(arch_report.__dict__ if hasattr(arch_report, "__dict__") else {}, default=str)[:4000]
            except Exception:
                arch_info = str(arch_report)[:4000]

        system = (
            "You are a documentation expert. Generate a comprehensive README.md for a Python project. "
            "Output raw Markdown only — no explanation, no fences."
        )
        user = (
            f"Project structure:\n{tree_str}\n\n"
            f"pyproject.toml:\n{pyproject_text}\n\n"
            f"Architectural context:\n{arch_info}\n\n"
            "Include these sections:\n"
            "# ProjectName + one-line description\n"
            "## Installation\n"
            "## Usage — CLI commands and examples\n"
            "## How it Works — module overview\n"
            "## Architecture — ASCII layer diagram\n"
            "## Contributing\n"
        )
        try:
            return await self.provider.complete(system, user)
        except Exception as exc:
            logger.error("README generation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # 2B. ARCHITECTURE.md Generator
    # ------------------------------------------------------------------

    async def _generate_architecture_md(
        self, repo_path: Path, inventory: DocInventory, arch_report: Any,
    ) -> str | None:
        if not self.provider:
            return None

        modules: list[str] = []
        for py_file in sorted(repo_path.rglob("*.py")):
            if _should_skip(py_file):
                continue
            modules.append(str(py_file.relative_to(repo_path)))

        arch_info = ""
        if arch_report:
            try:
                arch_info = json.dumps(arch_report.__dict__ if hasattr(arch_report, "__dict__") else {}, default=str)[:4000]
            except Exception:
                arch_info = str(arch_report)[:4000]

        system = (
            "You are a documentation expert. Generate an ARCHITECTURE.md explaining the codebase structure. "
            "Include: module responsibilities, data flow between modules, key design decisions. "
            "Output raw Markdown only."
        )
        user = (
            f"Python modules in project:\n" + "\n".join(modules[:60]) + "\n\n"
            f"Architectural context:\n{arch_info}\n"
        )
        try:
            return await self.provider.complete(system, user)
        except Exception as exc:
            logger.error("ARCHITECTURE.md generation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # 2C. Docstring Generator
    # ------------------------------------------------------------------

    async def _generate_docstrings(
        self, repo_path: Path, inventory: DocInventory, *, dry_run: bool = False,
    ) -> tuple[int, int]:
        """Generate and insert docstrings for undocumented functions.

        Returns:
            (added_count, failed_count)
        """
        if not self.provider:
            return 0, len(inventory.undocumented_functions)

        semaphore = asyncio.Semaphore(10)
        results: list[tuple[dict, str | None]] = []

        async def _gen_one(func_info: dict) -> tuple[dict, str | None]:
            async with semaphore:
                system = (
                    "You are a Python documentation expert. Generate a Google-style docstring for the given function. "
                    "Output ONLY the docstring text (without triple quotes). No extra explanation."
                )
                user = json.dumps({
                    "function_name": func_info["name"],
                    "file": func_info["file"],
                    "signature": func_info["signature"],
                    "body_summary": func_info.get("body_summary", "")[:500],
                    "calls": func_info.get("calls", []),
                    "complexity": func_info.get("complexity", 1),
                })
                try:
                    raw = await self.provider.complete(system, user)  # type: ignore[union-attr]
                    text = (raw or "").strip()
                    # Strip markdown fences if present
                    text = re.sub(r"^```(?:python|text)?\s*", "", text)
                    text = re.sub(r"\s*```$", "", text).strip()
                    return func_info, text
                except Exception as exc:
                    logger.warning("Docstring generation failed for %s: %s", func_info["name"], exc)
                    return func_info, None

        tasks = [_gen_one(f) for f in inventory.undocumented_functions]
        results = await asyncio.gather(*tasks)

        added = 0
        failed = 0
        # Group by file so we can batch edits
        by_file: dict[str, list[tuple[dict, str]]] = {}
        for func_info, docstring in results:
            if docstring:
                abs_path = func_info["abs_file"]
                by_file.setdefault(abs_path, []).append((func_info, docstring))
            else:
                failed += 1

        for abs_path, items in by_file.items():
            path = Path(abs_path)
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
                lines = source.splitlines(keepends=True)

                # Sort by line number descending so inserts don't shift earlier lines
                items.sort(key=lambda x: x[0]["body_lineno"], reverse=True)

                for func_info, docstring in items:
                    insert_line = func_info["body_lineno"] - 1  # 0-indexed
                    if insert_line < 0 or insert_line > len(lines):
                        failed += 1
                        continue

                    # Determine indentation of the function body
                    if insert_line < len(lines):
                        existing = lines[insert_line]
                        indent = len(existing) - len(existing.lstrip())
                    else:
                        indent = 8

                    indent_str = " " * indent
                    doc_lines = docstring.splitlines()
                    formatted = f'{indent_str}"""{doc_lines[0]}\n'
                    for dl in doc_lines[1:]:
                        formatted += f"{indent_str}{dl}\n"
                    formatted += f'{indent_str}"""\n'

                    lines.insert(insert_line, formatted)
                    added += 1

                if not dry_run:
                    # Create backup
                    self._backup(path)
                    path.write_text("".join(lines), encoding="utf-8")

            except Exception as exc:
                logger.error("Failed to insert docstrings in %s: %s", abs_path, exc)
                failed += len(items)

        return added, failed

    # ------------------------------------------------------------------
    # 2D. API docs
    # ------------------------------------------------------------------

    async def _generate_api_md(self, repo_path: Path, inventory: DocInventory) -> str | None:
        """Build docs/api.md listing all public functions with signatures."""
        lines = ["# API Reference\n\n"]
        # Group by file
        by_file: dict[str, list[dict]] = {}
        for func in inventory.undocumented_functions:
            by_file.setdefault(func["file"], []).append(func)

        # Also walk documented functions from the AST
        for py_file in sorted(repo_path.rglob("*.py")):
            if _should_skip(py_file):
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(content, filename=str(py_file))
                rel = str(py_file.relative_to(repo_path))
                file_funcs: list[str] = []

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name.startswith("_") and not node.name.startswith("__"):
                            continue
                        sig = _extract_signature(node)
                        ds = ast.get_docstring(node)
                        entry = f"### `{sig}`\n"
                        if ds:
                            entry += f"\n{ds}\n"
                        else:
                            entry += "\n*No docstring.*\n"
                        file_funcs.append(entry)

                if file_funcs:
                    lines.append(f"## `{rel}`\n\n")
                    lines.extend(file_funcs)
                    lines.append("\n---\n\n")

            except Exception:
                continue

        return "\n".join(lines) if len(lines) > 2 else None

    # ------------------------------------------------------------------
    # 3. Mode 1 — UPDATE
    # ------------------------------------------------------------------

    async def run_update(
        self,
        repo_path: Path,
        inventory: DocInventory,
        events: list[SemanticEvent],
        *,
        dry_run: bool = False,
    ) -> ShadowReport:
        """Update existing documentation based on semantic changes."""
        report = ShadowReport(
            mode="update",
            coverage_before=self._calc_coverage(inventory),
        )

        if not self.provider:
            report.error = "No LLM provider configured"
            return report

        # 3A — Detect relevant changes
        changed_entities = self._detect_changed_entities(events)
        if not changed_entities:
            console.print("[dim]  No signature changes detected — nothing to update.[/dim]")
            report.coverage_after = report.coverage_before
            return report

        # 3B — Update doc mentions in markdown files
        mentions = self._find_doc_mentions(inventory.markdown_files, changed_entities, repo_path)
        for mention in mentions:
            if mention.needs_update:
                updated = await self._update_doc_mention(mention, changed_entities)
                if updated and not dry_run:
                    self._apply_mention_update(mention, updated)
                    report.files_updated.append(str(mention.file.relative_to(repo_path)))

        # 3C — Update docstrings for modified functions
        for entity in changed_entities:
            if entity["type"] == "FunctionModified" and entity.get("signature_changed"):
                count = await self._update_docstring(
                    repo_path, entity, dry_run=dry_run,
                )
                report.docstrings_updated += count

            # New functions get docstrings
            if entity["type"] == "FunctionAdded":
                for uf in inventory.undocumented_functions:
                    if uf["name"] == entity["name"] and uf["file"] == entity.get("file"):
                        ds = await self._generate_single_docstring(uf)
                        if ds and not dry_run:
                            self._insert_single_docstring(Path(uf["abs_file"]), uf, ds)
                            report.docstrings_added += 1

        # Recalculate
        if not dry_run:
            new_inv = self.scan_repo(repo_path)
            report.coverage_after = self._calc_coverage(new_inv)
        else:
            report.coverage_after = report.coverage_before

        return report

    # ------------------------------------------------------------------
    # 3A. Change Detector
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_changed_entities(events: list[SemanticEvent]) -> list[dict]:
        """Extract entities whose signatures or structure changed."""
        entities: list[dict] = []
        for ev in events:
            if isinstance(ev, FunctionModified) and ev.signature_changed:
                entities.append({
                    "type": "FunctionModified",
                    "name": ev.name,
                    "file": ev.file,
                    "signature_changed": True,
                })
            elif isinstance(ev, FunctionAdded):
                entities.append({
                    "type": "FunctionAdded",
                    "name": ev.name,
                    "file": ev.file,
                })
            elif isinstance(ev, ClassModified):
                entities.append({
                    "type": "ClassModified",
                    "name": ev.name,
                    "file": ev.file,
                    "methods_added": ev.methods_added,
                    "methods_removed": ev.methods_removed,
                })
        return entities

    # ------------------------------------------------------------------
    # 3B. Documentation Mention Finder
    # ------------------------------------------------------------------

    @staticmethod
    def _find_doc_mentions(
        md_files: list[Path],
        entities: list[dict],
        repo_path: Path,
    ) -> list[DocMention]:
        """Search Markdown files for references to changed entities."""
        mentions: list[DocMention] = []
        entity_names = {e["name"] for e in entities}

        for md_file in md_files:
            try:
                lines = md_file.read_text(encoding="utf-8", errors="replace").splitlines()
                for idx, line in enumerate(lines, 1):
                    for name in entity_names:
                        if name in line:
                            context_start = max(0, idx - 3)
                            context_end = min(len(lines), idx + 2)
                            context = "\n".join(lines[context_start:context_end])
                            mentions.append(DocMention(
                                file=md_file,
                                line=idx,
                                context=context,
                                entity_name=name,
                                needs_update=True,
                            ))
            except Exception:
                continue
        return mentions

    async def _update_doc_mention(
        self, mention: DocMention, entities: list[dict],
    ) -> str | None:
        """Use LLM to rewrite a documentation snippet around a changed entity."""
        if not self.provider:
            return None

        entity_info = next(
            (e for e in entities if e["name"] == mention.entity_name), {}
        )

        system = (
            "You are a documentation updater. Rewrite ONLY the documentation snippet "
            "to reflect the code change described. Output the updated snippet only — "
            "no explanation."
        )
        user = (
            f"Entity changed: {json.dumps(entity_info, default=str)}\n\n"
            f"Current documentation snippet (around line {mention.line} of {mention.file.name}):\n"
            f"{mention.context}\n\n"
            "Update this snippet to reflect the change."
        )
        try:
            return await self.provider.complete(system, user)
        except Exception as exc:
            logger.warning("Doc mention update failed for %s: %s", mention.entity_name, exc)
            return None

    @staticmethod
    def _apply_mention_update(mention: DocMention, new_text: str) -> None:
        """Replace the context lines in the markdown file with *new_text*."""
        path = mention.file
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            start = max(0, mention.line - 3)
            end = min(len(lines), mention.line + 2)

            # Backup
            bak = path.with_suffix(path.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(path, bak)

            new_lines = [l + "\n" for l in new_text.splitlines()]
            lines[start:end] = new_lines
            path.write_text("".join(lines), encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to apply mention update to %s: %s", path, exc)

    # ------------------------------------------------------------------
    # 3C. Docstring Updater
    # ------------------------------------------------------------------

    async def _update_docstring(
        self, repo_path: Path, entity: dict, *, dry_run: bool = False,
    ) -> int:
        """Update existing docstring of a modified function to reflect new signature."""
        if not self.provider:
            return 0

        file_path = repo_path / entity["file"]
        if not file_path.exists():
            return 0

        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name != entity["name"]:
                        continue
                    existing_ds = ast.get_docstring(node)
                    if not existing_ds:
                        continue

                    sig = _extract_signature(node)

                    system = (
                        "You are a Python documentation expert. Update this docstring to reflect "
                        "the new function signature. Output ONLY the updated docstring text "
                        "(without triple quotes)."
                    )
                    user = (
                        f"Function: {sig}\n\n"
                        f"Current docstring:\n{existing_ds}\n\n"
                        "Update the docstring to match the current signature and parameters."
                    )
                    new_ds = await self.provider.complete(system, user)
                    new_ds = (new_ds or "").strip()
                    new_ds = re.sub(r"^```(?:python|text)?\s*", "", new_ds)
                    new_ds = re.sub(r"\s*```$", "", new_ds).strip()

                    if new_ds and not dry_run:
                        # Replace old docstring in source
                        source = source.replace(existing_ds, new_ds, 1)
                        self._backup(file_path)
                        file_path.write_text(source, encoding="utf-8")
                    return 1
        except Exception as exc:
            logger.error("Docstring update failed for %s: %s", entity["name"], exc)
        return 0

    async def _generate_single_docstring(self, func_info: dict) -> str | None:
        """Generate a single docstring for a function."""
        if not self.provider:
            return None
        system = (
            "You are a Python documentation expert. Generate a Google-style docstring. "
            "Output ONLY the docstring text (without triple quotes)."
        )
        user = json.dumps({
            "function_name": func_info["name"],
            "file": func_info["file"],
            "signature": func_info["signature"],
            "body_summary": func_info.get("body_summary", "")[:500],
            "calls": func_info.get("calls", []),
        })
        try:
            raw = await self.provider.complete(system, user)
            text = (raw or "").strip()
            text = re.sub(r"^```(?:python|text)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()
            return text if text else None
        except Exception as exc:
            logger.warning("Docstring generation failed for %s: %s", func_info["name"], exc)
            return None

    @staticmethod
    def _insert_single_docstring(path: Path, func_info: dict, docstring: str) -> None:
        """Insert a docstring into a single file at the correct line."""
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            lines = source.splitlines(keepends=True)
            insert_line = func_info["body_lineno"] - 1  # 0-indexed

            if insert_line < 0 or insert_line > len(lines):
                return

            # Determine indentation
            if insert_line < len(lines):
                existing = lines[insert_line]
                indent = len(existing) - len(existing.lstrip())
            else:
                indent = 8

            indent_str = " " * indent
            doc_lines = docstring.splitlines()
            formatted = f'{indent_str}"""{doc_lines[0]}\n'
            for dl in doc_lines[1:]:
                formatted += f"{indent_str}{dl}\n"
            formatted += f'{indent_str}"""\n'

            # Backup
            bak = path.with_suffix(path.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(path, bak)

            lines.insert(insert_line, formatted)
            path.write_text("".join(lines), encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to insert docstring in %s: %s", path, exc)

    # ------------------------------------------------------------------
    # File I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _backup(path: Path) -> None:
        """Create a .bak copy if one doesn't already exist."""
        bak = path.with_suffix(path.suffix + ".bak")
        if path.exists() and not bak.exists():
            shutil.copy2(path, bak)

    @staticmethod
    def _write_file(
        path: Path,
        content: str,
        *,
        dry_run: bool,
        report: ShadowReport,
        generated: bool = True,
    ) -> None:
        """Write *content* to *path*, with backup and dry-run support."""
        rel = str(path)
        if dry_run:
            line_count = content.count("\n") + 1
            console.print(f"  [dim]+ {path.name:<25} ({line_count} lines) [DRY RUN][/dim]")
            if generated:
                report.files_generated.append(rel)
            return

        # Backup existing
        if path.exists():
            bak = path.with_suffix(path.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(path, bak)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        if generated:
            report.files_generated.append(rel)
        else:
            report.files_updated.append(rel)

    @staticmethod
    def _calc_coverage(inventory: DocInventory) -> float:
        total = inventory.functions_with_docstrings + inventory.functions_without_docstrings
        if total == 0:
            return 100.0
        return round(inventory.functions_with_docstrings / total * 100, 1)

    # ------------------------------------------------------------------
    # Preview / Confirmation
    # ------------------------------------------------------------------

    def print_preview(
        self,
        repo_path: Path,
        inventory: DocInventory,
        *,
        no_inline: bool = False,
    ) -> None:
        """Print what the engine would do before executing."""
        console.print(f"\n[bold magenta]Shadow Developer — {inventory.mode.upper()} mode[/bold magenta]")
        console.print("━" * 40)

        if inventory.mode == "generate":
            console.print("\n[bold]Files to create:[/bold]")
            if not inventory.has_readme:
                console.print("  [green]+[/green] README.md")
            if not inventory.has_architecture_md:
                console.print("  [green]+[/green] ARCHITECTURE.md")
            console.print("  [green]+[/green] docs/api.md")

            if not no_inline and inventory.undocumented_functions:
                console.print("\n[bold]Docstrings to insert:[/bold]")
                by_file: dict[str, int] = {}
                for f in inventory.undocumented_functions:
                    by_file[f["file"]] = by_file.get(f["file"], 0) + 1
                for file, count in sorted(by_file.items()):
                    console.print(f"  [green]+[/green] {file:<40} → {count} functions")

        total = inventory.functions_with_docstrings + inventory.functions_without_docstrings
        if total:
            pct = round(inventory.functions_without_docstrings / total * 100, 1)
            console.print(
                f"\n[dim]Current coverage: {100 - pct:.1f}% "
                f"({inventory.functions_with_docstrings}/{total} documented)[/dim]"
            )
        console.print()
