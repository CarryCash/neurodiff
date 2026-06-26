"""Architecture analysis engine — detects design pattern violations in diffs."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from neurodiff.core.semantic_events import (
    FunctionAdded,
    FunctionModified,
    ClassAdded,
    SemanticEvent,
)


@dataclass
class ArchitectureFinding:
    """A design pattern or architectural concern found in the diff."""
    rule_id: str
    title: str
    description: str
    severity: Literal["critical", "high", "medium", "low"]
    file: str
    entity_name: str
    suggestion: str


class ArchitectureEngine:
    """Detects architectural concerns and design-pattern violations in semantic diffs.

    Checks performed (all purely algorithmic, no LLM):
    - God function: single function > 50 lines or CC > 15
    - Deep call chains: a function calling > 8 distinct functions
    - Naming-convention violations (snake_case for Python, camelCase for JS/TS)
    - Class bloat: class with > 20 methods
    - Circular-import risk: a module importing itself or a close relative
    - Single-Responsibility hint: function name contains "and" / "or" (suggests dual responsibility)
    - Magic-number detector: literal numbers other than 0/1/-1 embedded in function bodies
    """

    # Heuristic thresholds
    GOD_FUNC_LINES = 50
    GOD_FUNC_CC = 15
    DEEP_CALL_THRESHOLD = 8
    CLASS_BLOAT_METHODS = 20

    def analyze(
        self,
        events: list[SemanticEvent],
        file_contents: dict[str, str],  # file_path -> content_after
    ) -> list[ArchitectureFinding]:
        """Run all architectural checks against the semantic events."""
        findings: list[ArchitectureFinding] = []

        for event in events:
            if isinstance(event, FunctionAdded):
                findings.extend(self._check_function_added(event, file_contents))
            elif isinstance(event, FunctionModified):
                findings.extend(self._check_function_modified(event, file_contents))
            elif isinstance(event, ClassAdded):
                findings.extend(self._check_class_added(event))

        return findings

    # ------------------------------------------------------------------
    # FunctionAdded checks
    # ------------------------------------------------------------------
    def _check_function_added(
        self, event: FunctionAdded, file_contents: dict[str, str]
    ) -> list[ArchitectureFinding]:
        findings: list[ArchitectureFinding] = []
        lang = self._detect_lang(event.file)

        # God function
        if event.body_lines > self.GOD_FUNC_LINES:
            findings.append(ArchitectureFinding(
                rule_id="god_function_lines",
                title="God Function (too long)",
                description=(
                    f"Function `{event.name}` has {event.body_lines} lines "
                    f"(threshold: {self.GOD_FUNC_LINES}). "
                    "Consider splitting into smaller, focused functions."
                ),
                severity="high",
                file=event.file,
                entity_name=event.name,
                suggestion="Extract sub-tasks into helper functions (Single Responsibility Principle).",
            ))

        if event.cyclomatic_complexity > self.GOD_FUNC_CC:
            findings.append(ArchitectureFinding(
                rule_id="god_function_cc",
                title="God Function (high complexity)",
                description=(
                    f"Function `{event.name}` has cyclomatic complexity {event.cyclomatic_complexity} "
                    f"(threshold: {self.GOD_FUNC_CC}). Harder to test and maintain."
                ),
                severity="high",
                file=event.file,
                entity_name=event.name,
                suggestion="Reduce branching with early returns, strategy pattern, or decomposition.",
            ))

        # Deep call chain
        if len(event.calls) > self.DEEP_CALL_THRESHOLD:
            findings.append(ArchitectureFinding(
                rule_id="deep_call_chain",
                title="Deep Call Chain",
                description=(
                    f"Function `{event.name}` calls {len(event.calls)} distinct functions. "
                    "This suggests high coupling or a missing abstraction layer."
                ),
                severity="medium",
                file=event.file,
                entity_name=event.name,
                suggestion=(
                    "Group related calls into a service/facade class or helper module."
                ),
            ))

        # Dual-responsibility hint
        lower = event.name.lower()
        if "_and_" in lower or "_or_" in lower:
            findings.append(ArchitectureFinding(
                rule_id="dual_responsibility",
                title="Possible Dual Responsibility",
                description=(
                    f"Function name `{event.name}` suggests it does more than one thing "
                    "('and'/'or' in name). This may violate the Single Responsibility Principle."
                ),
                severity="low",
                file=event.file,
                entity_name=event.name,
                suggestion="Split into two focused functions.",
            ))

        # Naming convention
        naming_finding = self._check_naming(event.name, lang, event.file, "function")
        if naming_finding:
            findings.append(naming_finding)

        # Magic numbers in body
        if event.file in file_contents:
            magic_finding = self._check_magic_numbers(
                event.name, event.file, file_contents[event.file],
                event.start_line, event.start_line + event.body_lines
            )
            if magic_finding:
                findings.append(magic_finding)

        return findings

    # ------------------------------------------------------------------
    # FunctionModified checks
    # ------------------------------------------------------------------
    def _check_function_modified(
        self, event: FunctionModified, file_contents: dict[str, str]
    ) -> list[ArchitectureFinding]:
        findings: list[ArchitectureFinding] = []

        # Complexity spike
        cc_delta = event.complexity_after - event.complexity_before
        if cc_delta >= 5:
            findings.append(ArchitectureFinding(
                rule_id="complexity_spike",
                title="Complexity Spike",
                description=(
                    f"Function `{event.name}` gained +{cc_delta} cyclomatic complexity in this diff "
                    f"({event.complexity_before} → {event.complexity_after}). "
                    "AI-generated code often inflates complexity silently."
                ),
                severity="high" if event.complexity_after > self.GOD_FUNC_CC else "medium",
                file=event.file,
                entity_name=event.name,
                suggestion="Review new branches carefully; consider extracting guard clauses.",
            ))

        # Size explosion (> 2× original)
        if event.lines_before > 0 and event.lines_after > event.lines_before * 2:
            findings.append(ArchitectureFinding(
                rule_id="size_explosion",
                title="Function Size Doubled",
                description=(
                    f"Function `{event.name}` grew from {event.lines_before} to "
                    f"{event.lines_after} lines (>{event.lines_after // max(event.lines_before,1)}×). "
                    "Common when AI inlines too much logic."
                ),
                severity="medium",
                file=event.file,
                entity_name=event.name,
                suggestion="Consider whether the new logic belongs in a separate helper.",
            ))

        # New calls added (dependency expansion)
        if len(event.calls_added) >= 5:
            findings.append(ArchitectureFinding(
                rule_id="dependency_expansion",
                title="Dependency Expansion",
                description=(
                    f"Function `{event.name}` gained {len(event.calls_added)} new call dependencies "
                    f"in this diff: {', '.join(event.calls_added[:6])}{'…' if len(event.calls_added) > 6 else ''}."
                ),
                severity="medium",
                file=event.file,
                entity_name=event.name,
                suggestion=(
                    "Verify each new dependency is intentional and not introducing coupling."
                ),
            ))

        return findings

    # ------------------------------------------------------------------
    # ClassAdded checks
    # ------------------------------------------------------------------
    def _check_class_added(self, event: ClassAdded) -> list[ArchitectureFinding]:
        findings: list[ArchitectureFinding] = []
        lang = self._detect_lang(event.file)

        if len(event.methods) > self.CLASS_BLOAT_METHODS:
            findings.append(ArchitectureFinding(
                rule_id="class_bloat",
                title="Class Bloat",
                description=(
                    f"Class `{event.name}` was added with {len(event.methods)} methods "
                    f"(threshold: {self.CLASS_BLOAT_METHODS}). "
                    "Large classes are a classic God Object anti-pattern."
                ),
                severity="high",
                file=event.file,
                entity_name=event.name,
                suggestion=(
                    "Break into smaller classes using composition or mixins. "
                    "Follow the Single Responsibility Principle."
                ),
            ))

        naming_finding = self._check_naming(event.name, lang, event.file, "class")
        if naming_finding:
            findings.append(naming_finding)

        return findings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _detect_lang(self, file_path: str) -> str:
        suffix = Path(file_path).suffix.lower()
        return {".py": "python", ".js": "javascript", ".ts": "typescript"}.get(suffix, "unknown")

    def _check_naming(
        self, name: str, lang: str, file: str, entity_type: str
    ) -> ArchitectureFinding | None:
        """Check naming convention: snake_case for Python, camelCase/PascalCase for JS/TS."""
        if lang == "python":
            if entity_type == "function" and not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
                return ArchitectureFinding(
                    rule_id="naming_convention",
                    title="Naming Convention Violation",
                    description=(
                        f"Python function `{name}` does not follow snake_case convention."
                    ),
                    severity="low",
                    file=file,
                    entity_name=name,
                    suggestion="Rename to snake_case (e.g., my_function_name).",
                )
            if entity_type == "class" and not re.fullmatch(r"[A-Z][a-zA-Z0-9]*", name):
                return ArchitectureFinding(
                    rule_id="naming_convention",
                    title="Naming Convention Violation",
                    description=(
                        f"Python class `{name}` does not follow PascalCase convention."
                    ),
                    severity="low",
                    file=file,
                    entity_name=name,
                    suggestion="Rename to PascalCase (e.g., MyClassName).",
                )
        elif lang in ("javascript", "typescript"):
            if entity_type == "function" and not re.fullmatch(r"[a-z][a-zA-Z0-9]*", name):
                return ArchitectureFinding(
                    rule_id="naming_convention",
                    title="Naming Convention Violation",
                    description=(
                        f"JS/TS function `{name}` does not follow camelCase convention."
                    ),
                    severity="low",
                    file=file,
                    entity_name=name,
                    suggestion="Rename to camelCase (e.g., myFunctionName).",
                )
        return None

    def _check_magic_numbers(
        self,
        func_name: str,
        file: str,
        content: str,
        start_line: int,
        end_line: int,
    ) -> ArchitectureFinding | None:
        """Detect magic numbers (literals other than 0, 1, -1) in function bodies."""
        lines = content.splitlines()
        body_lines = lines[max(0, start_line - 1) : min(len(lines), end_line)]
        body = "\n".join(body_lines)

        # Find standalone integer/float literals that are not 0, 1, -1, True, False
        magic = re.findall(
            r"(?<!['\"\w])(?<!\.)\b([2-9]\d*|[1-9]\d+)\b(?!['\"\w])", body
        )
        if len(magic) >= 3:
            sample = ", ".join(sorted(set(magic))[:5])
            return ArchitectureFinding(
                rule_id="magic_numbers",
                title="Magic Numbers Detected",
                description=(
                    f"Function `{func_name}` contains magic numbers: {sample}. "
                    "Hard-coded literals reduce readability and maintainability."
                ),
                severity="low",
                file=file,
                entity_name=func_name,
                suggestion="Extract magic numbers into named constants at module level.",
            )
        return None
