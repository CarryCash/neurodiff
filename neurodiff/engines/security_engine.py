"""Security analysis engine."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Literal
from pathlib import Path

from neurodiff.core.semantic_events import SemanticEvent, FunctionAdded, FunctionModified


@dataclass
class SecurityFinding:
    """Represents a security finding."""
    severity: Literal["critical", "high", "medium", "low"]
    category: str
    file: str
    line: int
    function_name: str | None
    description: str
    rule_id: str | None


class SecurityEngine:
    """Engine for security analysis using semgrep and pattern matching."""

    # Updated fallback pattern matching
    FALLBACK_PATTERNS = {
        "hardcoded_secret": {
            "pattern": r"(?i)(api_key|secret|password|token)\s*=\s*['\"][^'\"]{8,}['\"]",
            "severity": "critical",
            "category": "secret",
            "description": "Hardcoded secret detected",
        },
    }

    def __init__(self) -> None:
        """Initialize the SecurityEngine."""
        self.has_semgrep = self._check_semgrep()

    def _check_semgrep(self) -> bool:
        """Check if semgrep is available."""
        try:
            result = subprocess.run(
                ["semgrep", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def analyze(self, content_after: str, file_path: str, events: list[SemanticEvent]) -> list[SecurityFinding]:
        """Analyze code for security vulnerabilities."""
        findings: list[SecurityFinding] = []

        # Try semgrep first
        if self.has_semgrep:
            findings.extend(self._analyze_with_semgrep(content_after, file_path))

        # Fall back to pattern matching
        findings.extend(self._analyze_with_patterns(content_after, file_path))

        # Filter findings based on semantic events (overlap logic)
        valid_findings: list[SecurityFinding] = []
        for finding in findings:
            overlapping_function = None
            is_relevant = False
            for event in events:
                if isinstance(event, FunctionAdded):
                    if event.file == finding.file and event.start_line <= finding.line < (event.start_line + event.body_lines):
                        is_relevant = True
                        overlapping_function = event.name
                        break
                elif isinstance(event, FunctionModified):
                    if event.file == finding.file and event.start_line <= finding.line < (event.start_line + event.lines_after):
                        is_relevant = True
                        overlapping_function = event.name
                        break

            if is_relevant:
                finding.function_name = overlapping_function
                valid_findings.append(finding)

        return valid_findings

    def _analyze_with_semgrep(
        self, content: str, file_path: str
    ) -> list[SecurityFinding]:
        """Analyze code using semgrep."""
        findings: list[SecurityFinding] = []

        try:
            temp_path = Path(file_path).with_suffix('.tmp' + Path(file_path).suffix)
            try:
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path.write_text(content)
            except OSError:
                temp_path = Path("/tmp/neurodiff_semgrep_temp.py")
                temp_path.write_text(content)

            result = subprocess.run(
                ["semgrep", "--json", "--config=auto", str(temp_path)],
                capture_output=True,
                text=True,
                timeout=15,
            )

            if result.returncode == 0 and result.stdout:
                import json
                try:
                    data = json.loads(result.stdout)
                    for result_item in data.get("results", []):
                        severity_str = result_item.get("extra", {}).get("severity", "INFO").lower()
                        if severity_str not in ("critical", "high", "medium", "low"):
                            severity_str = self._map_severity(severity_str)
                            
                        finding = SecurityFinding(
                            severity=severity_str, # type: ignore
                            category=result_item.get("extra", {}).get("metadata", {}).get("category", "unknown"),
                            file=file_path,
                            line=result_item.get("start", {}).get("line", 0),
                            function_name=None, # Filled in filter phase
                            description=result_item.get("extra", {}).get("message", "Security vulnerability detected"),
                            rule_id=result_item.get("check_id", "unknown"),
                        )
                        findings.append(finding)
                except json.JSONDecodeError:
                    pass

            temp_path.unlink(missing_ok=True)
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

        return findings

    def _analyze_with_patterns(
        self, content: str, file_path: str
    ) -> list[SecurityFinding]:
        """Analyze code using regex patterns."""
        findings: list[SecurityFinding] = []
        lines = content.splitlines()

        for rule_name, rule_config in self.FALLBACK_PATTERNS.items():
            pattern = rule_config["pattern"]
            severity = rule_config["severity"]
            category = rule_config["category"]
            description = rule_config["description"]

            try:
                regex = re.compile(pattern) # ?i is handled inside pattern if present
                for line_num, line_content in enumerate(lines, 1):
                    if regex.search(line_content):
                        finding = SecurityFinding(
                            severity=severity, # type: ignore
                            category=category,
                            file=file_path,
                            line=line_num,
                            function_name=None, # Filled in filter phase
                            description=description,
                            rule_id=rule_name,
                        )
                        findings.append(finding)
            except re.error:
                pass

        return findings

    def _map_severity(self, semgrep_severity: str) -> Literal["critical", "high", "medium", "low"]:
        """Map semgrep severity to our expected values."""
        severity_map = {
            "error": "critical",
            "warning": "high",
            "info": "low",
        }
        return severity_map.get(semgrep_severity, "medium") # type: ignore
