"""Security analysis engine."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Severity(str, Enum):
    """Security finding severity levels."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class SecurityFinding:
    """Represents a security finding."""

    rule_id: str
    title: str
    description: str
    severity: Severity
    file_path: str
    line: int
    code_snippet: str
    recommendation: str


class SecurityEngine:
    """Engine for security analysis using semgrep and pattern matching."""

    # Regex patterns for common security issues (fallback)
    FALLBACK_PATTERNS = {
        "hardcoded_secret": {
            "pattern": r"(password|secret|api[_-]?key|token)\s*=\s*['\"]([^'\"]+)['\"]",
            "severity": Severity.CRITICAL,
            "title": "Hardcoded Secret Detected",
        },
        "sql_injection": {
            "pattern": r"(execute|query|sql)\s*\(\s*['\"].*\+.*['\"]",
            "severity": Severity.HIGH,
            "title": "SQL Injection Vulnerability",
        },
        "command_injection": {
            "pattern": r"(subprocess|os\.system|popen)\s*\(\s*['\"].*\+.*['\"]",
            "severity": Severity.HIGH,
            "title": "Command Injection Vulnerability",
        },
        "hardcoded_ip": {
            "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "severity": Severity.MEDIUM,
            "title": "Hardcoded IP Address",
        },
    }

    def __init__(self) -> None:
        """Initialize the SecurityEngine."""
        self.has_semgrep = self._check_semgrep()

    def _check_semgrep(self) -> bool:
        """Check if semgrep is available.

        Returns:
            True if semgrep is installed and available.
        """
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

    def analyze(self, content_after: str, file_path: str) -> list[SecurityFinding]:
        """Analyze code for security vulnerabilities.

        Args:
            content_after: The code content to analyze.
            file_path: The path to the file being analyzed.

        Returns:
            List of security findings.
        """
        findings: list[SecurityFinding] = []

        # Try semgrep first
        if self.has_semgrep:
            findings.extend(self._analyze_with_semgrep(content_after, file_path))

        # Fall back to pattern matching
        findings.extend(self._analyze_with_patterns(content_after, file_path))

        return findings

    def _analyze_with_semgrep(
        self, content: str, file_path: str
    ) -> list[SecurityFinding]:
        """Analyze code using semgrep.

        Args:
            content: The code content to analyze.
            file_path: The path to the file being analyzed.

        Returns:
            List of security findings from semgrep.
        """
        findings: list[SecurityFinding] = []

        try:
            # Write content to a temporary file
            temp_path = Path("/tmp/neurodiff_semgrep_temp.py")
            temp_path.write_text(content)

            result = subprocess.run(
                ["semgrep", "--json", str(temp_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0 and result.stdout:
                import json

                data = json.loads(result.stdout)
                for result_item in data.get("results", []):
                    finding = SecurityFinding(
                        rule_id=result_item.get("check_id", "unknown"),
                        title=result_item.get(
                            "check_id", "Security Issue"
                        ).upper(),
                        description=result_item.get("extra", {}).get(
                            "message", "Security vulnerability detected"
                        ),
                        severity=self._map_severity(
                            result_item.get("extra", {}).get(
                                "severity", "INFO"
                            )
                        ),
                        file_path=file_path,
                        line=result_item.get("start", {}).get("line", 0),
                        code_snippet=result_item.get("extra", {}).get(
                            "lines", ""
                        ),
                        recommendation="Review and fix this security issue",
                    )
                    findings.append(finding)

            # Clean up temp file
            temp_path.unlink(missing_ok=True)
        except subprocess.TimeoutExpired:
            pass  # Graceful degradation
        except Exception:
            pass  # Graceful degradation

        return findings

    def _analyze_with_patterns(
        self, content: str, file_path: str
    ) -> list[SecurityFinding]:
        """Analyze code using regex patterns.

        Args:
            content: The code content to analyze.
            file_path: The path to the file being analyzed.

        Returns:
            List of security findings from pattern matching.
        """
        findings: list[SecurityFinding] = []
        lines = content.split("\n")

        for rule_name, rule_config in self.FALLBACK_PATTERNS.items():
            pattern = rule_config["pattern"]
            severity = rule_config["severity"]
            title = rule_config["title"]

            try:
                regex = re.compile(pattern, re.IGNORECASE)
                for line_num, line_content in enumerate(lines, 1):
                    if regex.search(line_content):
                        finding = SecurityFinding(
                            rule_id=rule_name,
                            title=title,
                            description=f"Detected by pattern: {pattern}",
                            severity=severity,
                            file_path=file_path,
                            line=line_num,
                            code_snippet=line_content.strip(),
                            recommendation="Review and address this security concern",
                        )
                        findings.append(finding)
            except re.error:
                pass  # Skip invalid patterns

        return findings

    def _map_severity(self, semgrep_severity: str) -> Severity:
        """Map semgrep severity to our Severity enum.

        Args:
            semgrep_severity: The severity from semgrep.

        Returns:
            The mapped Severity value.
        """
        severity_map = {
            "ERROR": Severity.CRITICAL,
            "WARNING": Severity.HIGH,
            "INFO": Severity.INFO,
        }
        return severity_map.get(semgrep_severity.upper(), Severity.MEDIUM)
