"""SARIF v2.1.0 formatter for NeuroDiff findings.

Converts security, duplication, and architecture findings into the OASIS SARIF
format understood by GitHub Code Scanning and the GitHub Security tab.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


# ---------------------------------------------------------------------------
# SARIF constants
# ---------------------------------------------------------------------------
SARIF_VERSION = "2.1.0"
SARIF_SCHEMA = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"

_LEVEL_MAP = {
    "critical": "error",
    "high": "error",
    "medium": "warning",
    "low": "note",
    "info": "note",
}


def _rule(rule_id: str, name: str, short_desc: str, full_desc: str, tags: list[str]) -> dict:
    return {
        "id": rule_id,
        "name": name,
        "shortDescription": {"text": short_desc},
        "fullDescription": {"text": full_desc},
        "properties": {"tags": tags},
    }


def _result(
    rule_id: str,
    message: str,
    level: str,
    file_path: str | None,
    line: int | None,
) -> dict:
    location: dict[str, Any] = {}
    if file_path:
        location = {
            "physicalLocation": {
                "artifactLocation": {"uri": file_path.replace("\\", "/")},
                "region": {"startLine": max(1, line or 1)},
            }
        }
    return {
        "ruleId": rule_id,
        "level": level,
        "message": {"text": message},
        "locations": [location] if location else [],
    }


def build_sarif(
    security_findings: list,
    duplication_findings: list,
    arch_report: Any | None,
) -> dict:
    """Build a SARIF 2.1.0 document from NeuroDiff findings."""
    rules: list[dict] = []
    results: list[dict] = []

    rule_ids_seen: set[str] = set()

    def ensure_rule(rid: str, name: str, short: str, full: str, tags: list[str]) -> None:
        if rid not in rule_ids_seen:
            rules.append(_rule(rid, name, short, full, tags))
            rule_ids_seen.add(rid)

    # Security findings
    for f in security_findings:
        rid = f"ND-SEC-{f.category.upper().replace(' ', '-')}"
        ensure_rule(
            rid,
            f"Security/{f.category}",
            f"Security issue detected: {f.category}",
            f"NeuroDiff Security Engine detected a {f.severity} severity {f.category} issue.",
            ["security", "neurodiff"],
        )
        results.append(_result(
            rid,
            f.description,
            _LEVEL_MAP.get(f.severity, "warning"),
            getattr(f, "file", None),
            getattr(f, "line", None),
        ))

    # Duplication findings
    for f in duplication_findings:
        rid = "ND-DUP-CODE-CLONE"
        ensure_rule(
            rid,
            "Duplication/CodeClone",
            "Duplicate code detected",
            "NeuroDiff Duplication Engine detected a code clone above the similarity threshold.",
            ["duplication", "code-smell", "neurodiff"],
        )
        score_pct = round(f.similarity_score * 100, 1)
        results.append(_result(
            rid,
            f"{f.new_function} ({f.new_file}) is {score_pct}% similar to {f.similar_function} ({f.similar_file}). Consider merging.",
            "warning" if score_pct < 90 else "error",
            getattr(f, "new_file", None),
            None,
        ))

    # Architecture findings
    if arch_report:
        for f in arch_report.layer_violations:
            rid = "ND-ARCH-LAYER-VIOLATION"
            ensure_rule(
                rid,
                "Architecture/LayerViolation",
                "Architectural layer violation detected",
                "NeuroDiff Architecture Engine detected a forbidden import between architectural layers.",
                ["architecture", "neurodiff"],
            )
            results.append(_result(
                rid,
                f.description,
                _LEVEL_MAP.get(f.severity, "warning"),
                None,
                None,
            ))

        for f in arch_report.circular_deps:
            rid = "ND-ARCH-CIRCULAR-DEP"
            ensure_rule(
                rid,
                "Architecture/CircularDependency",
                "Circular dependency detected",
                "NeuroDiff Architecture Engine detected a circular import cycle in the dependency graph.",
                ["architecture", "neurodiff"],
            )
            results.append(_result(
                rid,
                f.description,
                _LEVEL_MAP.get(f.severity, "error"),
                None,
                None,
            ))

        for f in arch_report.solid_findings:
            rid = f"ND-ARCH-SOLID-{getattr(f, 'principle', 'VIOLATION').upper()}"
            ensure_rule(
                rid,
                f"Architecture/SOLID/{getattr(f, 'principle', 'Violation')}",
                f"SOLID principle violation: {getattr(f, 'principle', 'Unknown')}",
                "NeuroDiff Architecture Engine detected a heuristic violation of a SOLID principle.",
                ["architecture", "solid", "neurodiff"],
            )
            results.append(_result(
                rid,
                f.description,
                _LEVEL_MAP.get(f.severity, "warning"),
                None,
                None,
            ))

    return {
        "$schema": SARIF_SCHEMA,
        "version": SARIF_VERSION,
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "NeuroDiff",
                        "version": "0.1.0",
                        "informationUri": "https://github.com/neurodiff/neurodiff",
                        "rules": rules,
                    }
                },
                "results": results,
            }
        ],
    }
