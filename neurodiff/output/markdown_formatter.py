"""Markdown formatter for NeuroDiff reports.

Generates GitHub-compatible Markdown for PR comments and CI logs.
Rich console markup is NOT used here — only standard Markdown.
"""
from __future__ import annotations

from typing import Any


def _severity_badge(severity: str) -> str:
    badges = {
        "critical": "🔴 CRITICAL",
        "high": "🟠 HIGH",
        "medium": "🟡 MEDIUM",
        "low": "🟢 LOW",
        "info": "ℹ️ INFO",
    }
    return badges.get(severity.lower(), severity.upper())


def build_pr_comment(
    base_ref: str,
    head_ref: str,
    files_changed: int,
    semantic_events: list,
    security_findings: list,
    duplication_findings: list,
    arch_report: Any | None,
    llm_report: Any | None,
    is_safe: bool,
) -> str:
    """Build a GitHub PR comment in Markdown."""
    lines: list[str] = []

    # Header
    verdict_emoji = "✅" if is_safe else "⛔"
    verdict_text = "SAFE TO MERGE" if is_safe else "NOT SAFE TO MERGE"
    lines.append(f"## {verdict_emoji} NeuroDiff — {verdict_text}")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Base → Head | `{base_ref}` → `{head_ref}` |")
    lines.append(f"| Files Changed | {files_changed} |")
    lines.append(f"| Semantic Events | {len(semantic_events)} |")
    lines.append(f"| Security Findings | {len(security_findings)} |")
    lines.append(f"| Duplication Findings | {len(duplication_findings)} |")
    lines.append(f"")

    # LLM verdict (if available)
    if llm_report and llm_report.executive_summary:
        es = llm_report.executive_summary
        lines.append(f"### 🤖 LLM Verdict ({llm_report.provider_used})")
        conf = es.get("confidence", 0)
        conf_label = "HIGH" if conf >= 0.8 else "MEDIUM" if conf >= 0.5 else "LOW"
        lines.append(f"> **{es.get('one_line_verdict', '')}**  ")
        lines.append(f"> Confidence: {conf_label} ({conf:.0%}) · Risk: `{es.get('overall_risk', 'unknown').upper()}`")
        lines.append(f"")

        if not is_safe and es.get("merge_blockers"):
            lines.append(f"#### ❌ Merge Blockers")
            for b in es["merge_blockers"]:
                lines.append(f"- {b}")
            lines.append(f"")

        if llm_report.pattern_analysis:
            pa = llm_report.pattern_analysis
            patterns = pa.get("patterns_detected", [])
            if patterns:
                lines.append(f"#### 🔍 Design Patterns Detected")
                for p in patterns:
                    sev = _severity_badge(p.get("severity", "medium"))
                    lines.append(f"- **{p.get('pattern')}** — {sev}")
                    for ev in p.get("evidence", []):
                        lines.append(f"  - *{ev}*")
                lines.append(f"")
            tech_debt = pa.get("estimated_tech_debt_hours")
            if tech_debt:
                lines.append(f"⏱️ **Estimated Tech Debt:** ~{tech_debt} hours to address HIGH+ findings")
                lines.append(f"")

        if llm_report.fix_plan:
            fp = llm_report.fix_plan
            actions = fp.get("immediate_actions", [])
            if actions:
                lines.append(f"#### 🛠️ Immediate Actions")
                for act in sorted(actions, key=lambda x: x.get("priority", 99)):
                    file_tag = f"`{act.get('file')}` — " if act.get("file") else ""
                    lines.append(f"{act.get('priority')}. {file_tag}{act.get('action')}")
                lines.append(f"")
            checklist = fp.get("before_merge_checklist", [])
            if checklist:
                lines.append(f"#### ✅ Before Merge Checklist")
                for item in checklist:
                    lines.append(f"- [ ] {item}")
                lines.append(f"")

    # Security findings
    critical_sec = [f for f in security_findings if f.severity in ("critical", "high")]
    if critical_sec:
        lines.append(f"### 🔐 Critical Security Findings")
        lines.append(f"")
        lines.append(f"| Severity | File | Line | Description |")
        lines.append(f"|----------|------|------|-------------|")
        for f in critical_sec[:10]:  # cap at 10 in PR comment
            sev = _severity_badge(f.severity)
            file_str = getattr(f, "file", "—")
            line_str = str(getattr(f, "line", "—"))
            lines.append(f"| {sev} | `{file_str}` | {line_str} | {f.description} |")
        lines.append(f"")

    # Architecture
    if arch_report:
        arch_issues = arch_report.layer_violations + arch_report.circular_deps
        if arch_issues:
            lines.append(f"### 🏗️ Architecture Issues")
            for f in arch_issues:
                lines.append(f"- {_severity_badge(f.severity)}: {f.description}")
            lines.append(f"")

        br = arch_report.blast_radius
        if br and br.get("total_affected", 0) > 0:
            lines.append(f"### 💥 Blast Radius")
            lines.append(f"This diff potentially affects **{br['total_affected']} modules** transitively.")
            lines.append(f"")

    # Duplication
    crit_dup = [f for f in duplication_findings if f.similarity_score >= 0.85]
    if crit_dup:
        lines.append(f"### 🔁 Significant Code Duplication")
        for f in crit_dup[:5]:
            pct = round(f.similarity_score * 100, 1)
            lines.append(f"- `{f.new_function}` ({f.new_file}) is **{pct}%** similar to `{f.similar_function}` ({f.similar_file})")
        lines.append(f"")

    # Footer
    lines.append(f"---")
    lines.append(f"<sub>🧠 Generated by [NeuroDiff](https://github.com/neurodiff/neurodiff) · Semantic AI Code Review</sub>")

    return "\n".join(lines)
