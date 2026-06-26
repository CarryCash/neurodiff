"""Rich-based output reporter."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


from neurodiff.core.semantic_events import (
    ClassAdded,
    ClassModified,
    FunctionAdded,
    FunctionModified,
    FunctionRemoved,
    ImportAdded,
    ImportRemoved,
    SemanticEvent,
)
from neurodiff.engines.architecture_engine import ArchitectureFinding
from neurodiff.engines.duplication_engine import DuplicationFinding
from neurodiff.engines.security_engine import SecurityFinding
from neurodiff.engines.llm_engine import LLMReport

# Severity order for sorting (low index = highest severity)
_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


class Reporter:
    """Reporter for generating rich output."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def report(
        self,
        repo_path: str,
        base_ref: str,
        head_ref: str,
        files_analyzed: int,
        semantic_events: list[SemanticEvent],
        security_findings: list[SecurityFinding],
        duplication_findings: list[DuplicationFinding],
        architecture_findings: list[ArchitectureFinding] | None = None,
        arch_report: Any = None,
        llm_report: LLMReport | None = None,
        cognitive_report: Any | None = None,
        verbose: bool = False,
    ) -> None:
        """Generate and print a complete analysis report."""
        self._print_header(repo_path, base_ref, head_ref, files_analyzed)
        self._print_semantic_summary(semantic_events)
        self._print_file_details(semantic_events)
        self._print_security_findings(security_findings)
        self._print_duplication_warnings(duplication_findings)
        if architecture_findings is not None:
            self._print_architecture_findings(architecture_findings)
        if arch_report is not None:
            self._print_arch_report(arch_report)
        
        if cognitive_report is not None:
            self._print_cognitive_report(cognitive_report)
            
        self._print_risk_score(
            semantic_events, security_findings, duplication_findings,
            architecture_findings or [], arch_report
        )

        if llm_report:
            self._print_llm_report(llm_report)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    def _print_header(
        self, repo_path: str, base_ref: str, head_ref: str, files_analyzed: int
    ) -> None:
        header_content = (
            f"[bold cyan]Repository:[/bold cyan] {repo_path}\n"
            f"[bold cyan]Diff Range:[/bold cyan] {base_ref}...{head_ref}\n"
            f"[bold cyan]Files Analyzed:[/bold cyan] {files_analyzed}"
        )
        panel = Panel(
            header_content,
            title="[bold white]🧠 NeuroDiff — Semantic Analysis[/bold white]",
            border_style="bright_blue",
        )
        self.console.print(panel)
        self.console.print()

    # ------------------------------------------------------------------
    # Semantic Summary
    # ------------------------------------------------------------------
    def _print_semantic_summary(self, events: list[SemanticEvent]) -> None:
        event_counts: dict[str, int] = {
            "Functions Added": 0,
            "Functions Modified": 0,
            "Functions Removed": 0,
            "Classes Added": 0,
            "Classes Modified": 0,
            "Imports Added": 0,
            "Imports Removed": 0,
        }

        for event in events:
            if isinstance(event, FunctionAdded):
                event_counts["Functions Added"] += 1
            elif isinstance(event, FunctionModified):
                event_counts["Functions Modified"] += 1
            elif isinstance(event, FunctionRemoved):
                event_counts["Functions Removed"] += 1
            elif isinstance(event, ClassAdded):
                event_counts["Classes Added"] += 1
            elif isinstance(event, ClassModified):
                event_counts["Classes Modified"] += 1
            elif isinstance(event, ImportAdded):
                event_counts["Imports Added"] += 1
            elif isinstance(event, ImportRemoved):
                event_counts["Imports Removed"] += 1

        table = Table(title="📊 Semantic Summary", border_style="blue")
        table.add_column("Event Type", style="cyan")
        table.add_column("Count", justify="right", style="magenta bold")

        for event_type, count in event_counts.items():
            if count > 0:
                table.add_row(event_type, str(count))

        if all(c == 0 for c in event_counts.values()):
            self.console.print("[yellow]No semantic changes detected.[/yellow]")
        else:
            self.console.print(table)
        self.console.print()

    # ------------------------------------------------------------------
    # Per-file details
    # ------------------------------------------------------------------
    def _print_file_details(self, events: list[SemanticEvent]) -> None:
        if not events:
            self.console.print("[yellow]No changes detected[/yellow]\n")
            return

        self.console.print("[bold white]📁 File Changes:[/bold white]")

        # Group events by file
        file_events: dict[str, list[SemanticEvent]] = {}
        for event in events:
            file_key = self._get_event_file(event)
            file_events.setdefault(file_key, []).append(event)

        for file_path, file_event_list in sorted(file_events.items()):
            added_count = sum(
                1 for e in file_event_list
                if isinstance(e, (FunctionAdded, ClassAdded, ImportAdded))
            )
            modified_count = sum(
                1 for e in file_event_list
                if isinstance(e, (FunctionModified, ClassModified))
            )
            removed_count = sum(
                1 for e in file_event_list
                if isinstance(e, (FunctionRemoved, ImportRemoved))
            )

            icons = []
            if added_count > 0:
                icons.append(f"[green]➕ {added_count}[/green]")
            if modified_count > 0:
                icons.append(f"[yellow]✏️  {modified_count}[/yellow]")
            if removed_count > 0:
                icons.append(f"[red]❌ {removed_count}[/red]")

            self.console.print(f"  [cyan]{file_path}[/cyan]  {' '.join(icons)}")

            # Detail lines per event
            for event in file_event_list:
                detail = self._format_event_detail(event)
                if detail:
                    self.console.print(f"    {detail}")

        self.console.print()

    def _format_event_detail(self, event: SemanticEvent) -> str:
        """Return a short detail string for an event."""
        if isinstance(event, FunctionAdded):
            cc = event.cyclomatic_complexity
            cc_warn = " [red bold][CC!][/red bold]" if cc > 10 else ""
            return f"[green]+[/green] fn [bold]{event.name}[/bold]  lines={event.body_lines}  cc={cc}{cc_warn}"
        elif isinstance(event, FunctionModified):
            cc_delta = event.complexity_after - event.complexity_before
            cc_str = f"{event.complexity_after}"
            if cc_delta > 0:
                cc_str += f" [red](+{cc_delta})[/red]"
            elif cc_delta < 0:
                cc_str += f" [green]({cc_delta})[/green]"
            cc_warn = " [red bold][CC!][/red bold]" if event.complexity_after > 10 else ""
            return f"[yellow]~[/yellow] fn [bold]{event.name}[/bold]  lines {event.lines_before}→{event.lines_after}  cc={cc_str}{cc_warn}"
        elif isinstance(event, FunctionRemoved):
            return f"[red]-[/red] fn [bold]{event.name}[/bold]"
        elif isinstance(event, ClassAdded):
            methods = ", ".join(event.methods[:5])
            if len(event.methods) > 5:
                methods += "…"
            inh = f"  inherits=[{', '.join(event.inherits_from)}]" if event.inherits_from else ""
            return f"[green]+[/green] class [bold]{event.name}[/bold]  methods=[{methods}]{inh}"
        elif isinstance(event, ClassModified):
            parts = []
            if event.methods_added:
                parts.append(f"[green]+{', '.join(event.methods_added)}[/green]")
            if event.methods_removed:
                parts.append(f"[red]-{', '.join(event.methods_removed)}[/red]")
            return f"[yellow]~[/yellow] class [bold]{event.name}[/bold]  " + "  ".join(parts)
        elif isinstance(event, ImportAdded):
            syms = f" ({', '.join(event.symbols)})" if event.symbols else ""
            return f"[green]+[/green] import [bold]{event.module}[/bold]{syms}"
        elif isinstance(event, ImportRemoved):
            return f"[red]-[/red] import [bold]{event.module}[/bold]"
        return ""

    # ------------------------------------------------------------------
    # Security Findings
    # ------------------------------------------------------------------
    def _print_security_findings(self, findings: list[SecurityFinding]) -> None:
        if not findings:
            self.console.print("[green]✓ No security issues detected[/green]")
            self.console.print()
            return

        sorted_findings = sorted(
            findings,
            key=lambda x: _SEVERITY_ORDER.get(x.severity.lower(), 99),
        )

        table = Table(title="🔴 Security Findings", border_style="red")
        table.add_column("Severity", style="bold")
        table.add_column("Category", style="yellow")
        table.add_column("File", style="cyan")
        table.add_column("Line", justify="right", style="magenta")
        table.add_column("Function", style="green")
        table.add_column("Description", style="white")
        table.add_column("Rule ID", style="dim")

        severity_colors = {
            "critical": "red bold",
            "high": "red",
            "medium": "yellow",
            "low": "cyan",
            "info": "blue",
        }

        for f in sorted_findings:
            sev_color = severity_colors.get(f.severity.lower(), "white")
            table.add_row(
                f"[{sev_color}]{f.severity.upper()}[/{sev_color}]",
                f.category,
                Path(f.file).name,
                str(f.line),
                f.function_name or "—",
                f.description[:60],
                f.rule_id or "—",
            )

        self.console.print(table)
        self.console.print()

    # ------------------------------------------------------------------
    # Duplication Warnings
    # ------------------------------------------------------------------
    def _print_duplication_warnings(self, findings: list[DuplicationFinding]) -> None:
        if not findings:
            self.console.print("[green]✓ No code duplication detected[/green]")
            self.console.print()
            return

        table = Table(title="🟠 Duplication Warnings", border_style="yellow")
        table.add_column("New Function", style="green bold")
        table.add_column("New File", style="cyan")
        table.add_column("Similar Function", style="yellow bold")
        table.add_column("Similar File", style="cyan")
        table.add_column("Similarity", justify="right")
        table.add_column("Severity", style="bold")

        severity_colors = {"high": "red", "medium": "yellow"}

        for f in sorted(findings, key=lambda x: -x.similarity_score):
            sev_color = severity_colors.get(f.severity.lower(), "white")
            similarity_pct = f"{f.similarity_score:.1%}"
            if f.similarity_score >= 0.9:
                similarity_pct = f"[red bold]{similarity_pct}[/red bold]"
            else:
                similarity_pct = f"[yellow]{similarity_pct}[/yellow]"

            table.add_row(
                f.new_function,
                Path(f.new_file).name,
                f.similar_function,
                Path(f.similar_file).name,
                similarity_pct,
                f"[{sev_color}]{f.severity.upper()}[/{sev_color}]",
            )

        self.console.print(table)
        self.console.print(
            "[dim]💡 Suggestion: Consider extracting shared logic into a common utility function.[/dim]"
        )
        self.console.print()

    # ------------------------------------------------------------------
    # Architecture Findings
    # ------------------------------------------------------------------
    def _print_architecture_findings(self, findings: list[ArchitectureFinding]) -> None:
        if not findings:
            self.console.print("[green]✓ No architectural concerns detected[/green]")
            self.console.print()
            return

        table = Table(title="🏛️  Architecture Analysis", border_style="blue")
        table.add_column("Severity", style="bold")
        table.add_column("Rule", style="dim")
        table.add_column("File", style="cyan")
        table.add_column("Entity", style="green bold")
        table.add_column("Issue", style="white")
        table.add_column("Suggestion", style="dim italic")

        severity_colors = {
            "critical": "red bold", "high": "red",
            "medium": "yellow", "low": "cyan",
        }
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_findings = sorted(findings, key=lambda f: severity_order.get(f.severity, 9))

        for f in sorted_findings:
            color = severity_colors.get(f.severity, "white")
            table.add_row(
                f"[{color}]{f.severity.upper()}[/{color}]",
                f.rule_id,
                Path(f.file).name,
                f.entity_name,
                f.title,
                f.suggestion[:60],
            )

        self.console.print(table)
        self.console.print()

    def _print_arch_report(self, report: Any) -> None:
        """Print the complete architectural report (layer violations, circular deps, SOLID findings)."""
        pass # Placeholder for actual implementation if needed

    # ------------------------------------------------------------------
    # Cognitive Report
    # ------------------------------------------------------------------
    def _print_cognitive_report(self, report: Any) -> None:
        """Print the cognitive load and AI generation analysis."""
        self.console.print("\n[bold cyan]━━━ Cognitive Load Analysis ━━━━━━━━━━━━━━━━━━[/bold cyan]\n")

        # 1. AI-Generated Probability
        ai = report.ai_generated
        conf = "HIGH" if ai.confidence == "high" else "MED" if ai.confidence == "medium" else "LOW"
        color = "red" if ai.probability > 0.7 else "yellow" if ai.probability > 0.4 else "green"
        self.console.print(f"[bold]AI-Generated Code Probability:[/bold] [{color}]{ai.probability:.0%}[/{color}] ({conf} confidence)")
        
        if ai.signals_triggered:
            self.console.print(f"  [dim]Signals: {', '.join(ai.signals_triggered)}[/dim]")
        self.console.print(f"  [dim]Estimated: ~{ai.estimated_ai_lines} AI lines / ~{ai.human_lines} human lines[/dim]\n")

        # 2. Cognitive Fatigue Index
        cfi = report.fatigue_index
        grade_color = {"A": "green", "B": "cyan", "C": "yellow", "D": "red", "F": "red bold"}
        g_col = grade_color.get(cfi.grade, "white")
        
        self.console.print(f"[bold]Cognitive Fatigue Index:[/bold] {cfi.total_score}/100  [bold {g_col}][Grade: {cfi.grade}][/bold {g_col}]")
        
        comp = cfi.component_scores
        table = Table(box=None, show_header=False)
        table.add_column(style="dim")
        table.add_column(justify="right", style="cyan")
        
        table.add_row("  Volume:", f"{comp.get('A_volume', 0)}/25")
        table.add_row("  Spread:", f"{comp.get('B_spread', 0)}/20")
        table.add_row("  Conceptual Dist:", f"{comp.get('C_conceptual_distance', 0)}/20")
        table.add_row("  Context Switches:", f"{comp.get('D_context_switches', 0)}/15")
        table.add_row("  Complexity Delta:", f"{comp.get('E_complexity_delta', 0)}/10")
        table.add_row("  AI Amplification:", f"{comp.get('F_ai_amplification', 0)}/10")
        self.console.print(table)
        
        self.console.print(f"\n  [dim]Estimated review time: {cfi.review_time_estimate}[/dim]")
        self.console.print(f"  [dim]Recommendation: {cfi.reviewer_recommendation}[/dim]\n")

        # 3. Blast Radius
        br = report.blast_radius
        r_col = {"contained": "green", "moderate": "yellow", "wide": "red", "critical": "red bold"}
        r_c = r_col.get(br.risk_score, "white")
        
        self.console.print(f"[bold]Blast Radius:[/bold] [{r_c}]{br.risk_score.upper()}[/{r_c}]  [dim]({br.total_affected} modules affected)[/dim]")
        if br.hotspot:
            self.console.print(f"  [dim]Hotspot: {br.hotspot} → {br.hotspot_dependents} dependents[/dim]")
        self.console.print()

        # 4. Commit Anti-Patterns
        ap = report.commit_patterns
        if ap.anti_patterns:
            self.console.print("[bold]Commit Anti-Patterns Detected:[/bold]")
            for p in ap.anti_patterns:
                icon = "🔴" if p.severity == "high" else "🟡" if p.severity == "medium" else "🟢"
                self.console.print(f"  {icon} [bold]{p.severity.upper():<4}[/bold] \"{p.name}\"")
                self.console.print(f"         [dim]— {p.evidence}[/dim]")
            self.console.print()

        # 5. Overall Verdict
        v_map = {"safe": ("✅ SAFE", "green"), "caution": ("⚠ CAUTION", "yellow"), "danger": ("⛔ DANGER", "red bold")}
        v_str, v_col = v_map.get(report.overall_verdict, ("UNKNOWN", "white"))
        self.console.print(f"[bold]Overall Verdict:[/bold] [{v_col}]{v_str}[/{v_col}]\n")

    # ------------------------------------------------------------------
    # Risk Score
    # ------------------------------------------------------------------
    def _print_risk_score(
        self,
        semantic_events: list[SemanticEvent],
        security_findings: list[SecurityFinding],
        duplication_findings: list[DuplicationFinding],
        architecture_findings: list[ArchitectureFinding] | None = None,
        arch_report: Any = None,
    ) -> None:
        arch = architecture_findings or []
        has_critical_security = any(f.severity.lower() == "critical" for f in security_findings)
        has_high_security = any(f.severity.lower() == "high" for f in security_findings)
        has_high_duplication = any(f.similarity_score > 0.90 for f in duplication_findings)
        has_high_architecture = any(f.severity in ("critical", "high") for f in arch)

        complexity_warnings = [
            e for e in semantic_events
            if isinstance(e, (FunctionAdded, FunctionModified))
            and (
                (isinstance(e, FunctionAdded) and e.cyclomatic_complexity > 10)
                or (isinstance(e, FunctionModified) and e.complexity_after > 10)
            )
        ]

        if has_critical_security:
            risk_level, risk_badge, border = "CRITICAL", "[red bold]🚨 CRITICAL[/red bold]", "red"
        elif has_high_security or has_high_duplication or has_high_architecture:
            risk_level, risk_badge, border = "HIGH", "[red]⚠ HIGH[/red]", "red"
        elif security_findings or duplication_findings or complexity_warnings or arch:
            risk_level, risk_badge, border = "MEDIUM", "[yellow bold]⚠ MEDIUM[/yellow bold]", "yellow"
        else:
            risk_level, risk_badge, border = "LOW", "[green bold]✓ LOW[/green bold]", "green"

        lines = [
            f"Overall Risk: {risk_badge}",
            f"Security:      {len(security_findings):>3}  findings  "
            f"(critical={sum(1 for f in security_findings if f.severity == 'critical')}, "
            f"high={sum(1 for f in security_findings if f.severity == 'high')})",
            f"Duplication:   {len(duplication_findings):>3}  alerts    "
            f"(>90%={sum(1 for f in duplication_findings if f.similarity_score > 0.90)})",
            f"Architecture:  {len(arch):>3}  concerns  "
            f"(high={sum(1 for f in arch if f.severity in ('critical','high'))})",
            f"Complexity:    {len(complexity_warnings):>3}  warnings  (cc > 10)",
        ]

        panel = Panel(
            "\n".join(lines),
            title="[bold]📋 Risk Assessment[/bold]",
            border_style=border,
        )
        self.console.print(panel)


    # ------------------------------------------------------------------
    # LLM Deep Analysis
    # ------------------------------------------------------------------
    def _print_llm_report(self, report: LLMReport) -> None:
        if report.error:
            self.console.print(f"[red bold]LLM Error:[/red bold] {report.error}")
            return

        self.console.print(f"\n[bold magenta]━━━ 🤖 LLM Deep Analysis ({report.provider_used}) ━━━━━━━━━━━━━━[/bold magenta]\n")

        # Call 1: Executive Summary
        exec_sum = report.executive_summary
        if exec_sum:
            safe = exec_sum.get("safe_to_merge")
            verdict_icon = "✅ SAFE TO MERGE" if safe else "⛔ NOT SAFE TO MERGE"
            conf = exec_sum.get("confidence", 0)
            conf_str = f"[{'HIGH' if conf >= 0.8 else 'MEDIUM' if conf >= 0.5 else 'LOW'} confidence: {conf}]"
            
            self.console.print(f"[bold]Verdict:[/bold] {verdict_icon}  [dim]{conf_str}[/dim]")
            self.console.print(f"[italic]\"{exec_sum.get('one_line_verdict', '')}\"[/italic]\n")

            if not safe and exec_sum.get("merge_blockers"):
                self.console.print("[bold red]Merge Blockers:[/bold red]")
                for blocker in exec_sum.get("merge_blockers", []):
                    self.console.print(f"  • {blocker}")
                self.console.print()

        # Call 2: Pattern Analysis
        patt_an = report.pattern_analysis
        if patt_an:
            patterns = patt_an.get("patterns_detected", [])
            if patterns:
                self.console.print("[bold cyan]Design Patterns Detected:[/bold cyan]")
                for p in patterns:
                    sev = p.get("severity", "medium").upper()
                    self.console.print(f"  • {p.get('pattern')} ({sev})")
                    for ev in p.get("evidence", []):
                        self.console.print(f"    [dim]Evidence: {ev}[/dim]")
                self.console.print()

            tech_debt = patt_an.get("estimated_tech_debt_hours")
            if tech_debt:
                self.console.print(f"[bold]Estimated Tech Debt:[/bold] ~{tech_debt} hours to address HIGH+ findings\n")

        # Call 3: Fix Plan
        fix_plan = report.fix_plan
        if fix_plan:
            actions = fix_plan.get("immediate_actions", [])
            if actions:
                self.console.print("[bold yellow]Immediate Actions (priority order):[/bold yellow]")
                for act in sorted(actions, key=lambda x: x.get("priority", 99)):
                    file_tag = f"[{act.get('file')}] " if act.get('file') else ""
                    self.console.print(f"  {act.get('priority', '*')}. {file_tag}{act.get('action')}")
                self.console.print()

            checklist = fix_plan.get("before_merge_checklist", [])
            if checklist:
                self.console.print("[bold]Before Merge Checklist:[/bold]")
                for item in checklist:
                    self.console.print(f"  ✗ {item}")
                self.console.print()

            split = fix_plan.get("suggested_split")
            if split is not None:
                split_str = "YES" if split else "NO"
                self.console.print(f"[bold]Split PR recommended:[/bold] {split_str}")
                if split and fix_plan.get("split_rationale"):
                    self.console.print(f"  [dim]\"{fix_plan.get('split_rationale')}\"[/dim]")
        
        self.console.print()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_event_file(self, event: SemanticEvent) -> str:
        """Extract the file path from any semantic event."""
        if isinstance(event, (
            FunctionAdded, FunctionModified, FunctionRemoved,
            ClassAdded, ClassModified, ImportAdded, ImportRemoved,
        )):
            return event.file  # type: ignore[union-attr]
        return "unknown"
