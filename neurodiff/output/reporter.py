"""Rich-based output reporter."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from neurodiff.core import SemanticEvent
from neurodiff.core.semantic_events import (
    ClassAdded,
    ClassModified,
    FunctionAdded,
    FunctionModified,
    FunctionRemoved,
    ImportAdded,
    ImportRemoved,
)
from neurodiff.engines import DuplicationFinding, SecurityFinding


class Reporter:
    """Reporter for generating rich output."""

    def __init__(self, console: Console | None = None) -> None:
        """Initialize the Reporter.

        Args:
            console: Optional Rich Console instance.
        """
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
    ) -> None:
        """Generate and print a complete analysis report.

        Args:
            repo_path: Path to the analyzed repository.
            base_ref: Base Git reference.
            head_ref: Head Git reference.
            files_analyzed: Number of files analyzed.
            semantic_events: List of semantic events found.
            security_findings: List of security findings.
            duplication_findings: List of duplication findings.
        """
        # Header
        self._print_header(repo_path, base_ref, head_ref, files_analyzed)

        # Semantic Summary
        self._print_semantic_summary(semantic_events)

        # Per-file details
        self._print_file_details(semantic_events)

        # Security Findings
        self._print_security_findings(security_findings)

        # Duplication Warnings
        self._print_duplication_warnings(duplication_findings)

        # Risk Score
        self._print_risk_score(
            semantic_events, security_findings, duplication_findings
        )

    def _print_header(
        self, repo_path: str, base_ref: str, head_ref: str, files_analyzed: int
    ) -> None:
        """Print the header panel.

        Args:
            repo_path: Path to the repository.
            base_ref: Base Git reference.
            head_ref: Head Git reference.
            files_analyzed: Number of files analyzed.
        """
        header_content = f"""[bold cyan]Repository:[/bold cyan] {repo_path}
[bold cyan]Diff Range:[/bold cyan] {base_ref}...{head_ref}
[bold cyan]Files Analyzed:[/bold cyan] {files_analyzed}"""

        panel = Panel(
            header_content,
            title="[bold]NeuroDiff Analysis[/bold]",
            border_style="blue",
        )
        self.console.print(panel)
        self.console.print()

    def _print_semantic_summary(self, events: list[SemanticEvent]) -> None:
        """Print semantic analysis summary table.

        Args:
            events: List of semantic events.
        """
        # Count events by type
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

        table = Table(title="Semantic Summary")
        table.add_column("Event Type", style="cyan")
        table.add_column("Count", style="magenta")

        for event_type, count in event_counts.items():
            if count > 0:
                table.add_row(event_type, str(count))

        self.console.print(table)
        self.console.print()

    def _print_file_details(self, events: list[SemanticEvent]) -> None:
        """Print per-file details with icons.

        Args:
            events: List of semantic events.
        """
        if not events:
            self.console.print("[yellow]No changes detected[/yellow]")
            return

        self.console.print("[bold]File Changes:[/bold]")

        # Group events by file (use event details like language)
        file_events: dict[str, list[SemanticEvent]] = {}

        for event in events:
            key = self._get_event_file_key(event)
            if key not in file_events:
                file_events[key] = []
            file_events[key].append(event)

        for file_key, file_event_list in file_events.items():
            added_count = sum(
                1
                for e in file_event_list
                if isinstance(e, (FunctionAdded, ClassAdded, ImportAdded))
            )
            modified_count = sum(
                1
                for e in file_event_list
                if isinstance(e, (FunctionModified, ClassModified))
            )
            removed_count = sum(
                1
                for e in file_event_list
                if isinstance(e, (FunctionRemoved, ImportRemoved))
            )

            icons = []
            if added_count > 0:
                icons.append(f"➕ {added_count}")
            if modified_count > 0:
                icons.append(f"✏️  {modified_count}")
            if removed_count > 0:
                icons.append(f"❌ {removed_count}")

            icon_str = " ".join(icons)
            self.console.print(f"  {file_key}: {icon_str}")

        self.console.print()

    def _print_security_findings(
        self, findings: list[SecurityFinding]
    ) -> None:
        """Print security findings section.

        Args:
            findings: List of security findings.
        """
        if not findings:
            self.console.print("[green]✓ No security issues detected[/green]")
            self.console.print()
            return

        # Sort by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        sorted_findings = sorted(
            findings,
            key=lambda x: severity_order.get(x.severity.value, 99),
        )

        table = Table(title="Security Findings", style="red")
        table.add_column("Severity", style="red")
        table.add_column("Rule ID", style="yellow")
        table.add_column("File", style="cyan")
        table.add_column("Line", style="magenta")
        table.add_column("Issue", style="white")

        for finding in sorted_findings:
            table.add_row(
                finding.severity.value,
                finding.rule_id,
                Path(finding.file_path).name,
                str(finding.line),
                finding.title[:50],
            )

        self.console.print(table)
        self.console.print()

    def _print_duplication_warnings(
        self, findings: list[DuplicationFinding]
    ) -> None:
        """Print duplication warnings section.

        Args:
            findings: List of duplication findings.
        """
        if not findings:
            self.console.print("[green]✓ No code duplication detected[/green]")
            self.console.print()
            return

        table = Table(title="Code Duplication Warnings", style="yellow")
        table.add_column("Source File", style="cyan")
        table.add_column("Target File", style="cyan")
        table.add_column("Similarity", style="yellow")
        table.add_column("Severity", style="red")

        for finding in findings:
            similarity_str = f"{finding.similarity:.1%}"
            table.add_row(
                Path(finding.source_file).name,
                Path(finding.target_file).name,
                similarity_str,
                finding.severity,
            )

        self.console.print(table)
        self.console.print()

    def _print_risk_score(
        self,
        semantic_events: list[SemanticEvent],
        security_findings: list[SecurityFinding],
        duplication_findings: list[DuplicationFinding],
    ) -> None:
        """Print risk score summary badge.

        Args:
            semantic_events: List of semantic events.
            security_findings: List of security findings.
            duplication_findings: List of duplication findings.
        """
        # Calculate risk score (simplified)
        complexity_increase = sum(
            1
            for e in semantic_events
            if isinstance(e, FunctionModified)
            and e.complexity_after > e.complexity_before
        )

        critical_findings = sum(
            1 for f in security_findings if f.severity.value == "CRITICAL"
        )
        high_findings = sum(
            1 for f in security_findings if f.severity.value == "HIGH"
        )
        high_duplication = sum(
            1 for f in duplication_findings if f.severity == "HIGH"
        )

        risk_score = (
            critical_findings * 10
            + high_findings * 5
            + high_duplication * 2
            + complexity_increase * 1
        )

        if risk_score == 0:
            risk_badge = "[green bold]✓ LOW RISK[/green bold]"
        elif risk_score < 5:
            risk_badge = "[yellow bold]⚠ MEDIUM RISK[/yellow bold]"
        elif risk_score < 10:
            risk_badge = "[orange bold]⚠ HIGH RISK[/orange bold]"
        else:
            risk_badge = "[red bold]🚨 CRITICAL RISK[/red bold]"

        panel = Panel(
            f"Overall Risk Assessment: {risk_badge}\nRisk Score: {risk_score}",
            border_style="red" if risk_score > 0 else "green",
        )
        self.console.print(panel)

    def _get_event_file_key(self, event: SemanticEvent) -> str:
        """Get a file key from an event.

        Args:
            event: A semantic event.

        Returns:
            A file key string.
        """
        if isinstance(event, (FunctionAdded, FunctionModified, FunctionRemoved)):
            return f"{event.language}:{event.name}"
        elif isinstance(event, (ClassAdded, ClassModified)):
            return f"{event.language}:{event.name}"
        elif isinstance(event, (ImportAdded, ImportRemoved)):
            return f"{event.language}:{event.module}"
        return "unknown"
