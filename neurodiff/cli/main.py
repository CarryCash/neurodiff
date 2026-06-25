"""NeuroDiff CLI main module."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer
from rich.console import Console

from neurodiff.core.ast_engine import ASTEngine
from neurodiff.core.git_parser import GitParser
from neurodiff.core.semantic_events import NeuroDiffError
from neurodiff.engines.duplication_engine import DuplicationEngine
from neurodiff.engines.security_engine import SecurityEngine
from neurodiff.output.reporter import Reporter

app = typer.Typer(
    name="neurodiff",
    help="Semantic git diff analysis CLI tool",
)

console = Console()


@app.command()
def analyze(
    base_ref: str = typer.Argument(..., help="Base Git reference"),
    head_ref: str = typer.Argument(..., help="Head Git reference"),
    repo_path: str = typer.Option(
        ".",
        "--repo-path",
        "-r",
        help="Path to the Git repository",
    ),
    format_type: Literal["terminal", "json"] = typer.Option(
        "terminal",
        "--format",
        "-f",
        help="Output format",
    ),
    language: Literal["python", "javascript", "typescript", "auto"] = typer.Option(
        "auto",
        "--lang",
        "-l",
        help="Programming language to analyze",
    ),
) -> None:
    """Analyze semantic differences between two Git references.

    Example:
        neurodiff analyze main feature/new-feature
        neurodiff analyze HEAD~5 HEAD --format json
        neurodiff analyze v1.0 v2.0 --repo-path /path/to/repo
    """
    try:
        repo_path_obj = Path(repo_path).resolve()

        # Initialize components
        console.print("[blue]Initializing NeuroDiff...[/blue]")
        git_parser = GitParser(repo_path_obj)
        ast_engine = ASTEngine()
        security_engine = SecurityEngine()
        duplication_engine = DuplicationEngine()

        # Get file diffs
        console.print(f"[blue]Analyzing diffs: {base_ref}...{head_ref}[/blue]")
        file_diffs = git_parser.get_file_diffs(base_ref, head_ref)

        if not file_diffs:
            console.print("[yellow]No changes found between the specified refs[/yellow]")
            raise typer.Exit(0)

        # Analyze each file
        all_semantic_events = []
        all_security_findings = []
        all_duplication_snippets = []

        for file_diff in file_diffs:
            # Skip if language doesn't match (if specific language requested)
            if language != "auto" and file_diff.language != language:
                continue

            console.print(
                f"  Analyzing [cyan]{file_diff.path}[/cyan] "
                f"([yellow]{file_diff.language}[/yellow])"
            )

            # Extract semantic events
            try:
                events = ast_engine.extract_events(
                    file_diff.content_before,
                    file_diff.content_after,
                    file_diff.language,
                )
                all_semantic_events.extend(events)
            except Exception as e:
                console.print(
                    f"  [yellow]Warning: Could not extract events: {e}[/yellow]"
                )

            # Security analysis
            try:
                findings = security_engine.analyze(
                    file_diff.content_after, file_diff.path
                )
                all_security_findings.extend(findings)
            except Exception as e:
                console.print(
                    f"  [yellow]Warning: Could not run security analysis: {e}[/yellow]"
                )

            # Collect snippets for duplication analysis
            if file_diff.content_after.strip():
                all_duplication_snippets.append(
                    (file_diff.path, file_diff.content_after)
                )

        # Duplication analysis
        try:
            all_duplication_findings = duplication_engine.analyze(
                all_duplication_snippets
            )
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not run duplication analysis: {e}[/yellow]"
            )
            all_duplication_findings = []

        console.print()

        # Generate report
        if format_type == "json":
            _output_json(
                all_semantic_events,
                all_security_findings,
                all_duplication_findings,
            )
        else:
            reporter = Reporter(console)
            reporter.report(
                str(repo_path_obj),
                base_ref,
                head_ref,
                len(file_diffs),
                all_semantic_events,
                all_security_findings,
                all_duplication_findings,
            )

    except NeuroDiffError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


def _output_json(
    semantic_events: list,
    security_findings: list,
    duplication_findings: list,
) -> None:
    """Output analysis results as JSON.

    Args:
        semantic_events: List of semantic events.
        security_findings: List of security findings.
        duplication_findings: List of duplication findings.
    """
    import json

    from neurodiff.core.semantic_events import (
        ClassAdded,
        ClassModified,
        FunctionAdded,
        FunctionModified,
        FunctionRemoved,
        ImportAdded,
        ImportRemoved,
    )

    def event_to_dict(event) -> dict:
        """Convert semantic event to dictionary."""
        if isinstance(event, FunctionAdded):
            return {
                "type": "FunctionAdded",
                "name": event.name,
                "language": event.language,
                "line_start": event.line_start,
                "line_end": event.line_end,
                "cyclomatic_complexity": event.cyclomatic_complexity,
            }
        elif isinstance(event, FunctionModified):
            return {
                "type": "FunctionModified",
                "name": event.name,
                "language": event.language,
                "complexity_before": event.complexity_before,
                "complexity_after": event.complexity_after,
                "changes_summary": event.changes_summary,
            }
        elif isinstance(event, FunctionRemoved):
            return {
                "type": "FunctionRemoved",
                "name": event.name,
                "language": event.language,
                "cyclomatic_complexity": event.cyclomatic_complexity,
            }
        elif isinstance(event, ClassAdded):
            return {
                "type": "ClassAdded",
                "name": event.name,
                "language": event.language,
                "methods": event.methods,
            }
        elif isinstance(event, ClassModified):
            return {
                "type": "ClassModified",
                "name": event.name,
                "language": event.language,
                "methods_added": event.methods_added,
                "methods_removed": event.methods_removed,
                "methods_modified": event.methods_modified,
            }
        elif isinstance(event, ImportAdded):
            return {
                "type": "ImportAdded",
                "module": event.module,
                "language": event.language,
                "line": event.line,
            }
        elif isinstance(event, ImportRemoved):
            return {
                "type": "ImportRemoved",
                "module": event.module,
                "language": event.language,
                "line": event.line,
            }
        return {}

    output = {
        "semantic_events": [event_to_dict(e) for e in semantic_events],
        "security_findings": [
            {
                "rule_id": f.rule_id,
                "title": f.title,
                "severity": f.severity.value,
                "file_path": f.file_path,
                "line": f.line,
            }
            for f in security_findings
        ],
        "duplication_findings": [
            {
                "source_file": f.source_file,
                "target_file": f.target_file,
                "similarity": f.similarity,
                "severity": f.severity,
            }
            for f in duplication_findings
        ],
    }

    console.print(json.dumps(output, indent=2))


if __name__ == "__main__":
    app()
