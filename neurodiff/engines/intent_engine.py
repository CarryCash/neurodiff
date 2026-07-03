import os
import subprocess
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from neurodiff.core.semantic_events import (
    SemanticEvent, FunctionAdded, FunctionRemoved, FunctionModified,
    # pyrefly: ignore [missing-module-attribute]
    ClassAdded, ClassRemoved, ClassModified, ImportAdded, ImportRemoved
)
from neurodiff.engines.llm_engine import get_provider, LLMProvider, parse_json_response

@dataclass
class CommitIntent:
    commit_message: str
    commit_title: str           
    pr_title: str | None        
    pr_description: str | None  
    raw_combined: str           

@dataclass
class IntentReport:
    intent: CommitIntent
    semantic_summary: str
    verdict: Literal["match", "partial_match", "mismatch", "suspicious"]
    confidence: float
    alignment_score: int             
    intent_keywords: list[str]
    matched_changes: list[dict]
    unmatched_changes: list[dict]
    missing_changes: list[dict]
    verdict_explanation: str
    recommendation: Literal["approve", "request_changes", "block"]
    provider_used: str
    error: str | None

def extract_intent(repo_path: Path, head_ref: str) -> CommitIntent:
    """Extract commit message and PR context."""
    commit_msg = ""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=%B", head_ref],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        commit_msg = result.stdout.strip()
    except subprocess.CalledProcessError:
        commit_msg = "No commit message found."

    commit_title = commit_msg.split('\n')[0] if commit_msg else ""
    
    pr_title = os.environ.get("GITHUB_PR_TITLE")
    pr_description = os.environ.get("GITHUB_PR_BODY")
    
    combined_parts = [
        f"Commit Message:\n{commit_msg}",
    ]
    if pr_title:
        combined_parts.append(f"PR Title:\n{pr_title}")
    if pr_description:
        combined_parts.append(f"PR Description:\n{pr_description}")
        
    raw_combined = "\n\n".join(combined_parts)
    
    return CommitIntent(
        commit_message=commit_msg,
        commit_title=commit_title,
        pr_title=pr_title,
        pr_description=pr_description,
        raw_combined=raw_combined
    )

def build_semantic_summary(events: list[SemanticEvent]) -> str:
    """Convert SemanticEvents into a readable summary for the LLM."""
    lines = []
    for e in events:
        if isinstance(e, FunctionAdded):
            lines.append(f"- ADDED function: {e.name}() in {e.file} (complexity: {e.cyclomatic_complexity})")
        elif isinstance(e, FunctionRemoved):
            lines.append(f"- REMOVED function: {e.name}() in {e.file}")
        elif isinstance(e, FunctionModified):
            lines.append(f"- MODIFIED function: {e.name}() in {e.file} (signature changed, {len(e.calls_added)} new calls)")
        elif isinstance(e, ClassAdded):
            lines.append(f"- ADDED class: {e.name} in {e.file} ({len(e.methods)} methods)")
        elif isinstance(e, ClassRemoved):
            lines.append(f"- REMOVED class: {e.name} in {e.file}")
        elif isinstance(e, ClassModified):
            lines.append(f"- MODIFIED class: {e.name} in {e.file} ({len(e.methods_added)} methods added)")
        elif isinstance(e, ImportAdded):
            lines.append(f"- ADDED import: {e.module} in {e.file}")
        elif isinstance(e, ImportRemoved):
            lines.append(f"- REMOVED import: {e.module} in {e.file}")
    
    if not lines:
        return "- No semantic changes detected."
    return "\n".join(lines)

SYSTEM_PROMPT = """\
You are NeuroDiff's intent verification engine.
You receive:
1. The stated intent of a commit/PR (what the developer SAID they did)
2. A structured summary of what ACTUALLY changed in the code (AST-level facts)

Your job: determine if the actual changes match the stated intent.

Rules:
- Be specific. Point to exact function names and files from the semantic summary.
- A commit doing MORE than stated is a warning, not necessarily a blocker.
- A commit doing LESS than stated is always a concern.
- A commit doing DIFFERENT things than stated is always a blocker.
- Ignore style/formatting changes — focus on logic and structure.
- Respond ONLY in the JSON format below. No preamble, no markdown fences.
"""

async def run_intent_analysis(
    intent: CommitIntent,
    semantic_summary: str,
    provider: LLMProvider
) -> IntentReport:
    """Run intent vs reality analysis via LLM."""
    
    user_prompt = f"""\
STATED INTENT:
{intent.raw_combined}

ACTUAL CHANGES (AST analysis):
{semantic_summary}

Analyze the match between intent and reality.
Output schema:
{{
  "verdict": "match" | "partial_match" | "mismatch" | "suspicious",
  "confidence": 0.0-1.0,
  "alignment_score": 0-100,
  "intent_keywords": ["str"],
  "matched_changes": [
    {{
      "change": "str",
      "supports_intent": "str",
      "confidence": 0.0-1.0
    }}
  ],
  "unmatched_changes": [
    {{
      "change": "str",
      "concern": "str",
      "severity": "high" | "medium" | "low"
    }}
  ],
  "missing_changes": [
    {{
      "expected": "str",
      "reason": "str"
    }}
  ],
  "verdict_explanation": "str",
  "recommendation": "approve" | "request_changes" | "block"
}}
"""
    try:
        resp = await provider.complete(SYSTEM_PROMPT, user_prompt)
        data = parse_json_response(resp)
        
        if "error" in data:
            return IntentReport(
                intent=intent,
                semantic_summary=semantic_summary,
                verdict="suspicious",
                confidence=0.0,
                alignment_score=0,
                intent_keywords=[],
                matched_changes=[],
                unmatched_changes=[],
                missing_changes=[],
                verdict_explanation="Failed to parse LLM response.",
                recommendation="block",
                provider_used=provider.name,
                error=data.get("error")
            )

        return IntentReport(
            intent=intent,
            semantic_summary=semantic_summary,
            verdict=data.get("verdict", "suspicious"),
            confidence=float(data.get("confidence", 0.0)),
            alignment_score=int(data.get("alignment_score", 0)),
            intent_keywords=data.get("intent_keywords", []),
            matched_changes=data.get("matched_changes", []),
            unmatched_changes=data.get("unmatched_changes", []),
            missing_changes=data.get("missing_changes", []),
            verdict_explanation=data.get("verdict_explanation", ""),
            recommendation=data.get("recommendation", "block"),
            provider_used=provider.name,
            error=None
        )
    except Exception as e:
        return IntentReport(
            intent=intent,
            semantic_summary=semantic_summary,
            verdict="suspicious",
            confidence=0.0,
            alignment_score=0,
            intent_keywords=[],
            matched_changes=[],
            unmatched_changes=[],
            missing_changes=[],
            verdict_explanation=f"LLM API call failed: {e}",
            recommendation="block",
            provider_used=provider.name,
            error=str(e)
        )
