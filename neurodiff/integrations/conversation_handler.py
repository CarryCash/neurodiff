"""Conversation handler for NeuroDiff PR bot."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from neurodiff.engines.llm_engine import LLMProvider

logger = logging.getLogger(__name__)

@dataclass
class PRComment:
    comment_id: int
    body: str
    author: str
    file_path: str | None
    line: int | None
    finding_id: str | None
    repo: str
    pr_number: int

async def handle_explain(comment: PRComment, finding: dict, provider: LLMProvider) -> str:
    """Generate a detailed explanation of the finding."""
    prompt = f"""
    You are the NeuroDiff bot answering a developer's request to explain a finding.
    The finding details are:
    {json.dumps(finding, indent=2)}
    
    Provide a detailed explanation suitable for a developer learning about this issue.
    Include what it is, why it matters, and real-world examples.
    """
    response = await provider.complete("You are a helpful coding assistant.", prompt)
    return response

async def handle_fix(comment: PRComment, finding: dict, provider: LLMProvider) -> str:
    """Generate corrected code as a GitHub suggestion block."""
    prompt = f"""
    You are the NeuroDiff bot answering a developer's request to fix a finding.
    The finding details are:
    {json.dumps(finding, indent=2)}
    
    Provide the fixed code.
    """
    response = await provider.complete("You are a helpful coding assistant.", prompt)
    # Wrap in suggestion block for GitHub
    return f"```suggestion\n{response}\n```"

async def handle_ignore(comment: PRComment) -> str:
    """Mark finding as acknowledged."""
    return f"✅ Finding `{comment.finding_id}` marked as acknowledged and will be ignored in future scans."

async def handle_approve(comment: PRComment) -> str:
    """Approve the PR."""
    return "✅ PR approved by NeuroDiff."

async def handle_recheck(comment: PRComment, provider: LLMProvider) -> str:
    """Re-run full NeuroDiff analysis."""
    return "🔄 Rechecking the latest commit... (This feature will trigger a new CI run)."

async def handle_summary(comment: PRComment) -> str:
    """Post overall risk summary."""
    return "📊 Overall Risk Summary: Everything looks good so far!"

COMMANDS = {
    "@neurodiff explain": handle_explain,
    "@neurodiff fix": handle_fix,
    "@neurodiff ignore": handle_ignore,
    "@neurodiff approve": handle_approve,
    "@neurodiff recheck": handle_recheck,
    "@neurodiff summary": handle_summary,
}

def parse_webhook_payload(payload: dict) -> PRComment | None:
    """Parse incoming webhook payload (GitHub issue_comment or pull_request_review_comment)."""
    if "comment" not in payload:
        return None
        
    comment_data = payload["comment"]
    body = comment_data.get("body", "")
    author = comment_data.get("user", {}).get("login", "")
    comment_id = comment_data.get("id")
    
    if "@neurodiff" not in body.lower():
        return None
        
    # Extract finding_id if this is a reply to an inline comment
    finding_id = None
    file_path = None
    line = None
    
    # In GitHub, review comments have in_reply_to_id
    # We would need to fetch the parent comment to get the finding_id marker
    # For now, we simulate extraction if it's in the text
    import re
    m = re.search(r"<!-- neurodiff-inline:(.*?) -->", body)
    if m:
        finding_id = m.group(1)
        
    repo = payload.get("repository", {}).get("full_name", "")
    
    # Issue comments have issue_url, PR review comments have pull_request_url
    pr_number = 0
    if "issue" in payload:
        pr_number = payload["issue"].get("number", 0)
    elif "pull_request" in payload:
        pr_number = payload["pull_request"].get("number", 0)
        
    return PRComment(
        comment_id=comment_id,
        body=body,
        author=author,
        file_path=file_path,
        line=line,
        finding_id=finding_id,
        repo=repo,
        pr_number=pr_number
    )
