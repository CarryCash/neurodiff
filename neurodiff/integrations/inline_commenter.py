"""Inline PR commenter for NeuroDiff (GitHub, GitLab, Bitbucket)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Literal
import logging

import httpx

logger = logging.getLogger(__name__)

@dataclass
class InlineComment:
    file_path: str
    line: int
    side: Literal["RIGHT"]
    severity: str
    engine: str
    title: str
    body: str
    suggestion: str | None
    finding_id: str

def map_findings_to_comments(
    security_findings: list[Any],
    zeroday_report: Any | None,
    perf_report: Any | None,
) -> list[InlineComment]:
    """Map findings from all engines to inline comments."""
    comments = []
    
    # 1. Security Engine
    if security_findings:
        for f in security_findings:
            comments.append(InlineComment(
                file_path=f.file,
                line=f.line,
                side="RIGHT",
                severity=f.severity,
                engine="security",
                title=f.category,
                body=f.description,
                suggestion=None, # Security logic to be added if needed
                finding_id=f"sec_{id(f)}"
            ))
            
    # 2. ZeroDay Engine
    if zeroday_report and zeroday_report.findings:
        for f in zeroday_report.findings:
            comments.append(InlineComment(
                file_path=f.file,
                line=f.line,
                side="RIGHT",
                severity=f.severity,
                engine="zeroday",
                title=f.vulnerability_type,
                body=f.description,
                suggestion=None,
                finding_id=f"zd_{id(f)}"
            ))

    # 3. Perf Engine (Feature 4)
    if perf_report and perf_report.findings:
        for rw in perf_report.rewrites:
            f = rw.finding
            body = (f"💡 **Why slow:** {rw.why_slow}\n\n"
                    f"🚀 **Why fast:** {rw.why_fast}\n\n"
                    f"⏱️ **Estimated Speedup:** {rw.estimated_speedup}\n\n"
                    f"📐 **Big O Proof:** {rw.big_o_proof}")
            
            comments.append(InlineComment(
                file_path=f.file,
                line=f.line,
                side="RIGHT",
                severity=f.severity,
                engine="perf",
                title=f"Performance: {f.pattern_type} ({f.complexity_before} → {f.complexity_after})",
                body=body,
                suggestion=rw.rewritten_function,
                finding_id=f"perf_{id(f)}"
            ))
            
    return comments

def _map_line_to_position(raw_diff: str, target_line: int) -> int | None:
    """Map an absolute file line number to a GitHub diff 'position'.
    
    The position is 1-based starting from the line immediately following the first @@ hunk header.
    It counts every line in the patch (context, additions, deletions, and subsequent @@ headers).
    """
    if not raw_diff:
        return None
        
    lines = raw_diff.splitlines()
    position = 0
    current_line = 0
    in_hunk = False
    
    # Regex to match @@ -old_start,old_len +new_start,new_len @@
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    
    for i, line in enumerate(lines):
        m = hunk_re.match(line)
        if m:
            current_line = int(m.group(1))
            if not in_hunk:
                in_hunk = True
            else:
                position += 1
            continue
            
        if not in_hunk:
            continue
            
        # We are inside the diff hunks
        position += 1
        
        if line.startswith("+") or line.startswith(" "):
            if current_line == target_line:
                return position
            current_line += 1
            
    return None

def detect_platform() -> Literal["github", "gitlab", "bitbucket", "none"]:
    """Detect the current CI platform from environment variables."""
    if os.environ.get("GITHUB_ACTIONS"):
        return "github"
    if os.environ.get("GITLAB_CI"):
        return "gitlab"
    if os.environ.get("BITBUCKET_BUILD_NUMBER"):
        return "bitbucket"
    return "none"

def render_inline_body(comment: InlineComment) -> str:
    """Render the markdown body for an inline comment."""
    severity_label = f"[{comment.severity.upper()} — {comment.engine.capitalize()}]"
    if comment.severity == "critical":
        icon = "🔴"
    elif comment.severity == "high":
        icon = "🟠"
    elif comment.severity == "medium":
        icon = "🟡"
    else:
        icon = "🔵"
        
    body = f"<!-- neurodiff-inline:{comment.finding_id} -->\n"
    body += f"{icon} **{severity_label}** {comment.title}\n\n"
    body += comment.body + "\n\n"
    
    if comment.suggestion:
        body += "**Fix:**\n```suggestion\n"
        body += comment.suggestion
        body += "\n```\n\n"
        
    body += f"*Detected by NeuroDiff {comment.engine.capitalize()} Engine — [View full report](#)*"
    return body

def render_review_summary(verdict: str) -> str:
    """Render the top-level review summary."""
    if verdict == "REQUEST_CHANGES":
        return "🔴 NeuroDiff found critical issues that must be fixed before merging."
    return "✅ NeuroDiff found some issues, but nothing critical."

async def post_github_review(
    comments: list[InlineComment],
    repo: str,
    pr_number: int,
    token: str,
    overall_verdict: str,
    file_diffs: list[Any],
) -> bool:
    """Post an inline review to GitHub."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/reviews"
    
    # Needs head commit sha. We can fetch it or pass it. 
    # For now, let's assume we can fetch it via API or it's in the environment.
    # In GitHub Actions, github.event.pull_request.head.sha is available, or we can fetch PR details.
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "NeuroDiff/1.0"
    }
    
    async with httpx.AsyncClient() as client:
        # Get PR head SHA
        pr_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
        resp = await client.get(pr_url, headers=headers)
        resp.raise_for_status()
        head_sha = resp.json()["head"]["sha"]
        
        # Build comments
        gh_comments = []
        for c in comments:
            # Find the file diff for this comment to map the line
            fd = next((fd for fd in file_diffs if fd.path == c.file_path), None)
            if not fd:
                continue
                
            position = _map_line_to_position(fd.raw_diff, c.line)
            if position is None:
                continue
                
            gh_comments.append({
                "path": c.file_path,
                "position": position,
                "body": render_inline_body(c)
            })
            
        if not gh_comments:
            logger.info("No mappable comments to post on GitHub.")
            return True
            
        body = {
            "commit_id": head_sha,
            "body": render_review_summary(overall_verdict),
            "event": overall_verdict,
            "comments": gh_comments
        }
        
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        
    return True

async def post_gitlab_review(
    comments: list[InlineComment],
    project_id: str,
    mr_iid: str,
    token: str,
) -> bool:
    """Post an inline review to GitLab MR Discussions API."""
    base_url = os.environ.get("CI_SERVER_URL", "https://gitlab.com")
    url = f"{base_url}/api/v4/projects/{project_id}/merge_requests/{mr_iid}/discussions"
    
    headers = {
        "PRIVATE-TOKEN": token
    }
    
    base_sha = os.environ.get("CI_MERGE_REQUEST_DIFF_BASE_SHA")
    head_sha = os.environ.get("CI_COMMIT_SHA")
    if not base_sha or not head_sha:
        logger.warning("GitLab CI vars missing base_sha or head_sha.")
        return False
        
    async with httpx.AsyncClient() as client:
        for c in comments:
            position = {
                "base_sha": base_sha,
                "start_sha": base_sha,
                "head_sha": head_sha,
                "position_type": "text",
                "new_path": c.file_path,
                "new_line": c.line
            }
            body = {
                "body": render_inline_body(c),
                "position": position
            }
            resp = await client.post(url, headers=headers, json=body)
            if resp.status_code not in (200, 201):
                logger.warning("GitLab API error: %s", resp.text)
                
    return True

async def post_bitbucket_review(
    comments: list[InlineComment],
    workspace: str,
    repo_slug: str,
    pr_id: str,
    token: str,
) -> bool:
    """Post an inline review to Bitbucket PR comments API."""
    url = f"https://api.bitbucket.org/2.0/repositories/{workspace}/{repo_slug}/pullrequests/{pr_id}/comments"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        for c in comments:
            body = {
                "content": {"raw": render_inline_body(c)},
                "inline": {
                    "to": c.line,
                    "path": c.file_path
                }
            }
            resp = await client.post(url, headers=headers, json=body)
            if resp.status_code not in (200, 201):
                logger.warning("Bitbucket API error: %s", resp.text)
                
    return True
