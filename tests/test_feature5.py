"""Tests for Feature 5: Collaborative PR Review Engine."""
from __future__ import annotations

import os
from unittest.mock import patch, MagicMock
import pytest

from neurodiff.integrations.inline_commenter import (
    InlineComment,
    _map_line_to_position,
    detect_platform,
    render_inline_body,
    map_findings_to_comments
)
from neurodiff.integrations.conversation_handler import (
    PRComment,
    parse_webhook_payload,
    handle_ignore
)
from neurodiff.engines.security_engine import SecurityFinding


# ---------------------------------------------------------------------------
# Inline Commenter Tests
# ---------------------------------------------------------------------------

def test_map_line_to_position():
    raw_diff = (
        "@@ -10,5 +10,7 @@\n"
        " def some_func():\n"
        "-    pass\n"
        "+    do_something()\n"
        "+    return True\n"
        " \n"
    )
    # The first line (def some_func():) is position 1, line 10
    # The deleted line is position 2
    # The first added line (+ do_something()) is position 3, line 11
    # The second added line (+ return True) is position 4, line 12
    assert _map_line_to_position(raw_diff, 11) == 3
    assert _map_line_to_position(raw_diff, 12) == 4
    # Line 10 should be position 1, although normally it's context
    assert _map_line_to_position(raw_diff, 10) == 1
    # Line not in diff
    assert _map_line_to_position(raw_diff, 100) is None

def test_detect_platform():
    with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}, clear=True):
        assert detect_platform() == "github"
    with patch.dict(os.environ, {"GITLAB_CI": "true"}, clear=True):
        assert detect_platform() == "gitlab"
    with patch.dict(os.environ, {"BITBUCKET_BUILD_NUMBER": "123"}, clear=True):
        assert detect_platform() == "bitbucket"
    with patch.dict(os.environ, {}, clear=True):
        assert detect_platform() == "none"

def test_render_inline_body():
    c = InlineComment(
        file_path="app.py",
        line=10,
        side="RIGHT",
        severity="critical",
        engine="security",
        title="SQL Injection",
        body="Bad query",
        suggestion="cursor.execute('?')",
        finding_id="sec_123"
    )
    body = render_inline_body(c)
    assert "<!-- neurodiff-inline:sec_123 -->" in body
    assert "🔴 **[CRITICAL — Security]** SQL Injection" in body
    assert "Bad query" in body
    assert "```suggestion\ncursor.execute('?')\n```" in body

def test_map_findings_to_comments():
    security = [
        SecurityFinding(
            severity="critical",
            category="SQLi",
            file="app.py",
            line=45,
            function_name="login",
            description="desc",
            rule_id=None
        )
    ]
    
    comments = map_findings_to_comments(security, None, None)
    assert len(comments) == 1
    assert comments[0].file_path == "app.py"
    assert comments[0].line == 45
    assert comments[0].finding_id.startswith("sec_")
    
# ---------------------------------------------------------------------------
# Conversation Handler Tests
# ---------------------------------------------------------------------------

def test_parse_webhook_payload():
    payload = {
        "comment": {
            "id": 123,
            "body": "Hey @neurodiff fix this <!-- neurodiff-inline:sec_001 -->",
            "user": {"login": "dev1"}
        },
        "pull_request": {"number": 42},
        "repository": {"full_name": "org/repo"}
    }
    
    comment = parse_webhook_payload(payload)
    assert comment is not None
    assert comment.comment_id == 123
    assert comment.author == "dev1"
    assert comment.finding_id == "sec_001"
    assert comment.pr_number == 42
    assert comment.repo == "org/repo"

@pytest.mark.asyncio
async def test_handle_ignore():
    c = PRComment(
        comment_id=1,
        body="ignore",
        author="dev",
        file_path=None,
        line=None,
        finding_id="sec_001",
        repo="repo",
        pr_number=1
    )
    res = await handle_ignore(c)
    assert "sec_001" in res
    assert "ignored" in res.lower()
