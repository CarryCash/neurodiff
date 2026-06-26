import os
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neurodiff.engines.context_engine import ArchitecturalMap, ContextEngine, GlobalContext
from neurodiff.engines.correction_engine import CodeCorrection, apply_correction, generate_patch
from neurodiff.output.github_commenter import get_pr_number, post_pr_comment, render_github_markdown
from neurodiff.output.html_reporter import FullReport, generate_html


# ---------------------------------------------------------------------------
# Test Context Engine
# ---------------------------------------------------------------------------
def test_framework_detection_fastapi(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    main = repo / "main.py"
    main.write_text("from fastapi import FastAPI\n\napp = FastAPI()\n", encoding="utf-8")
    
    engine = ContextEngine(repo)
    arch_map, conv = engine._build_arch_map_and_conventions(None, [])
    
    assert arch_map.detected_framework == "fastapi"
    assert "main.py" in arch_map.entry_points[0]


def test_framework_detection_django(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    manage = repo / "manage.py"
    manage.write_text("import os\n\nos.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproj.settings')\n", encoding="utf-8")
    
    engine = ContextEngine(repo)
    arch_map, conv = engine._build_arch_map_and_conventions(None, [])
    
    assert arch_map.detected_framework == "django"


# ---------------------------------------------------------------------------
# Test Correction Engine
# ---------------------------------------------------------------------------
def test_generate_patch():
    original = "def foo():\n    return 1\n"
    corrected = "def foo():\n    return 2\n"
    patch_str = generate_patch(original, corrected, "test.py")
    
    assert "--- a/test.py" in patch_str
    assert "+++ b/test.py" in patch_str
    assert "-    return 1" in patch_str
    assert "+    return 2" in patch_str


@patch("subprocess.run")
def test_apply_correction_dry_run(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    
    correction = CodeCorrection(
        finding_id="sec_1", file_path="test.py", original_code="a", corrected_code="b",
        explanation="x", confidence=0.9, patch="some patch\n", breaking_changes=[], follow_up_tasks=[]
    )
    
    result = apply_correction(correction, Path("/fake/repo"), dry_run=True)
    
    assert result is True
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert "git" in args
    assert "apply" in args
    assert "--check" in args


# ---------------------------------------------------------------------------
# Test HTML Reporter
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_report():
    am = ArchitecturalMap("test_repo", "fastapi", ["layered"], {}, [], [], {}, 10, 2, 2.5)
    gc = GlobalContext(am)
    corr = CodeCorrection(
        "SEC-001", "auth.py", "x = 1\n", "x = 2\n", "Fixed x", 0.99, "patch_data", [], []
    )
    
    class DummyCF:
        total_score = 65
        grade = "D"
    
    class DummyAI:
        probability = 0.8
        
    class DummyBR:
        risk_score = "moderate"
        first_ring = []
        second_ring = []
        third_ring = []
    
    class DummyCog:
        fatigue_index = DummyCF()
        ai_generated = DummyAI()
        blast_radius = DummyBR()
        commit_patterns = None
        overall_verdict = "danger"
        
    return FullReport(
        metadata={"repo": "test", "base_ref": "HEAD~1", "head_ref": "HEAD", "duration_s": 1.2},
        semantic_events=[], security=[], duplication=[], arch=MagicMock(),
        cognitive=DummyCog(), llm=None, global_context=gc, corrections=[corr]
    )


def test_generate_html_contains_sections(mock_report):
    html = generate_html(mock_report)
    
    assert "NeuroDiff Report" in html
    assert "Overall Risk:" in html
    assert "Active Corrections" in html
    assert "Project Understanding" in html
    assert "SEC-001" in html
    assert "fastapi" in html


def test_generate_html_is_self_contained(mock_report):
    html = generate_html(mock_report)
    
    # We should not have external CDN links like fonts.googleapis or unpkg or cdnjs
    urls = re.findall(r'http[s]?://[^\s\"\'<>]+', html)
    for url in urls:
        # Only github link allowed
        assert "github.com/user/neurodiff" in url or "w3.org" in url


# ---------------------------------------------------------------------------
# Test GitHub Commenter
# ---------------------------------------------------------------------------
def test_get_pr_number_extraction():
    os.environ["GITHUB_REF"] = "refs/pull/42/merge"
    assert get_pr_number() == 42
    
    os.environ["GITHUB_REF"] = "refs/heads/main"
    assert get_pr_number() is None


def test_render_github_markdown(mock_report):
    md = render_github_markdown(mock_report)
    
    assert "## 🔍 NeuroDiff Analysis" in md
    assert "<!-- neurodiff-report -->" in md
    assert "Grade D" in md
    assert "Review 1 active corrections generated" in md


import sys

@pytest.mark.asyncio
async def test_post_pr_comment_no_token(mock_report):
    result = await post_pr_comment(mock_report, "", "owner/repo", 1)
    assert result is False


@pytest.mark.asyncio
async def test_post_pr_comment_updates_existing(mock_report):
    mock_client = AsyncMock()
    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.json.return_value = [
        {"body": "Some old comment"},
        {"body": "Old neurodiff <!-- neurodiff-report --> here", "url": "https://api/comment/123"}
    ]
    mock_client.get.return_value = mock_get_resp
    
    mock_patch_resp = MagicMock()
    mock_patch_resp.status_code = 200
    mock_client.patch.return_value = mock_patch_resp
    
    # Patch httpx where it is used or inject mock via sys.modules
    mock_httpx = MagicMock()
    mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client
    
    with patch.dict(sys.modules, {"httpx": mock_httpx}):
        result = await post_pr_comment(mock_report, "token", "owner/repo", 1)
    
    assert result is True
    mock_client.patch.assert_called_once()
    mock_client.post.assert_not_called()
    assert mock_client.patch.call_args[0][0] == "https://api/comment/123"
