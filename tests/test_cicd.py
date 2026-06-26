"""Tests for Phase 5 CI/CD Integration — SARIF, Markdown, GitHub client, exit codes."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_security_finding(severity: str = "critical") -> MagicMock:
    f = MagicMock()
    f.severity = severity
    f.category = "hardcoded_secret"
    f.file = "auth/config.py"
    f.line = 42
    f.description = "Hardcoded API key detected"
    return f


def _make_dup_finding(score: float = 0.92) -> MagicMock:
    f = MagicMock()
    f.similarity_score = score
    f.new_function = "processPayment"
    f.new_file = "utils/payments.py"
    f.similar_function = "handlePayment"
    f.similar_file = "core/billing.py"
    return f


def _make_arch_report(
    circular: bool = False, layer: bool = False
) -> MagicMock:
    report = MagicMock()
    report.layer_violations = (
        [MagicMock(severity="critical", description="core imports engines")] if layer else []
    )
    report.circular_deps = (
        [MagicMock(severity="critical", description="core→engines→core")] if circular else []
    )
    report.solid_findings = []
    report.blast_radius = {"total_affected": 5, "score": "high"}
    return report


def _make_llm_report(safe: bool = True) -> MagicMock:
    r = MagicMock()
    r.provider_used = "gemini"
    r.error = None
    r.executive_summary = {
        "safe_to_merge": safe,
        "one_line_verdict": "All good" if safe else "Circular dep found",
        "confidence": 0.92,
        "overall_risk": "low" if safe else "critical",
        "merge_blockers": [] if safe else ["Resolve circular dep"],
    }
    r.pattern_analysis = {"patterns_detected": [], "architectural_smell": False}
    r.fix_plan = {"immediate_actions": [], "before_merge_checklist": [], "suggested_split": False}
    return r


# ---------------------------------------------------------------------------
# SARIF Formatter Tests
# ---------------------------------------------------------------------------

class TestSarifFormatter:
    def test_empty_findings_produce_valid_sarif(self):
        from neurodiff.output.sarif_formatter import build_sarif
        sarif = build_sarif([], [], None)
        assert sarif["version"] == "2.1.0"
        assert len(sarif["runs"]) == 1
        assert sarif["runs"][0]["results"] == []

    def test_security_finding_appears_as_result(self):
        from neurodiff.output.sarif_formatter import build_sarif
        sarif = build_sarif([_make_security_finding()], [], None)
        results = sarif["runs"][0]["results"]
        assert len(results) == 1
        assert results[0]["level"] == "error"  # critical maps to error
        assert "ND-SEC-" in results[0]["ruleId"]

    def test_medium_security_maps_to_warning(self):
        from neurodiff.output.sarif_formatter import build_sarif
        sarif = build_sarif([_make_security_finding("medium")], [], None)
        results = sarif["runs"][0]["results"]
        assert results[0]["level"] == "warning"

    def test_duplication_finding_appears(self):
        from neurodiff.output.sarif_formatter import build_sarif
        sarif = build_sarif([], [_make_dup_finding()], None)
        results = sarif["runs"][0]["results"]
        assert len(results) == 1
        assert results[0]["ruleId"] == "ND-DUP-CODE-CLONE"

    def test_arch_circular_dep_appears(self):
        from neurodiff.output.sarif_formatter import build_sarif
        arch = _make_arch_report(circular=True)
        sarif = build_sarif([], [], arch)
        results = sarif["runs"][0]["results"]
        assert any(r["ruleId"] == "ND-ARCH-CIRCULAR-DEP" for r in results)

    def test_sarif_is_valid_json(self):
        from neurodiff.output.sarif_formatter import build_sarif
        sarif = build_sarif(
            [_make_security_finding()],
            [_make_dup_finding()],
            _make_arch_report(circular=True, layer=True),
        )
        # Must not raise
        serialized = json.dumps(sarif)
        reparsed = json.loads(serialized)
        assert reparsed["version"] == "2.1.0"

    def test_rules_deduplicated(self):
        """Two findings with the same category should produce only 1 rule entry."""
        from neurodiff.output.sarif_formatter import build_sarif
        sarif = build_sarif(
            [_make_security_finding("critical"), _make_security_finding("high")],
            [], None,
        )
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        rule_ids = [r["id"] for r in rules]
        assert len(rule_ids) == len(set(rule_ids))


# ---------------------------------------------------------------------------
# Markdown Formatter Tests
# ---------------------------------------------------------------------------

class TestMarkdownFormatter:
    def test_safe_verdict_shows_green_icon(self):
        from neurodiff.output.markdown_formatter import build_pr_comment
        md = build_pr_comment("main", "feature", 3, [], [], [], None, _make_llm_report(safe=True), True)
        assert "✅" in md
        assert "SAFE TO MERGE" in md

    def test_unsafe_verdict_shows_red_icon(self):
        from neurodiff.output.markdown_formatter import build_pr_comment
        md = build_pr_comment("main", "feature", 3, [], [], [], None, _make_llm_report(safe=False), False)
        assert "⛔" in md
        assert "NOT SAFE TO MERGE" in md

    def test_merge_blockers_shown(self):
        from neurodiff.output.markdown_formatter import build_pr_comment
        md = build_pr_comment("main", "feature", 3, [], [], [], None, _make_llm_report(safe=False), False)
        assert "Resolve circular dep" in md

    def test_critical_security_shown_as_table(self):
        from neurodiff.output.markdown_formatter import build_pr_comment
        md = build_pr_comment(
            "main", "feature", 3, [],
            [_make_security_finding("critical")],
            [], None, None, True,
        )
        assert "Hardcoded API key" in md
        assert "auth/config.py" in md

    def test_low_security_not_shown_in_table(self):
        from neurodiff.output.markdown_formatter import build_pr_comment
        md = build_pr_comment(
            "main", "feature", 3, [],
            [_make_security_finding("low")],
            [], None, None, True,
        )
        # Low findings should NOT appear in the PR comment table
        assert "Critical Security Findings" not in md

    def test_footer_always_present(self):
        from neurodiff.output.markdown_formatter import build_pr_comment
        md = build_pr_comment("main", "HEAD", 1, [], [], [], None, None, True)
        assert "NeuroDiff" in md
        assert "Generated by" in md


# ---------------------------------------------------------------------------
# GitHub Integration Tests
# ---------------------------------------------------------------------------

class TestGitHubIntegration:
    def test_pr_number_extracted_from_ref(self):
        from neurodiff.integrations.github import _get_pr_number
        assert _get_pr_number("refs/pull/42/merge") == 42
        assert _get_pr_number("refs/heads/main") is None
        assert _get_pr_number("refs/pull/100/head") == 100

    def test_client_built_from_env(self, monkeypatch):
        from neurodiff.integrations.github import get_github_client_from_env
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        client = get_github_client_from_env()
        assert client is not None
        assert client.repository == "owner/repo"

    def test_client_returns_none_without_token(self, monkeypatch):
        from neurodiff.integrations.github import get_github_client_from_env
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("NEURODIFF_GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        assert get_github_client_from_env() is None

    @pytest.mark.asyncio
    async def test_post_pr_comment_success(self):
        from neurodiff.integrations.github import GitHubClient
        client = GitHubClient(token="test", repository="owner/repo")

        mock_response = MagicMock()
        mock_response.status_code = 201

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_http

            result = await client.post_pr_comment(42, "## Test Comment")
            assert result is True

    @pytest.mark.asyncio
    async def test_post_pr_comment_handles_api_error(self):
        from neurodiff.integrations.github import GitHubClient
        client = GitHubClient(token="bad_token", repository="owner/repo")

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_http

            result = await client.post_pr_comment(42, "## Test")
            assert result is False


# ---------------------------------------------------------------------------
# Exit Code (_compute_is_safe) Tests
# ---------------------------------------------------------------------------

class TestExitCodes:
    def test_no_findings_is_safe(self):
        from neurodiff.cli.main import _compute_is_safe
        assert _compute_is_safe([], None, None) is True

    def test_critical_security_is_not_safe(self):
        from neurodiff.cli.main import _compute_is_safe
        assert _compute_is_safe([_make_security_finding("critical")], None, None) is False

    def test_high_security_safe_without_strict(self):
        from neurodiff.cli.main import _compute_is_safe
        assert _compute_is_safe([_make_security_finding("high")], None, None) is True

    def test_high_security_not_safe_with_strict(self):
        from neurodiff.cli.main import _compute_is_safe
        assert _compute_is_safe([_make_security_finding("high")], None, None, strict=True) is False

    def test_circular_dep_is_not_safe(self):
        from neurodiff.cli.main import _compute_is_safe
        arch = _make_arch_report(circular=True)
        assert _compute_is_safe([], arch, None) is False

    def test_llm_safe_verdict_overrides_static(self):
        """If LLM says safe, trust it even if static engine found something."""
        from neurodiff.cli.main import _compute_is_safe
        arch = _make_arch_report(circular=True)
        llm = _make_llm_report(safe=True)
        assert _compute_is_safe([], arch, llm) is True

    def test_llm_unsafe_verdict_fails(self):
        from neurodiff.cli.main import _compute_is_safe
        llm = _make_llm_report(safe=False)
        assert _compute_is_safe([], None, llm) is False

    def test_llm_error_falls_back_to_static(self):
        """If LLM errored, fall back to static analysis."""
        from neurodiff.cli.main import _compute_is_safe
        llm = MagicMock()
        llm.error = "Connection timeout"
        # Without critical static findings, should be safe
        assert _compute_is_safe([], None, llm) is True
        # With critical static finding, should be unsafe
        assert _compute_is_safe([_make_security_finding("critical")], None, llm) is False
