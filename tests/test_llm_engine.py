"""Tests for LLM Engine (Phase 4)."""
import os
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from pytest import MonkeyPatch

import httpx
from neurodiff.engines.llm_engine import (
    ContextBuilder,
    get_provider,
    GeminiProvider,
    ClaudeProvider,
    OllamaProvider,
    run_llm_analysis,
    LLMContext,
    get_cache_key,
    set_cached,
    CACHE_DIR
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: MonkeyPatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)


def test_provider_auto_detection(monkeypatch: MonkeyPatch):
    assert get_provider() is None

    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    p = get_provider()
    assert isinstance(p, OllamaProvider)

    monkeypatch.setenv("GOOGLE_API_KEY", "test-gemini")
    p = get_provider()
    assert isinstance(p, GeminiProvider)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-claude")
    p = get_provider()
    assert isinstance(p, ClaudeProvider)


def test_context_truncation():
    # Construct a massive payload
    security = [MagicMock(severity="low", category="test", file="f.py", line=i, description="x"*500) for i in range(100)]
    security += [MagicMock(severity="high", category="test", file="f.py", line=i, description="x"*500) for i in range(10)]
    
    ctx = ContextBuilder.build(
        events=[],
        security=security,
        duplication=[],
        arch_report=None,
        repo_stats={},
        diff_metadata={}
    )
    
    assert ctx.truncated is True
    # Verify low severity findings were dropped
    assert not any(f["severity"] == "low" for f in ctx.security_findings)
    # Verify high severity findings were kept
    assert any(f["severity"] == "high" for f in ctx.security_findings)


@pytest.mark.asyncio
async def test_concurrent_execution(monkeypatch: MonkeyPatch):
    class MockProvider:
        name = "mock"
        async def complete(self, system: str, user: str) -> str:
            if "Call 1" in user:
                return '{"safe_to_merge": true}'
            elif "Call 2" in user:
                return '{"architectural_smell": false}'
            else:
                return '{"suggested_split": false}'

    provider = MockProvider()
    
    ctx = LLMContext([], [], [], {}, {}, {})
    report = await run_llm_analysis(ctx, provider, use_cache=False)
    
    assert report.error is None
    assert report.executive_summary == {"safe_to_merge": True}
    assert report.pattern_analysis == {"architectural_smell": False}
    assert report.fix_plan == {"suggested_split": False}


@pytest.mark.asyncio
async def test_json_parsing_resilience(monkeypatch: MonkeyPatch):
    class MalformedProvider:
        name = "malformed"
        async def complete(self, system: str, user: str) -> str:
            return "```json\n{bad_json...}\n```"

    provider = MalformedProvider()
    ctx = LLMContext([], [], [], {}, {}, {})
    report = await run_llm_analysis(ctx, provider, use_cache=False)
    
    assert report.error is not None
    assert report.executive_summary.get("error") == "parse_failed"


@pytest.mark.asyncio
async def test_cache_hit(monkeypatch: MonkeyPatch):
    ctx = LLMContext([], [], [], {}, {}, {})
    key = get_cache_key(ctx)
    
    class UncalledProvider:
        name = "uncalled"
        async def complete(self, system: str, user: str) -> str:
            raise AssertionError("Provider should not be called on cache hit")

    # Manually populate cache
    from neurodiff.engines.llm_engine import LLMReport
    cached_report = LLMReport(provider_used="cache", executive_summary={"cached": True})
    set_cached(key, cached_report)
    
    report = await run_llm_analysis(ctx, UncalledProvider(), use_cache=True)
    assert report.executive_summary == {"cached": True}
    assert report.provider_used == "cache"

