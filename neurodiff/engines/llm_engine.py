"""LLM Deep Analysis Engine for Phase 4."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class LLMContext:
    semantic_events: list[dict]
    security_findings: list[dict]
    duplication_findings: list[dict]
    arch_report: dict[str, Any]
    repo_stats: dict[str, Any]
    diff_metadata: dict[str, Any]
    cognitive_report: dict[str, Any] | None = None
    truncated: bool = False

@dataclass
class LLMReport:
    provider_used: str
    executive_summary: dict[str, Any] = field(default_factory=dict)
    pattern_analysis: dict[str, Any] = field(default_factory=dict)
    fix_plan: dict[str, Any] = field(default_factory=dict)
    total_tokens_used: int | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Context Builder & Truncation (8000 token limit ~ 32000 chars)
# ---------------------------------------------------------------------------

class ContextBuilder:
    MAX_CHARS = 32000  # rough heuristic for 8000 tokens

    @classmethod
    def build(
        cls,
        events: list,
        security: list,
        duplication: list,
        arch_report: Any,
        repo_stats: dict,
        diff_metadata: dict,
        cognitive_report: Any | None = None,
    ) -> LLMContext:
        """Build and optionally truncate the context payload."""
        from neurodiff.core.semantic_events import FunctionAdded, FunctionModified, ClassAdded, ClassModified, ImportAdded

        # Serialize events minimally
        serialized_events = []
        for e in events:
            if isinstance(e, FunctionAdded):
                serialized_events.append({"type": "FunctionAdded", "name": e.name, "file": e.file, "cc": e.cyclomatic_complexity, "calls": e.calls})
            elif isinstance(e, FunctionModified):
                serialized_events.append({"type": "FunctionModified", "name": e.name, "file": e.file, "cc_delta": e.complexity_after - e.complexity_before, "calls_added": e.calls_added})
            elif isinstance(e, ClassAdded):
                serialized_events.append({"type": "ClassAdded", "name": e.name, "file": e.file, "methods_count": len(e.methods)})
            elif isinstance(e, ClassModified):
                serialized_events.append({"type": "ClassModified", "name": e.name, "file": e.file, "methods_added": e.methods_added})
            elif isinstance(e, ImportAdded):
                serialized_events.append({"type": "ImportAdded", "module": e.module, "file": e.file})

        # Serialize security
        serialized_sec = [
            {"severity": f.severity, "category": f.category, "file": f.file, "line": f.line, "desc": f.description}
            for f in security
        ]

        # Serialize duplication
        serialized_dup = [
            {"func": f.new_function, "file": f.new_file, "similar_to": f.similar_function, "score": round(f.similarity_score, 2)}
            for f in duplication
        ]

        # Serialize arch report
        serialized_arch = {}
        if arch_report:
            serialized_arch = {
                "layer_violations": [{"severity": f.severity, "desc": f.description} for f in arch_report.layer_violations],
                "circular_deps": [{"severity": f.severity, "desc": f.description} for f in arch_report.circular_deps],
                "solid_findings": [{"severity": f.severity, "principle": f.principle, "desc": f.description} for f in arch_report.solid_findings],
                "blast_radius": arch_report.blast_radius,
            }

        # Serialize cognitive report
        serialized_cog = None
        if cognitive_report:
            from dataclasses import asdict
            try:
                serialized_cog = asdict(cognitive_report)
            except Exception:
                pass

        ctx = LLMContext(
            semantic_events=serialized_events,
            security_findings=serialized_sec,
            duplication_findings=serialized_dup,
            arch_report=serialized_arch,
            repo_stats=repo_stats,
            diff_metadata=diff_metadata,
            cognitive_report=serialized_cog,
        )

        return cls._truncate_if_needed(ctx)

    @classmethod
    def _truncate_if_needed(cls, ctx: LLMContext) -> LLMContext:
        payload_str = json.dumps(asdict(ctx))
        if len(payload_str) <= cls.MAX_CHARS:
            return ctx

        ctx.truncated = True

        # Drop LOW findings from security and arch
        ctx.security_findings = [f for f in ctx.security_findings if f["severity"] != "low"]
        if ctx.arch_report.get("solid_findings"):
            ctx.arch_report["solid_findings"] = [f for f in ctx.arch_report["solid_findings"] if f["severity"] != "low"]

        # If still too large, limit MEDIUM to top 5
        payload_str = json.dumps(asdict(ctx))
        if len(payload_str) > cls.MAX_CHARS:
            high_sec = [f for f in ctx.security_findings if f["severity"] in ("critical", "high")]
            med_sec = [f for f in ctx.security_findings if f["severity"] == "medium"][:5]
            ctx.security_findings = high_sec + med_sec

            if ctx.arch_report.get("solid_findings"):
                high_arch = [f for f in ctx.arch_report["solid_findings"] if f["severity"] in ("critical", "high")]
                med_arch = [f for f in ctx.arch_report["solid_findings"] if f["severity"] == "medium"][:5]
                ctx.arch_report["solid_findings"] = high_arch + med_arch

        return ctx


# ---------------------------------------------------------------------------
# Caching Layer
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".neurodiff" / "llm_cache"

def get_cache_key(context: LLMContext) -> str:
    payload = json.dumps(asdict(context), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

def get_cached(key: str) -> LLMReport | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            return LLMReport(**data)
        except Exception:
            pass
    return None

def set_cached(key: str, report: LLMReport) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / f"{key}.json"
        path.write_text(json.dumps(asdict(report)))
    except Exception as e:
        logger.warning(f"Failed to cache LLM report: {e}")


# ---------------------------------------------------------------------------
# LLM Providers
# ---------------------------------------------------------------------------

class LLMProvider(Protocol):
    name: str
    async def complete(self, system: str, user: str) -> str: ...

def parse_json_response(text: str) -> dict:
    """Strip markdown fences and parse JSON."""
    text = text.strip()
    # Remove markdown code blocks if present
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "parse_failed", "raw": text}


class ClaudeProvider:
    name = "claude"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-3-5-haiku-20241022"

    async def complete(self, system: str, user: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data = {
            "model": self.model,
            "max_tokens": 1500,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": 0.1,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(self.url, headers=headers, json=data)
            resp.raise_for_status()
            result = resp.json()
            return result["content"][0]["text"]


class GeminiProvider:
    name = "gemini"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"

    async def complete(self, system: str, user: str) -> str:
        headers = {"Content-Type": "application/json"}
        data = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(self.url, headers=headers, json=data)
            resp.raise_for_status()
            result = resp.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]


class OllamaProvider:
    name = "ollama"
    
    def __init__(self, host: str, model: str = "llama3"):
        self.host = host.rstrip("/")
        self.url = f"{self.host}/api/chat"
        self.model = model

    async def complete(self, system: str, user: str) -> str:
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(self.url, json=data)
            resp.raise_for_status()
            result = resp.json()
            return result["message"]["content"]


def get_provider(force_provider: str | None = None) -> LLMProvider | None:
    # First, load config file if exists
    config = {}
    config_path = Path.home() / ".neurodiff" / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    def get_key(env_name: str, config_name: str) -> str | None:
        return os.environ.get(env_name) or config.get(config_name)

    if force_provider == "claude" or (not force_provider and get_key("ANTHROPIC_API_KEY", "claude_key")):
        key = get_key("ANTHROPIC_API_KEY", "claude_key")
        if key: return ClaudeProvider(key)
    
    if force_provider == "gemini" or (not force_provider and get_key("GOOGLE_API_KEY", "gemini_key")):
        key = get_key("GOOGLE_API_KEY", "gemini_key")
        if key: return GeminiProvider(key)

    if force_provider == "ollama" or (not force_provider and get_key("OLLAMA_HOST", "ollama_key")):
        host = get_key("OLLAMA_HOST", "ollama_key") or "http://localhost:11434"
        model = get_key("OLLAMA_MODEL", "ollama_model") or "llama3"
        return OllamaProvider(host, model)

    return None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are NeuroDiff, an expert code review AI specializing in AI-generated code analysis.
You receive structured analysis data from static analysis engines — NOT raw code.
Your job is to synthesize this data into actionable insights a senior engineer would value.

Rules:
- Base all conclusions strictly on the provided data. Do not invent findings.
- Be direct and specific. No generic advice.
- If data shows no issues, say so clearly — do not fabricate concerns.
- Respond ONLY in the JSON format specified. No preamble, no markdown fences.\
"""

async def run_llm_analysis(
    context: LLMContext, provider: LLMProvider, use_cache: bool = True
) -> LLMReport:
    """Run the 3 concurrent LLM analysis calls."""
    cache_key = get_cache_key(context)
    if use_cache:
        cached = get_cached(cache_key)
        if cached:
            return cached

    ctx_json = json.dumps(asdict(context), indent=2)

    prompt1 = f"""\
Input data (Note: 'cognitive_report' contains AI generation probability and Cognitive Fatigue Index):
{ctx_json}

Call 1 — Executive Summary
Output schema:
{{
  "overall_risk": "critical|high|medium|low",
  "confidence": 0.0-1.0,
  "one_line_verdict": "string (max 120 chars)",
  "key_concerns": ["string", ...],
  "safe_to_merge": boolean,
  "merge_blockers": ["string", ...]
}}\
"""

    prompt2 = f"""\
Input data (focus on semantic_events and arch_report):
{json.dumps({'semantic_events': context.semantic_events, 'arch_report': context.arch_report}, indent=2)}

Call 2 — Deep Pattern Analysis
Output schema:
{{
  "patterns_detected": [
    {{
      "pattern": "string",
      "evidence": ["string", ...],
      "severity": "high|medium|low",
      "recommendation": "string"
    }}
  ],
  "architectural_smell": boolean,
  "estimated_tech_debt_hours": number | null
}}\
"""

    # Filter High/Critical for Fix Plan
    high_crit_sec = [f for f in context.security_findings if f["severity"] in ("high", "critical")]
    high_crit_arch = []
    if context.arch_report.get("layer_violations"):
        high_crit_arch.extend([f for f in context.arch_report["layer_violations"] if f["severity"] in ("high", "critical")])
    if context.arch_report.get("circular_deps"):
        high_crit_arch.extend([f for f in context.arch_report["circular_deps"] if f["severity"] in ("high", "critical")])

    prompt3 = f"""\
Input data (HIGH/CRITICAL issues only):
{json.dumps({'security': high_crit_sec, 'arch': high_crit_arch}, indent=2)}

Call 3 — Actionable Fix Plan
Output schema:
{{
  "immediate_actions": [
    {{
      "priority": 1,
      "action": "string",
      "file": "string",
      "rationale": "string"
    }}
  ],
  "before_merge_checklist": ["string", ...],
  "suggested_split": boolean,
  "split_rationale": "string | null"
}}\
"""

    async def call_api(prompt: str) -> dict:
        try:
            resp = await provider.complete(SYSTEM_PROMPT, prompt)
            return parse_json_response(resp)
        except Exception as e:
            return {"error": str(e), "raw": ""}

    results = await asyncio.gather(
        call_api(prompt1),
        call_api(prompt2),
        call_api(prompt3),
    )

    report = LLMReport(
        provider_used=provider.name,
        executive_summary=results[0],
        pattern_analysis=results[1],
        fix_plan=results[2],
    )

    if any("error" in r and "raw" in r for r in results):
        report.error = "One or more LLM calls failed to parse correctly or raised an exception."

    if use_cache and not report.error:
        set_cached(cache_key, report)

    return report
