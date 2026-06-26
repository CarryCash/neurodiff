"""Phase 7 — Active Correction Engine.

Generates code corrections for HIGH/CRITICAL findings using the LLM.
Produces unified diff patches and can apply them to the repo.
"""
from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from neurodiff.engines.context_engine import GlobalContext
from neurodiff.engines.llm_engine import LLMProvider, parse_json_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class CorrectionRequest:
    finding_id: str
    finding_type: str              # "security", "duplication", "arch", "cognitive"
    severity: str
    file_path: str
    function_name: str | None
    original_code: str             # extracted from FileDiff content
    finding_description: str
    global_context: GlobalContext


@dataclass
class CodeCorrection:
    finding_id: str
    file_path: str
    original_code: str
    corrected_code: str
    explanation: str
    confidence: float
    patch: str
    breaking_changes: list[str]
    follow_up_tasks: list[str]


# ---------------------------------------------------------------------------
# Core Engine
# ---------------------------------------------------------------------------

class CorrectionEngine:
    """Generates and applies AI-powered code corrections."""

    SYSTEM_PROMPT = """\
You are NeuroDiff's active correction engine. You receive:
1. A specific code finding (security vulnerability, duplication, architecture violation, etc.)
2. The original code that has the issue
3. The full architectural context of the project (conventions, patterns, existing similar code)

Your job: produce a corrected version of the code that:
- Fixes the specific issue
- Follows the project's existing conventions (naming, docstring style, import style)
- Does NOT introduce new issues
- Is minimal — change only what's needed

Respond ONLY in this JSON format, no preamble, no markdown fences:
{
  "corrected_code": "...",
  "explanation": "...",
  "confidence": 0.0-1.0,
  "breaking_changes": ["..."],
  "follow_up_tasks": ["..."]
}"""

    def __init__(self, provider: LLMProvider | None = None):
        self.provider = provider
        # Semaphore to cap concurrent requests
        self._semaphore = asyncio.Semaphore(10)

    async def generate_all_corrections(
        self, requests: list[CorrectionRequest]
    ) -> list[CodeCorrection]:
        """Generate corrections for all requests concurrently."""
        if not self.provider or not requests:
            return []

        coros = [self._generate_correction(req) for req in requests]
        results = await asyncio.gather(*coros, return_exceptions=True)
        
        corrections = []
        for r in results:
            if isinstance(r, CodeCorrection):
                corrections.append(r)
            elif isinstance(r, Exception):
                logger.error(f"Failed to generate correction: {r}")
                
        return corrections

    async def _generate_correction(self, req: CorrectionRequest) -> CodeCorrection | None:
        async with self._semaphore:
            # Build prompt
            rag_examples = req.global_context.relevant_existing_code.get(req.finding_id, [])
            
            prompt_data = {
                "finding_id": req.finding_id,
                "finding_type": req.finding_type,
                "severity": req.severity,
                "file_path": req.file_path,
                "finding_description": req.finding_description,
                "original_code": req.original_code,
                "project_conventions": req.global_context.project_conventions,
                "architectural_summary": {
                    "framework": req.global_context.arch_map.detected_framework,
                    "patterns": req.global_context.arch_map.detected_patterns,
                },
                "rag_examples": rag_examples[:2]  # Top 2
            }
            
            prompt = json.dumps(prompt_data, indent=2)
            
            try:
                assert self.provider is not None
                response_text = await self.provider.complete(self.SYSTEM_PROMPT, prompt)
                parsed = parse_json_response(response_text)
                
                if "error" in parsed:
                    logger.warning(f"Failed to parse LLM output for {req.finding_id}")
                    return None
                    
                corrected_code = parsed.get("corrected_code", req.original_code)
                patch = generate_patch(req.original_code, corrected_code, req.file_path)
                
                return CodeCorrection(
                    finding_id=req.finding_id,
                    file_path=req.file_path,
                    original_code=req.original_code,
                    corrected_code=corrected_code,
                    explanation=parsed.get("explanation", ""),
                    confidence=float(parsed.get("confidence", 0.0)),
                    patch=patch,
                    breaking_changes=parsed.get("breaking_changes", []),
                    follow_up_tasks=parsed.get("follow_up_tasks", []),
                )
                
            except Exception as e:
                logger.error(f"LLM call failed for {req.finding_id}: {e}")
                return None


def generate_patch(original: str, corrected: str, file_path: str) -> str:
    """Generate a unified diff patch."""
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        corrected.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm=""
    )
    return "".join(diff)


def apply_correction(correction: CodeCorrection, repo_path: Path, dry_run: bool = True) -> bool:
    """
    Apply a patch using git apply.
    Returns True if applied cleanly.
    """
    patch_content = correction.patch
    if not patch_content.strip():
        return True  # No changes to apply
        
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".patch") as f:
        f.write(patch_content)
        patch_file = f.name

    try:
        cmd = ["git", "apply"]
        if dry_run:
            cmd.append("--check")
        cmd.append(patch_file)
        
        result = subprocess.run(
            cmd,
            cwd=str(repo_path),
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    finally:
        try:
            os.remove(patch_file)
        except OSError:
            pass
