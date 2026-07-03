import asyncio
import re
from dataclasses import dataclass
from typing import Literal

from neurodiff.core.semantic_events import SemanticEvent, FunctionAdded, FunctionModified
from neurodiff.engines.llm_engine import LLMProvider, parse_json_response

@dataclass
class FunctionContext:
    function_name: str
    file_path: str
    layer: str
    original_code: str
    new_code: str
    calls_added: list[str]
    calls_removed: list[str]
    imports_added: list[str]
    related_functions: list[str]
    is_auth_related: bool
    is_api_endpoint: bool
    is_data_mutation: bool

@dataclass
class PriorSecurityContext:
    known_findings: list[str]
    arch_layer_violations: list[str]
    blast_radius_hotspots: list[str]

@dataclass
class ZeroDayFinding:
    function_name: str
    file_path: str
    layer: str
    type: Literal["auth_gap", "state_inconsistency", "missing_side_effect", "exposed_internal", "trust_violation"]
    severity: Literal["critical", "high", "medium"]
    title: str
    description: str
    attack_vector: str
    missing_element: str
    suggested_fix: str
    confidence: float

@dataclass
class ZeroDayReport:
    findings: list[ZeroDayFinding]
    functions_analyzed: int
    functions_skipped: int
    provider_used: str
    error: str | None

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "high")

SYSTEM_PROMPT = """\
You are NeuroDiff's logic vulnerability detector.
You analyze code changes for business logic vulnerabilities — flaws that static analyzers
miss because they require understanding intent, flow, and security context.

Focus ONLY on:
1. Authentication/Authorization gaps (missing checks, privilege escalation paths)
2. State inconsistency (updating X but not updating Y that depends on X)
3. Missing security side-effects (changing password without invalidating sessions)
4. Exposed internal logic (public endpoint calling admin/internal functions)
5. Trust boundary violations (using user input to control privileged operations)

Rules:
- Only report vulnerabilities you are confident about (confidence > 0.7)
- Be specific: name the exact missing function call, missing check, or unsafe flow
- Do NOT repeat findings already covered by static analysis (provided below)
- If no logic vulnerability exists, return an empty findings array — do not fabricate
- Respond ONLY in the JSON format below. No preamble, no markdown fences.
"""

def detect_signals(func_name: str, file_path: str, code: str, calls: list[str]) -> tuple[bool, bool, bool]:
    auth_keywords = {"auth", "login", "token", "session", "password", "role", "permission", "user"}
    
    is_auth = any(k in func_name.lower() or k in file_path.lower() for k in auth_keywords)
    
    api_decorators = ["@app.route", "@router.get", "@router.post", "@router.put", "@router.delete", "@router.patch", "@app.post", "@app.get"]
    is_api = any(dec in code for dec in api_decorators)
    
    mutation_keywords = {"save", "update", "delete", "create", "insert", "commit"}
    is_mutation = any(any(k in call.lower() for k in mutation_keywords) for call in calls)
    
    return is_auth, is_api, is_mutation

class ZeroDayEngine:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
    
    # pyrefly: ignore [not-a-type]
    def build_contexts(self, events: list[SemanticEvent], file_diffs: list, arch_report: any) -> list[FunctionContext]:
        contexts = []
        
        # Build layer map
        layer_map = {}
        if arch_report and hasattr(arch_report, "layer_config"):
            for layer in arch_report.layer_config.get("layers", []):
                name = layer.get("name")
                for pattern in layer.get("patterns", []):
                    layer_map[pattern] = name
                    
        def get_layer(path: str) -> str:
            # Simple heuristic
            for pattern, name in layer_map.items():
                if pattern.replace("*", "") in path:
                    return name
            if "api" in path or "views" in path or "router" in path: return "api"
            if "service" in path: return "service"
            if "db" in path or "repo" in path or "model" in path: return "repository"
            if "auth" in path: return "auth"
            return "util"

        diff_map = {fd.path: fd for fd in file_diffs}
        
        for e in events:
            if isinstance(e, (FunctionAdded, FunctionModified)):
                fd = diff_map.get(e.file)
                if not fd: continue
                
                original_code = ""
                if isinstance(e, FunctionModified):
                    # We might not have perfect bounds for original function, but we can pass the whole old file or diff
                    original_code = fd.content_before
                    new_code = fd.content_after
                    calls_added = e.calls_added
                    calls_removed = [] # Assuming not easily available
                    imports_added = [] # Would need to check ImportAdded in same file
                else:
                    new_code = fd.content_after
                    calls_added = e.calls
                    calls_removed = []
                    imports_added = []
                
                is_auth, is_api, is_mutation = detect_signals(e.name, e.file, new_code, calls_added)
                
                if is_auth or is_api or is_mutation:
                    contexts.append(FunctionContext(
                        function_name=e.name,
                        file_path=e.file,
                        layer=get_layer(e.file),
                        original_code=original_code,
                        new_code=new_code,
                        calls_added=calls_added,
                        calls_removed=calls_removed,
                        imports_added=imports_added,
                        related_functions=[], # Skipping AST walk for brevity unless needed
                        is_auth_related=is_auth,
                        is_api_endpoint=is_api,
                        is_data_mutation=is_mutation
                    ))
        return contexts

    async def _analyze_function(self, ctx: FunctionContext, prior_ctx: PriorSecurityContext) -> list[ZeroDayFinding]:
        user_prompt = f"""\
FUNCTION CONTEXT:
Name: {ctx.function_name}
File: {ctx.file_path}
Layer: {ctx.layer}
Is auth-related: {ctx.is_auth_related}
Is API endpoint: {ctx.is_api_endpoint}
Is data mutation: {ctx.is_data_mutation}

NEW CODE:
{ctx.new_code}

CALLS ADDED: {ctx.calls_added}
IMPORTS ADDED: {ctx.imports_added}
RELATED FUNCTIONS IN FILE: {ctx.related_functions}

ALREADY KNOWN BY STATIC ANALYSIS (do not repeat):
{prior_ctx.known_findings}

Analyze for logic vulnerabilities only.
"""
        try:
            resp = await self.provider.complete(SYSTEM_PROMPT, user_prompt)
            data = parse_json_response(resp)
            if "error" in data: return []
            
            findings = []
            for item in data.get("findings", []):
                if item.get("confidence", 0.0) > 0.7:
                    findings.append(ZeroDayFinding(
                        function_name=ctx.function_name,
                        file_path=ctx.file_path,
                        layer=ctx.layer,
                        type=item.get("type", "state_inconsistency"),
                        severity=item.get("severity", "medium"),
                        title=item.get("title", "Unknown issue"),
                        description=item.get("description", ""),
                        attack_vector=item.get("attack_vector", ""),
                        missing_element=item.get("missing_element", ""),
                        suggested_fix=item.get("suggested_fix", ""),
                        confidence=item.get("confidence", 0.0)
                    ))
            return findings
        except Exception:
            return []

    async def run(self, events: list[SemanticEvent], file_diffs: list, prior_sec: list, arch_report) -> ZeroDayReport:
        contexts = self.build_contexts(events, file_diffs, arch_report)
        
        prior_ctx = PriorSecurityContext(
            known_findings=[f.description for f in prior_sec],
            arch_layer_violations=[f.description for f in getattr(arch_report, "layer_violations", [])],
            # pyrefly: ignore [bad-argument-type, not-a-type]
            blast_radius_hotspots=[getattr(arch_report, "blast_radius", {}).get("hotspot")] if getattr(arch_report, "blast_radius", None) else []
        )
        
        all_findings = []
        error = None
        
        # Run concurrently with a semaphore of 8
        semaphore = asyncio.Semaphore(8)
        
        async def sem_analyze(c):
            async with semaphore:
                return await self._analyze_function(c, prior_ctx)
                
        try:
            results = await asyncio.gather(*(sem_analyze(c) for c in contexts))
            for res in results:
                all_findings.extend(res)
        except Exception as e:
            error = str(e)
            
        return ZeroDayReport(
            findings=all_findings,
            functions_analyzed=len(contexts),
            functions_skipped=len([e for e in events if isinstance(e, (FunctionAdded, FunctionModified))]) - len(contexts),
            provider_used=self.provider.name,
            error=error
        )
