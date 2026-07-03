import asyncio
import json
from dataclasses import dataclass
import pytest

from neurodiff.core.semantic_events import FunctionAdded
from neurodiff.engines.zeroday_engine import ZeroDayEngine, detect_signals

class MockProvider:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0
        self.name = "mock"
        
    async def complete(self, system: str, user: str) -> str:
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp

@dataclass
class MockFileDiff:
    path: str
    content_before: str
    content_after: str

def test_context_builder_api_endpoint():
    is_auth, is_api, is_mutation = detect_signals("get_users", "api/users.py", "@router.post('/users')\ndef get_users(): pass", [])
    assert is_api == True

def test_context_builder_auth_related():
    is_auth, is_api, is_mutation = detect_signals("update_password", "user.py", "def update_password(): pass", [])
    assert is_auth == True

def test_filter_function():
    is_auth, is_api, is_mutation = detect_signals("calculate_sum", "math.py", "def calculate_sum(a, b): return a + b", ["add"])
    assert not is_auth and not is_api and not is_mutation

@pytest.mark.asyncio
async def test_mock_llm_auth_gap():
    provider = MockProvider([json.dumps({
        "findings": [{
            "type": "auth_gap",
            "severity": "critical",
            "title": "Auth Gap",
            "description": "Missing invalidation",
            "attack_vector": "Stolen token",
            "missing_element": "invalidate()",
            "suggested_fix": "Call invalidate()",
            "confidence": 0.9
        }]
    })])
    # pyrefly: ignore [bad-argument-type]
    engine = ZeroDayEngine(provider)
    
    # pyrefly: ignore [bad-argument-type]
    events = [FunctionAdded("update_password", "auth.py", 1, 10, 1, ["save"])]
    file_diffs = [MockFileDiff("auth.py", "", "def update_password():\n  save()")]
    
    # pyrefly: ignore [bad-argument-type]
    report = await engine.run(events, file_diffs, [], None)
    
    assert len(report.findings) == 1
    f = report.findings[0]
    assert f.type == "auth_gap"
    assert f.severity == "critical"
    assert report.critical_count == 1

@pytest.mark.asyncio
async def test_confidence_filter():
    provider = MockProvider([json.dumps({
        "findings": [{
            "type": "auth_gap",
            "severity": "critical",
            "title": "Auth Gap",
            "description": "Maybe an issue",
            "attack_vector": "Unknown",
            "missing_element": "Check",
            "suggested_fix": "Add check",
            "confidence": 0.5  # Below 0.7 threshold
        }]
    })])
    # pyrefly: ignore [bad-argument-type]
    engine = ZeroDayEngine(provider)
    
    # pyrefly: ignore [bad-argument-type]
    events = [FunctionAdded("update_password", "auth.py", 1, 10, 1, ["save"])]
    file_diffs = [MockFileDiff("auth.py", "", "def update_password():\n  save()")]
    
    # pyrefly: ignore [bad-argument-type]
    report = await engine.run(events, file_diffs, [], None)
    
    assert len(report.findings) == 0

@pytest.mark.asyncio
async def test_empty_findings():
    provider = MockProvider([json.dumps({"findings": []})])
    # pyrefly: ignore [bad-argument-type]
    engine = ZeroDayEngine(provider)
    
    # pyrefly: ignore [bad-argument-type]
    events = [FunctionAdded("update_password", "auth.py", 1, 10, 1, ["save"])]
    file_diffs = [MockFileDiff("auth.py", "", "def update_password():\n  save()")]
    
    # pyrefly: ignore [bad-argument-type]
    report = await engine.run(events, file_diffs, [], None)
    
    assert len(report.findings) == 0
    assert report.functions_analyzed == 1
