"""
Lightweight data schemas for the evaluation harness.
Uses TypedDict for simplicity and broad Python compatibility.
"""
from typing import Any, Dict, List, Optional
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class ToolCall(TypedDict):
    tool_name: str
    arguments: Dict[str, Any]


class TraceStep(TypedDict):
    step_id: int
    action_type: str          # "tool_call" | "final_answer" | "error"
    tool_name: Optional[str]
    arguments: Optional[Dict[str, Any]]
    status: str               # "success" | "failure" | "skipped"
    observation_summary: Optional[str]
    latency_ms: int
    tokens: int


class AgentTrace(TypedDict):
    trace_id: str
    task: str
    steps: List[TraceStep]
    final_answer: str


class ComponentExample(TypedDict):
    id: str
    query: str
    expected_tool: str
    expected_arguments: Dict[str, Any]
    allowed_tools: List[str]
    failure_category: Optional[str]


class OutcomeExample(TypedDict):
    id: str
    task: str
    evidence: List[str]
    agent_answer: str
    reference_answer: str
    metadata: Dict[str, Any]


class AdversarialExample(TypedDict):
    id: str
    task: str
    tool_output: str
    attack_type: str
    expected_safe_behavior: str
    severity: str             # "low" | "medium" | "high" | "critical"
