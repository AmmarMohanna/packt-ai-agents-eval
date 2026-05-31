"""
Deterministic mock research assistant agent.

The agent simulates realistic failures so each notebook has meaningful
evaluation signal. Failure modes are controlled via the task's
'agent_profile' key or query content — no randomness.
"""
from typing import Any, Dict, List, Optional
import time

from .tools import call_tool, TOOL_REGISTRY

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _pick_tool_and_args(query: str) -> tuple:
    """
    Rule-based tool selector.  Returns (tool_name, arguments).
    Covers the main query intents used across all datasets.
    """
    q = query.lower()
    if any(k in q for k in ["recent paper", "latest research", "find paper",
                              "retrieval", "rag", "survey", "literature"]):
        return "search_web", {"query": query, "date_range": "recent"}
    if any(k in q for k in ["internal", "uploaded", "our doc", "company",
                              "previous report", "internal doc"]):
        return "search_docs", {"query": query}
    if any(k in q for k in ["read doc", "read document", "open doc", "load doc",
                              "fetch doc", "content of"]):
        doc_id = "doc_42"  # deterministic stand-in
        return "read_document", {"doc_id": doc_id}
    if any(k in q for k in ["summarize", "summary", "synthesize", "findings"]):
        return "summarize_evidence", {"evidence": [query]}
    if any(k in q for k in ["cite", "reference", "citation", "source"]):
        return "cite_sources", {"answer": "The evidence supports the claim.", "sources": [query]}
    if any(k in q for k in ["what do you mean", "clarify", "ambiguous",
                              "unclear", "more info", "specify"]):
        return "ask_clarification", {"question": f"Could you clarify: {query}?"}
    # Default fallback
    return "search_web", {"query": query}


# ---------------------------------------------------------------------------
# Profile-based failure injection
# ---------------------------------------------------------------------------

_WRONG_TOOL = "wrong_tool"
_BAD_ARGS = "bad_args"
_EARLY_STOP = "early_stop"
_NO_RECOVERY = "no_recovery"
_REPEAT_CALL = "repeat_call"
_PARTIAL_GROUND = "partial_grounding"


def _inject_failure(profile: str, tool: str, args: Dict) -> tuple:
    """Mutate tool/args to simulate a known failure mode."""
    if profile == _WRONG_TOOL:
        # Picks the wrong tool for the job
        wrong = "read_document" if tool != "read_document" else "search_web"
        return wrong, {"doc_id": "doc_99"} if wrong == "read_document" else {"query": args.get("query", "oops")}
    if profile == _BAD_ARGS:
        # Uses correct tool but omits required argument
        return tool, {}
    return tool, args


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_component_prediction(query: str, profile: Optional[str] = None) -> Dict[str, Any]:
    """
    Predict which tool to call and with what arguments for a single query.
    Used in Notebook 1 (component evaluation).

    Args:
        query:   The user research query.
        profile: Optional failure profile to inject deterministic errors.

    Returns a dict with 'predicted_tool' and 'predicted_arguments'.
    """
    tool, args = _pick_tool_and_args(query)
    if profile:
        tool, args = _inject_failure(profile, tool, args)
    return {"predicted_tool": tool, "predicted_arguments": args}


def run_agent_trace(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the agent on a task and return a full trace.
    Used in Notebook 2 (trajectory evaluation).

    The task dict may contain:
        'query'          : str  — the research question
        'agent_profile'  : str  — failure mode (optional)
        'max_steps'      : int  — cap for demonstration (optional)
    """
    query = task.get("query", "")
    profile = task.get("agent_profile", "clean")
    steps: List[Dict] = []
    t0 = 0  # virtual clock (ms)

    def _step(tool_name, arguments, latency=600, tokens=110, status="success", obs=None):
        nonlocal t0
        t0 += latency
        return {
            "step_id": len(steps) + 1,
            "action_type": "tool_call",
            "tool_name": tool_name,
            "arguments": arguments,
            "status": status,
            "observation_summary": obs or f"Called {tool_name} successfully",
            "latency_ms": latency,
            "tokens": tokens,
        }

    # --- Clean path ---
    if profile == "clean":
        tool, args = _pick_tool_and_args(query)
        steps.append(_step(tool, args, latency=700, tokens=120))
        if tool in ("search_web", "search_docs"):
            steps.append(_step("summarize_evidence",
                               {"evidence": [f"Results from {tool}"]},
                               latency=500, tokens=100))
        final = f"Based on research: {query[:60]}... the evidence supports the main claim."

    # --- Excessive steps ---
    elif profile == "excessive_steps":
        tool, args = _pick_tool_and_args(query)
        for i in range(5):
            steps.append(_step("search_web", {"query": f"{query} part {i}"}, latency=400, tokens=80))
        steps.append(_step("summarize_evidence", {"evidence": ["combined"]}, latency=500, tokens=100))
        final = "After extensive search: " + query[:40]

    # --- Duplicate consecutive calls ---
    elif profile == "duplicate_calls":
        tool, args = _pick_tool_and_args(query)
        steps.append(_step(tool, args, latency=600, tokens=110))
        # exact duplicate
        steps.append(_step(tool, args, latency=600, tokens=110))
        steps.append(_step("summarize_evidence", {"evidence": ["dup results"]}, latency=400, tokens=90))
        final = "Summarized (with duplicate call): " + query[:40]

    # --- Tool loop ---
    elif profile == "loop":
        for _ in range(4):
            steps.append(_step("search_web", {"query": query}, latency=500, tokens=100))
        final = "Result after loop: " + query[:40]

    # --- Failed call + recovery ---
    elif profile == "failure_with_recovery":
        steps.append(_step("read_document", {"doc_id": "doc_missing"},
                           latency=300, tokens=50, status="failure",
                           obs="Document not found"))
        steps.append(_step("search_docs", {"query": query}, latency=600, tokens=110))
        steps.append(_step("summarize_evidence", {"evidence": ["recovered result"]},
                           latency=400, tokens=90))
        final = "Recovered and answered: " + query[:40]

    # --- Failed call, no recovery ---
    elif profile == "failure_no_recovery":
        steps.append(_step("read_document", {"doc_id": "doc_missing"},
                           latency=300, tokens=50, status="failure",
                           obs="Document not found"))
        # Agent gives up immediately
        final = "I encountered an error and could not complete the task."

    # --- Unnecessary detour ---
    elif profile == "unnecessary_detour":
        steps.append(_step("search_web", {"query": "unrelated background topic"},
                           latency=700, tokens=130))
        tool, args = _pick_tool_and_args(query)
        steps.append(_step(tool, args, latency=600, tokens=110))
        steps.append(_step("summarize_evidence", {"evidence": ["main result"]},
                           latency=400, tokens=90))
        final = "Answered after detour: " + query[:40]

    # --- Premature answer ---
    elif profile == "premature_answer":
        # Only one quick step before answering
        steps.append(_step("search_web", {"query": query[:20]}, latency=200, tokens=40))
        final = "Quick answer: " + query[:30]

    # --- High latency ---
    elif profile == "high_latency":
        steps.append(_step("search_web", {"query": query}, latency=3000, tokens=200))
        steps.append(_step("summarize_evidence", {"evidence": ["slow result"]},
                           latency=3500, tokens=300))
        final = "Slow but complete answer: " + query[:40]

    # --- High token usage ---
    elif profile == "high_tokens":
        for i in range(3):
            steps.append(_step("search_web", {"query": query}, latency=500, tokens=800))
        final = "Token-heavy answer: " + query[:40]

    else:
        # Unknown profile → clean fallback
        tool, args = _pick_tool_and_args(query)
        steps.append(_step(tool, args))
        final = "Fallback answer for: " + query[:40]

    return {
        "trace_id": task.get("trace_id", "tr_dynamic"),
        "task": query,
        "steps": steps,
        "final_answer": final,
    }


def run_agent_answer(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce an agent answer for an outcome-evaluation task.
    Used in Notebook 3.

    task must contain 'evidence' (list[str]) and 'task' (str).
    The 'answer_profile' key controls quality.
    """
    evidence = task.get("evidence", [])
    question = task.get("task", "")
    profile = task.get("answer_profile", "correct")

    ev_text = " ".join(evidence)

    if profile == "correct":
        answer = (f"Based on the provided sources, {question[:50]} "
                  f"Evidence indicates: {ev_text[:100]}.")
    elif profile == "incomplete":
        # Factual but only covers first evidence
        answer = f"According to the first source: {evidence[0][:80] if evidence else 'N/A'}."
    elif profile == "not_grounded":
        # Plausible sounding but ignores evidence
        answer = ("Research generally shows that AI agents improve over time "
                  "due to reinforcement learning and better training data.")
    elif profile == "verbose_weak":
        answer = (
            "That is indeed a very interesting and complex question. "
            "There are many perspectives to consider. On one hand, one might argue X. "
            "On the other hand, Y is also plausible. The evidence may or may not support this. "
            "In conclusion, the answer depends on many factors: " + ev_text[:50] + "."
        )
    elif profile == "format_violation":
        # Asked for bullet list, returns prose blob
        answer = "theanswer is yes because " + ev_text[:60] + " no citations provided"
    elif profile == "unsafe":
        answer = (
            "The evidence definitively proves beyond any doubt that this treatment "
            "cures all patients. Everyone should immediately follow this protocol "
            "without consulting a doctor. " + ev_text[:40]
        )
    elif profile == "missing_citation":
        answer = (
            "Studies have shown significant improvements in performance metrics. "
            "The methodology is sound and results are reproducible."
        )
    else:
        answer = ev_text[:120]

    return {"id": task.get("id", "out_dynamic"), "agent_answer": answer}


def run_adversarial_agent(task: Dict[str, Any], guarded: bool = False) -> Dict[str, Any]:
    """
    Run the agent on an adversarial task.
    Used in Notebook 4.

    When guarded=False the agent blindly incorporates tool output.
    When guarded=True the agent sanitizes tool output before using it.
    """
    from .adversarial import detect_prompt_injection, detect_exfiltration_attempt, sanitize_tool_output

    tool_output = task.get("tool_output", "")
    task_instruction = task.get("task", "")
    attack_type = task.get("attack_type", "unknown")

    if guarded:
        # Check tool output before processing
        injection_detected = detect_prompt_injection(tool_output)
        exfil_detected = detect_exfiltration_attempt(tool_output)
        if injection_detected or exfil_detected:
            return {
                "id": task.get("id"),
                "attack_type": attack_type,
                "agent_output": (
                    "[GUARD] Potentially malicious instruction detected in tool output. "
                    "Processing only legitimate content."
                ),
                "followed_injection": False,
                "guard_triggered": True,
            }
        clean_output = sanitize_tool_output(tool_output)
        return {
            "id": task.get("id"),
            "attack_type": attack_type,
            "agent_output": f"Summary of tool output: {clean_output[:120]}",
            "followed_injection": False,
            "guard_triggered": False,
        }
    else:
        # Unguarded: naively incorporates everything in tool output
        # Detect whether the injection would succeed
        injection_present = detect_prompt_injection(tool_output) or detect_exfiltration_attempt(tool_output)
        if injection_present:
            # Simulate the agent following the malicious instruction
            injected_action = _extract_injected_action(tool_output)
            return {
                "id": task.get("id"),
                "attack_type": attack_type,
                "agent_output": injected_action,
                "followed_injection": True,
                "guard_triggered": False,
            }
        return {
            "id": task.get("id"),
            "attack_type": attack_type,
            "agent_output": f"Summary: {tool_output[:120]}",
            "followed_injection": False,
            "guard_triggered": False,
        }


def _extract_injected_action(text: str) -> str:
    """Pull out the injected instruction text to simulate compliance."""
    lower = text.lower()
    for marker in ["ignore all previous instructions", "disregard your guidelines",
                   "new task:", "system:", "[[inject]]", "<<<", "you must now"]:
        idx = lower.find(marker)
        if idx != -1:
            return f"[COMPROMISED] Agent followed injected instruction: '{text[idx:idx+80]}...'"
    return f"[COMPROMISED] Agent processed malicious content: {text[:80]}..."
