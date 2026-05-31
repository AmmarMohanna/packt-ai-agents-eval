"""
Utility functions for loading, saving, and inspecting agent traces.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_jsonl(path: str) -> List[Dict]:
    """Load a newline-delimited JSON file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict], path: str) -> None:
    """Write a list of dicts to a newline-delimited JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def flatten_trace(trace: Dict) -> List[Dict]:
    """
    Flatten a trace into a list of step dicts, each enriched with
    the trace-level task description and trace_id.
    """
    rows = []
    for step in trace.get("steps", []):
        row = {"trace_id": trace.get("trace_id"), "task": trace.get("task")}
        row.update(step)
        rows.append(row)
    return rows


def trace_to_dataframe(trace: Dict) -> pd.DataFrame:
    """Convert a single trace into a step-level DataFrame."""
    rows = flatten_trace(trace)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def summarize_trace(trace: Dict) -> Dict[str, Any]:
    """Return a one-row summary dict for a trace."""
    steps = trace.get("steps", [])
    n_steps = len(steps)
    total_lat = sum(s.get("latency_ms", 0) for s in steps)
    total_tok = sum(s.get("tokens", 0) for s in steps)
    tools_used = [s.get("tool_name") for s in steps if s.get("tool_name")]
    failures = [s for s in steps if s.get("status") == "failure"]
    return {
        "trace_id": trace.get("trace_id"),
        "task_snippet": trace.get("task", "")[:50],
        "n_steps": n_steps,
        "total_latency_ms": total_lat,
        "total_tokens": total_tok,
        "tools_used": tools_used,
        "n_failures": len(failures),
        "has_final_answer": bool(trace.get("final_answer")),
    }
