"""
Mock tool implementations for the threaded research assistant agent.
All functions return deterministic results — no network calls.
"""
import hashlib
from typing import Any, Dict, List, Optional


def _hash_key(text: str) -> int:
    """Stable integer derived from text, used to pick deterministic responses."""
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % 100


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def search_web(query: str, date_range: Optional[str] = None) -> Dict[str, Any]:
    """Simulates a web search. Returns a list of mock result snippets."""
    key = _hash_key(query)
    results = [
        {"title": f"Study on {query[:30]}", "url": f"https://example.com/paper_{key}",
         "snippet": f"Recent findings suggest that {query} is an active area of research with growing interest."},
        {"title": f"Survey: {query[:25]}...", "url": f"https://example.com/survey_{key+1}",
         "snippet": f"A comprehensive survey covering {query} was published, identifying key benchmarks."},
    ]
    if date_range == "recent":
        results[0]["published"] = "2024"
        results[1]["published"] = "2024"
    return {"status": "success", "results": results, "query": query}


def search_docs(query: str) -> Dict[str, Any]:
    """Searches an internal document store. Returns matching doc stubs."""
    key = _hash_key(query)
    docs = [
        {"doc_id": f"doc_{key}", "title": f"Internal Report: {query[:30]}",
         "relevance_score": 0.91},
        {"doc_id": f"doc_{key+1}", "title": f"Technical Memo on {query[:25]}",
         "relevance_score": 0.74},
    ]
    return {"status": "success", "documents": docs, "query": query}


def read_document(doc_id: str) -> Dict[str, Any]:
    """Fetches the content of a single document by ID."""
    key = _hash_key(doc_id)
    return {
        "status": "success",
        "doc_id": doc_id,
        "title": f"Document {doc_id}",
        "content": (
            f"This document (id={doc_id}) covers several findings. "
            f"Section 1 discusses background context. "
            f"Section 2 presents experimental results showing a {30 + key % 40}% improvement. "
            f"Section 3 concludes with recommendations for future work."
        ),
        "word_count": 320 + key,
    }


def summarize_evidence(evidence: List[str]) -> Dict[str, Any]:
    """Produces a short mock summary of a list of evidence strings."""
    combined = " ".join(evidence)
    n = len(evidence)
    return {
        "status": "success",
        "summary": (
            f"Based on {n} source(s), the evidence consistently highlights key themes "
            f"including methodology, performance benchmarks, and practical applications. "
            f"First source begins: '{combined[:80]}...'"
        ),
        "source_count": n,
    }


def cite_sources(answer: str, sources: List[str]) -> Dict[str, Any]:
    """Annotates an answer with inline citations."""
    cited = answer
    for i, src in enumerate(sources, start=1):
        cited += f" [{i}: {src[:50]}]"
    return {"status": "success", "cited_answer": cited, "citation_count": len(sources)}


def ask_clarification(question: str) -> Dict[str, Any]:
    """Signals that the agent needs more information from the user."""
    return {
        "status": "success",
        "clarification_requested": True,
        "question": question,
    }


# Registry used by the mock agent to dispatch tool calls
TOOL_REGISTRY = {
    "search_web": search_web,
    "search_docs": search_docs,
    "read_document": read_document,
    "summarize_evidence": summarize_evidence,
    "cite_sources": cite_sources,
    "ask_clarification": ask_clarification,
}


def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a tool call by name. Returns error dict on unknown tool."""
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    try:
        return fn(**arguments)
    except TypeError as exc:
        return {"status": "error", "message": str(exc)}
