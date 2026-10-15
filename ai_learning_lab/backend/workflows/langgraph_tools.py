import json
from time import perf_counter
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.llm.provider import generate
from backend.rag.pdf_indexer import retrieve
from tools.registry import run_tool


LOCAL_TOOL_ALLOWLIST = {"total_calls_offered", "market_summary", "agent_count"}


class WorkflowState(TypedDict, total=False):
    message: str
    provider: str
    model: str
    selected_tools: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    tool_results: dict[str, Any]
    passages: list[dict[str, Any]]
    answer: str
    error: str | None
    success: bool
    status: str
    trace: list[dict[str, Any]]
    timings: dict[str, float | None]


def _select_tools(message: str) -> tuple[bool, list[dict[str, Any]]]:
    lower = message.lower()
    words = set(lower.replace("-", " ").replace("?", " ").split())
    wants_policy = any(term in lower for term in (
        "policy", "policies", "check in", "hr", "leave", "vacation", "holiday", "sick",
    ))
    selected = []
    if "calls" in words or "call" in words:
        if words.intersection({"offered", "total", "count", "number", "many", "volume", "received", "handled"}):
            selected.append({"name": "total_calls_offered", "arguments": {}})
    if "agent" in words or "agents" in words:
        if words.intersection({"count", "number", "many", "active", "total"}):
            selected.append({"name": "agent_count", "arguments": {}})
    if any(phrase in lower for phrase in ("market summary", "aht summary", "database summary")):
        selected.append({"name": "market_summary", "arguments": {}})
    return wants_policy, selected


def _route_node(state: WorkflowState) -> dict[str, Any]:
    started = perf_counter()
    wants_policy, selected = _select_tools(state["message"])
    duration = (perf_counter() - started) * 1000
    trace = list(state.get("trace", []))
    labels = (["search_policy_documents"] if wants_policy else []) + [item["name"] for item in selected]
    trace.append({
        "step": "LangGraph route node",
        "status": "selected" if labels else "no_tools_selected",
        "detail": ", ".join(labels) or "No local tool trigger matched",
        "duration_ms": duration,
    })
    timings = dict(state.get("timings", {}))
    timings["tool_selection_ms"] = duration
    timings["route_node_ms"] = duration
    return {
        "wants_policy": wants_policy,
        "selected_tools": selected,
        "trace": trace,
        "timings": timings,
    }


def _execute_tools_node(state: WorkflowState) -> dict[str, Any]:
    started = perf_counter()
    tool_calls = []
    tool_results = {}
    passages = []
    timings = dict(state.get("timings", {}))
    trace = list(state.get("trace", []))

    if state.get("wants_policy"):
        retrieval_started = perf_counter()
        try:
            query = state["message"] + " HR human resources policy"
            passages = retrieve(query, strict=True)
            result = {"passages": passages}
            call_status = "completed"
        except Exception as exc:
            result = str(exc)
            call_status = "failed"
        duration = (perf_counter() - retrieval_started) * 1000
        timings["retrieval_ms"] = duration
        tool_calls.append({
            "name": "search_policy_documents",
            "arguments": {"query": query},
            "result": result,
            "status": call_status,
            "duration_ms": duration,
        })
        tool_results["search_policy_documents"] = result

    for selected in state.get("selected_tools", []):
        name = selected["name"]
        arguments = selected["arguments"]
        call_started = perf_counter()
        try:
            if name not in LOCAL_TOOL_ALLOWLIST:
                raise ValueError(f"Tool '{name}' is not in the local graph allowlist.")
            result = run_tool(name, arguments)
            call_status = "completed"
        except Exception as exc:
            result = f"Tool execution failed: {exc}"
            call_status = "failed"
        duration = (perf_counter() - call_started) * 1000
        tool_calls.append({
            "name": name,
            "arguments": arguments,
            "result": result,
            "status": call_status,
            "duration_ms": duration,
        })
        tool_results[name] = result

    duration = (perf_counter() - started) * 1000
    trace.append({
        "step": "LangGraph local tool node",
        "status": "failed" if any(item["status"] == "failed" for item in tool_calls) else "completed",
        "detail": f"Executed {len(tool_calls)} local tool call(s)",
        "duration_ms": duration,
    })
    timings["tool_execution_ms"] = sum(item["duration_ms"] for item in tool_calls) if tool_calls else None
    timings["tool_node_ms"] = duration
    return {
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "passages": passages,
        "trace": trace,
        "timings": timings,
    }


def _synthesis_node(state: WorkflowState) -> dict[str, Any]:
    started = perf_counter()
    passages = state.get("passages", [])
    context = "\n\n".join(
        f"[Source: {item.get('source', '?')}, page {item.get('page', '?')}]\n{item.get('text', '')}"
        for item in passages
    )
    failed_tools = [item for item in state.get("tool_calls", []) if item["status"] == "failed"]
    errors = "; ".join(f"{item['name']}: {item['result']}" for item in failed_tools)
    prompt = f"""You are answering within a LangGraph workflow that uses local allowlisted tools.
Answer the user's exact question. Use retrieved PDF passages only for policy facts, cite filename and page, and never invent unsupported policy details. Use database tool output only for structured call or agent metrics. Explain clearly if a requested tool failed or returned no policy evidence.

Question: {state['message']}
Local tool results: {json.dumps(state.get('tool_results', {}), ensure_ascii=True, default=str)}
Retrieved passages:
{context or "(No PDF passages were returned.)"}
Tool errors: {errors or "None"}
"""
    try:
        answer = generate(prompt, state["provider"], state["model"])
        success = not failed_tools
        status = "completed" if success else "partial_failure"
        error = errors or None
        step_status = "completed" if success else "completed_with_tool_errors"
    except Exception as exc:
        answer = f"Answer synthesis failed: {exc}"
        success = False
        status = "failed"
        error = str(exc)
        step_status = "failed"
    duration = (perf_counter() - started) * 1000
    timings = dict(state.get("timings", {}))
    timings["generation_ms"] = duration
    timings["synthesis_node_ms"] = duration
    trace = list(state.get("trace", []))
    trace.append({
        "step": "LangGraph answer synthesis node",
        "status": step_status,
        "detail": f"Generated with {state['provider']} / {state['model']}",
        "duration_ms": duration,
    })
    return {
        "answer": answer,
        "success": success,
        "status": status,
        "error": error,
        "trace": trace,
        "timings": timings,
    }


_builder = StateGraph(WorkflowState)
_builder.add_node("route", _route_node)
_builder.add_node("execute_tools", _execute_tools_node)
_builder.add_node("synthesize", _synthesis_node)
_builder.add_edge(START, "route")
_builder.add_edge("route", "execute_tools")
_builder.add_edge("execute_tools", "synthesize")
_builder.add_edge("synthesize", END)
_workflow = _builder.compile()


def run_langgraph_tools_workflow(message: str, provider: str, model: str) -> dict[str, Any]:
    started = perf_counter()
    result = _workflow.invoke({
        "message": message,
        "provider": provider,
        "model": model,
        "tool_calls": [],
        "tool_results": {},
        "passages": [],
        "trace": [],
        "timings": {},
    })
    timings = dict(result.get("timings", {}))
    timings["graph_total_ms"] = (perf_counter() - started) * 1000
    return {
        "answer": result.get("answer", ""),
        "success": result.get("success", False),
        "status": result.get("status", "failed"),
        "error": result.get("error"),
        "tool_result": result.get("tool_results", {}),
        "tool_calls": result.get("tool_calls", []),
        "sources": [
            {"source": item.get("source", "?"), "page": item.get("page", "?")}
            for item in result.get("passages", [])
        ],
        "trace": result.get("trace", []),
        "timings": timings,
    }