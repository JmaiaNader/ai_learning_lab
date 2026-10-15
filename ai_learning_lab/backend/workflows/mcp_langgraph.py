import asyncio
import json
import os
import sys
from time import perf_counter
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from backend.llm.provider import generate


MCP_TOOL_ALLOWLIST = {"search_hr_policy", "get_total_calls_offered"}


class WorkflowState(TypedDict, total=False):
    message: str
    provider: str
    model: str
    selected_tools: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    tool_results: dict[str, Any]
    sources: list[dict[str, Any]]
    answer: str
    error: str | None
    success: bool
    status: str
    trace: list[dict[str, Any]]
    timings: dict[str, float | None]


def _wants_policy(message: str) -> bool:
    text = message.lower()
    return any(
        marker in text
        for marker in (
            "policy", "policies", "check-in", "check in", "hr", "leave",
            "vacation", "holiday", "sick", "entitlement", "eligibility",
        )
    )


def _wants_call_count(message: str) -> bool:
    words = set(message.lower().replace("-", " ").replace("?", " ").split())
    return bool(words.intersection({"call", "calls"}) and words.intersection({
        "volume", "offered", "total", "count", "received", "handled", "number", "many",
    }))


async def _call_mcp_tools(selected_tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "tools.mcp_server"],
        cwd=project_root,
        env=dict(os.environ),
    )
    setup_started = perf_counter()
    calls: list[dict[str, Any]] = []
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            registered = {tool.name for tool in (await session.list_tools()).tools}
            setup_ms = (perf_counter() - setup_started) * 1000
            for selected in selected_tools:
                name = selected["name"]
                arguments = selected["arguments"]
                started = perf_counter()
                if name not in MCP_TOOL_ALLOWLIST or name not in registered:
                    calls.append({
                        "name": name,
                        "arguments": arguments,
                        "result": "Tool is not registered in the MCP allowlist.",
                        "status": "failed",
                        "duration_ms": (perf_counter() - started) * 1000,
                    })
                    continue
                try:
                    result = await session.call_tool(name, arguments)
                    if result.isError:
                        detail = "; ".join(
                            getattr(item, "text", "")
                            for item in result.content
                            if getattr(item, "text", "")
                        )
                        raise RuntimeError(detail or "MCP server reported a tool execution error")
                    value = result.structuredContent
                    if value is None:
                        text_values = [
                            getattr(item, "text", str(item)) for item in result.content
                        ]
                        if len(text_values) == 1:
                            try:
                                value = json.loads(text_values[0])
                            except (TypeError, json.JSONDecodeError):
                                value = text_values[0]
                        else:
                            value = text_values
                    calls.append({
                        "name": name,
                        "arguments": arguments,
                        "result": value,
                        "status": "completed",
                        "duration_ms": (perf_counter() - started) * 1000,
                    })
                except Exception as exc:
                    calls.append({
                        "name": name,
                        "arguments": arguments,
                        "result": str(exc),
                        "status": "failed",
                        "duration_ms": (perf_counter() - started) * 1000,
                    })
    return calls, setup_ms


def _route_node(state: WorkflowState) -> dict[str, Any]:
    started = perf_counter()
    message = state["message"]
    selected = []
    if _wants_policy(message):
        selected.append({"name": "search_hr_policy", "arguments": {"query": message}})
    if _wants_call_count(message):
        selected.append({"name": "get_total_calls_offered", "arguments": {}})
    duration = (perf_counter() - started) * 1000
    trace = list(state.get("trace", []))
    trace.append({
        "step": "LangGraph route node",
        "status": "selected" if selected else "no_tools_selected",
        "detail": ", ".join(item["name"] for item in selected) or "No registered tool trigger matched",
        "duration_ms": duration,
    })
    timings = dict(state.get("timings", {}))
    timings["tool_selection_ms"] = duration
    timings["route_node_ms"] = duration
    return {"selected_tools": selected, "trace": trace, "timings": timings}


def _execute_tools_node(state: WorkflowState) -> dict[str, Any]:
    started = perf_counter()
    calls: list[dict[str, Any]] = []
    setup_ms = 0.0
    if state.get("selected_tools"):
        try:
            calls, setup_ms = asyncio.run(_call_mcp_tools(state["selected_tools"]))
        except Exception as exc:
            calls = [
                {
                    "name": item["name"],
                    "arguments": item["arguments"],
                    "result": str(exc),
                    "status": "failed",
                    "duration_ms": 0.0,
                }
                for item in state["selected_tools"]
            ]

    tool_results = {item["name"]: item["result"] for item in calls}
    sources = []
    policy_result = tool_results.get("search_hr_policy")
    if isinstance(policy_result, dict):
        sources = policy_result.get("passages", [])
    duration = (perf_counter() - started) * 1000
    trace = list(state.get("trace", []))
    trace.append({
        "step": "LangGraph MCP tool node",
        "status": "failed" if any(item["status"] == "failed" for item in calls) else "completed",
        "detail": f"Executed {len(calls)} MCP tool call(s)",
        "duration_ms": duration,
    })
    timings = dict(state.get("timings", {}))
    timings["mcp_client_setup_ms"] = setup_ms if calls else None
    timings["tool_execution_ms"] = sum(item["duration_ms"] for item in calls) if calls else None
    timings["tool_node_ms"] = duration
    return {
        "tool_calls": calls,
        "tool_results": tool_results,
        "sources": sources,
        "trace": trace,
        "timings": timings,
    }


def _synthesis_node(state: WorkflowState) -> dict[str, Any]:
    started = perf_counter()
    message = state["message"]
    passages = state.get("sources", [])
    context = "\n\n".join(
        f"[Source: {item.get('source', '?')}, page {item.get('page', '?')}]\n{item.get('text', '')}"
        for item in passages
    )
    failed_tools = [
        item for item in state.get("tool_calls", []) if item["status"] == "failed"
    ]
    if failed_tools:
        error = "; ".join(f"{item['name']}: {item['result']}" for item in failed_tools)
        tool_error_context = f"Tool failures (report these clearly; do not invent their results): {error}"
    else:
        tool_error_context = "All selected tools completed, or no tool was needed."
    prompt = f"""You are answering within a LangGraph + MCP learning workflow.
Answer the user's exact question. Use retrieved PDF passages only for HR policy facts, cite filename/page, and never invent policy content. Use the MySQL tool result only for call-count facts. Clearly state when a requested source returned no evidence or failed.

Question: {message}
MCP tool results: {json.dumps(state.get('tool_results', {}), ensure_ascii=True, default=str)}
Retrieved PDF passages:
{context or "(No policy passages were returned.)"}
{tool_error_context}
"""
    try:
        answer = generate(prompt, state["provider"], state["model"])
        success = not failed_tools
        status = "completed" if success else "partial_failure"
        error = "; ".join(f"{item['name']}: {item['result']}" for item in failed_tools) or None
        synthesis_status = "completed" if success else "completed_with_tool_errors"
    except Exception as exc:
        answer = f"Answer synthesis failed: {exc}"
        success = False
        status = "failed"
        error = str(exc)
        synthesis_status = "failed"
    duration = (perf_counter() - started) * 1000
    trace = list(state.get("trace", []))
    trace.append({
        "step": "LangGraph answer synthesis node",
        "status": synthesis_status,
        "detail": f"Generated with {state['provider']} / {state['model']}",
        "duration_ms": duration,
    })
    timings = dict(state.get("timings", {}))
    timings["generation_ms"] = duration
    timings["synthesis_node_ms"] = duration
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


def run_mcp_workflow(message: str, provider: str, model: str) -> dict[str, Any]:
    started = perf_counter()
    result = _workflow.invoke({
        "message": message,
        "provider": provider,
        "model": model,
        "tool_calls": [],
        "tool_results": {},
        "sources": [],
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
            for item in result.get("sources", [])
        ],
        "trace": result.get("trace", []),
        "timings": timings,
    }