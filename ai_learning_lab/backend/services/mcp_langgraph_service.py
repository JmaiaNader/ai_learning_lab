"""Tool Calling 4: LangGraph orchestrating the shared analytics tool through MCP.

The query planner is reused from Tool Calling 2. This track differs at execution:
contact-center data is discovered and called over the Model Context Protocol via a
local stdio MCP server. The server exposes the same allowlisted query function; it
never accepts model-generated SQL.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, TypedDict

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph

from backend.llm.provider import generate
from backend.rag.pdf_indexer import retrieve
from ai_learning_lab.backend.services.langgraph_service_OLD import (
    _extract_sql_arguments,
    _generate_node,
    _is_policy_question,
    _retrieve_node,
    _select_sql_tool,
    _with_trace,
)
from config.settings import PROJECT_ROOT, settings

MCP_TOOL_NAME = "query_contact_center_data"
MCP_SERVER_NAME = "contact_center_analytics"
MCP_DISCOVERY_TIMEOUT_SECONDS = 30
MCP_TOOL_TIMEOUT_SECONDS = 30


class MCPGraphState(TypedDict, total=False):
    message: str
    provider: str
    model: str
    use_rag: bool
    selected_sql_tool: str | None
    sql_arguments: dict[str, Any]
    should_retrieve: bool
    passages: list[dict[str, Any]]
    tool_result: Any
    tool_calls: list[dict[str, Any]]
    answer: str
    trace: list[dict[str, Any]]
    timings: dict[str, float | None]
    component_failed: bool
    tool_failed: bool
    status: str
    error: str | None
    success: bool
    mcp_tool_names: list[str]


def _mcp_client() -> MultiServerMCPClient:
    """Create a local stdio client using the same Python venv as FastAPI."""
    project_root = str(PROJECT_ROOT.resolve())
    child_env = os.environ.copy()
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = project_root + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    return MultiServerMCPClient({
        MCP_SERVER_NAME: {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "tools.mcp_server"],
            "cwd": project_root,
            "env": child_env,
        }
    })


def _plan_node(state: MCPGraphState) -> dict[str, Any]:
    started = perf_counter()
    message = state["message"]
    # Reuse Tool Calling 2's safe, generic planner. Tool Calling 4 only exposes
    # the shared analytics query over MCP; it does not duplicate SQL functions.
    selected = _select_sql_tool(message)
    selected_mcp_tool = MCP_TOOL_NAME if selected == MCP_TOOL_NAME else None
    should_retrieve = bool(state.get("use_rag", True) and _is_policy_question(message))
    intent = "sql_and_pdf" if selected_mcp_tool and should_retrieve else (
        "mcp_sql" if selected_mcp_tool else ("pdf_retrieval" if should_retrieve else "llm_only")
    )
    elapsed = (perf_counter() - started) * 1000
    timings = dict(state.get("timings", {}))
    timings["tool_selection_ms"] = elapsed
    return {
        "selected_sql_tool": selected_mcp_tool,
        "sql_arguments": _extract_sql_arguments(message, selected_mcp_tool) if selected_mcp_tool else {},
        "should_retrieve": should_retrieve,
        "passages": [],
        "tool_result": None,
        "tool_calls": [],
        "mcp_tool_names": [],
        "trace": _with_trace(state, {
            "step": "LangGraph MCP planner",
            "status": "selected",
            "detail": f"Intent={intent}; MCP analytics tool={selected_mcp_tool or 'none'}; PDF retrieval={'yes' if should_retrieve else 'no'}",
            "duration_ms": elapsed,
        }),
        "timings": timings,
        "component_failed": False,
        "tool_failed": False,
        "error": None,
    }


def _after_plan(state: MCPGraphState) -> str:
    if state.get("should_retrieve"):
        return "retrieve"
    if state.get("selected_sql_tool"):
        return "mcp_tool"
    return "generate"


def _after_retrieval(state: MCPGraphState) -> str:
    return "mcp_tool" if state.get("selected_sql_tool") else "generate"


def _decode_tool_result(raw: Any) -> Any:
    """Normalize MCP adapter results into the app's structured query payload.

    Depending on the installed langchain-mcp-adapters / MCP SDK versions,
    ``Tool.ainvoke`` may return the tool's JSON string, a CallToolResult with a
    ``content`` list, or the content-block list itself. Support all three forms.
    """
    if isinstance(raw, dict):
        # Some adapters may pass through the structured payload directly.
        if any(key in raw for key in ("has_data", "rows", "query_spec", "error", "tool")):
            return raw
        # Or a single text block may be represented as a plain mapping.
        if raw.get("type") == "text" and isinstance(raw.get("text"), str):
            return _decode_tool_result(raw["text"])
        return raw

    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "error": "MCP returned non-JSON output.",
                "raw_output": raw[:2000],
                "has_data": False,
                "rows": [],
            }
        # Decode wrapped JSON strings if the adapter encoded more than one layer.
        return _decode_tool_result(decoded) if isinstance(decoded, (str, list)) else decoded

    if hasattr(raw, "content"):
        return _decode_tool_result(raw.content)

    if isinstance(raw, list):
        # Some MCP clients return content blocks directly instead of wrapping
        # them in CallToolResult.content. First extract text blocks, which in
        # our server contain the JSON payload returned by tools/mcp_server.py.
        text_parts: list[str] = []
        structured_payloads: list[dict[str, Any]] = []
        for block in raw:
            if isinstance(block, str):
                if block.strip():
                    text_parts.append(block)
                continue
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
                    continue
                if any(key in block for key in ("has_data", "rows", "query_spec", "error", "tool")):
                    structured_payloads.append(block)
                    continue
                if isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
                    continue
            block_text = getattr(block, "text", None)
            if isinstance(block_text, str):
                text_parts.append(block_text)

        if structured_payloads:
            if len(structured_payloads) == 1 and not text_parts:
                return structured_payloads[0]
            # Prefer a complete structured response over its text representation.
            return structured_payloads[0]
        if text_parts:
            # Prefer a single parseable JSON text block. MCP can also return
            # more than one text block; inspect each before trying a joined value.
            for part in text_parts:
                try:
                    parsed = json.loads(part)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return _decode_tool_result(parsed)
                if isinstance(parsed, (list, str)):
                    nested = _decode_tool_result(parsed)
                    if isinstance(nested, dict) and not nested.get("error"):
                        return nested
            joined = "\n".join(text_parts)
            return {
                "error": "MCP returned text content that could not be decoded as a structured JSON result.",
                "raw_output": joined[:2000],
                "has_data": False,
                "rows": [],
            }
        return {
            "error": "MCP returned a list without supported text or structured content blocks.",
            "has_data": False,
            "rows": [],
        }

    return {"error": f"Unsupported MCP result type: {type(raw).__name__}", "has_data": False, "rows": []}


async def _invoke_mcp_tool(arguments: dict[str, Any]) -> tuple[Any, list[str], float, float]:
    client = _mcp_client()
    discovery_started = perf_counter()
    try:
        tools = await asyncio.wait_for(
            client.get_tools(), timeout=MCP_DISCOVERY_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"MCP server discovery timed out after {MCP_DISCOVERY_TIMEOUT_SECONDS} seconds. "
            "Check tools/mcp_server.py, installed MCP dependencies, and stderr in the FastAPI terminal."
        ) from exc
    discovery_ms = (perf_counter() - discovery_started) * 1000
    tool_names = [tool.name for tool in tools]
    selected = next((tool for tool in tools if tool.name == MCP_TOOL_NAME), None)
    if selected is None:
        raise RuntimeError(
            f"MCP server '{MCP_SERVER_NAME}' connected but did not expose '{MCP_TOOL_NAME}'. "
            f"Discovered tools: {tool_names or 'none'}"
        )
    call_started = perf_counter()
    try:
        raw = await asyncio.wait_for(
            selected.ainvoke(arguments), timeout=MCP_TOOL_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"MCP tool invocation timed out after {MCP_TOOL_TIMEOUT_SECONDS} seconds. "
            "Check MySQL connectivity and the query arguments in the execution trace."
        ) from exc
    call_ms = (perf_counter() - call_started) * 1000
    return _decode_tool_result(raw), tool_names, discovery_ms, call_ms


async def _mcp_tool_node(state: MCPGraphState) -> dict[str, Any]:
    arguments = state.get("sql_arguments", {})
    timings = dict(state.get("timings", {}))
    trace = list(state.get("trace", []))
    tool_calls = list(state.get("tool_calls", []))
    try:
        result, names, discovery_ms, call_ms = await _invoke_mcp_tool(arguments)
        timings["mcp_discovery_ms"] = discovery_ms
        timings["tool_execution_ms"] = call_ms
        trace.append({
            "step": "MCP server discovery",
            "status": "completed",
            "detail": f"Discovered {len(names)} MCP tool(s): {', '.join(names)}",
            "duration_ms": discovery_ms,
        })
        tool_failed = isinstance(result, dict) and bool(result.get("error"))
        tool_status = "failed" if tool_failed else "completed"
        if tool_failed:
            error_detail = _safe_mcp_error(str(result.get("error", "Unknown MCP query error")))
            invocation_detail = f"MCP tool returned a query error: {error_detail}"
        else:
            invocation_detail = f"Called '{MCP_TOOL_NAME}' through MCP server '{MCP_SERVER_NAME}'"
        trace.append({
            "step": "MCP tool invocation",
            "status": tool_status,
            "detail": invocation_detail,
            "duration_ms": call_ms,
        })
        tool_calls.append({
            "name": MCP_TOOL_NAME,
            "transport": "stdio",
            "server": MCP_SERVER_NAME,
            "arguments": arguments,
            "result": result,
            "status": tool_status,
            "duration_ms": call_ms,
        })
        return {
            "tool_result": result,
            "tool_calls": tool_calls,
            "mcp_tool_names": names,
            "trace": trace,
            "timings": timings,
            "tool_failed": tool_failed,
            "component_failed": bool(state.get("component_failed") or tool_failed),
            "error": result.get("error") if tool_failed else state.get("error"),
        }
    except Exception as exc:
        elapsed = 0.0
        timings["tool_execution_ms"] = elapsed
        safe_exc = _safe_mcp_error(f"{type(exc).__name__}: {exc}")
        trace.append({
            "step": "MCP tool invocation",
            "status": "failed",
            "detail": safe_exc,
            "duration_ms": elapsed,
        })
        tool_calls.append({
            "name": MCP_TOOL_NAME,
            "transport": "stdio",
            "server": MCP_SERVER_NAME,
            "arguments": arguments,
            "result": {"error": safe_exc},
            "status": "failed",
            "duration_ms": elapsed,
        })
        return {
            "tool_result": {"error": safe_exc, "has_data": False, "rows": [], "message": "MCP tool execution failed."},
            "tool_calls": tool_calls,
            "trace": trace,
            "timings": timings,
            "tool_failed": True,
            "component_failed": True,
            "error": safe_exc,
        }


def _safe_mcp_error(value: str) -> str:
    """Expose actionable tool errors without leaking configured secrets."""
    message = str(value or "Unknown MCP error")[:1200]
    for secret in (getattr(settings, "mysql_password", ""), getattr(settings, "openai_api_key", "")):
        if secret:
            message = message.replace(str(secret), "[REDACTED]")
    return message


def _after_mcp_tool(state: MCPGraphState) -> str:
    """Route errors to a deterministic diagnostic rather than hiding them in a generic LLM reply."""
    result = state.get("tool_result")
    if state.get("tool_failed") or (isinstance(result, dict) and result.get("error")):
        return "format_error"
    if (
        state.get("selected_sql_tool") == MCP_TOOL_NAME
        and not state.get("should_retrieve")
        and isinstance(result, dict)
        and result.get("requested_view") in {"summary", "table", "chart"}
        and not result.get("error")
    ):
        return "format_result"
    return "generate"


def _format_mcp_result_node(state: MCPGraphState) -> dict[str, Any]:
    """Return a concise, deterministic status while preserving rows for the UI."""
    started = perf_counter()
    result = state.get("tool_result") or {}
    rows = result.get("rows") or []
    row_count = int(result.get("row_count", len(rows)) or 0)
    spec = result.get("query_spec") or {}
    view_mode = result.get("requested_view") or spec.get("view_mode") or "table"
    metrics = spec.get("metrics") or []
    labels = {
        "offered": "offered calls",
        "answered": "answered calls",
        "abandoned": "abandoned calls",
        "sla": "SLA",
        "aht": "weighted AHT",
        "asa": "weighted ASA",
    }
    metric_text = ", ".join(labels.get(str(item), str(item)) for item in metrics) or "requested metrics"
    filters = spec.get("filters") or {}
    filter_parts = []
    if filters.get("channel"):
        filter_parts.append(str(filters["channel"]))
    if filters.get("market"):
        filter_parts.append(str(filters["market"]))
    filter_text = f" for {' in '.join(filter_parts)}" if filter_parts else ""
    date_preset = spec.get("date_preset")
    if date_preset:
        date_text = str(date_preset).replace("_", " ")
    elif spec.get("start_date") or spec.get("end_date"):
        date_text = f"from {spec.get('start_date') or 'the beginning'} to {spec.get('end_date') or 'the latest date'}"
    elif spec.get("dates"):
        date_text = "on " + ", ".join(str(item) for item in spec["dates"])
    else:
        date_text = "for the selected data range"

    if not rows:
        answer = (
            f"The SQL query completed, but no rows matched {metric_text}{filter_text} "
            f"for {date_text}. No values were estimated or filled in."
        )
    elif view_mode == "chart":
        answer = (
            f"Retrieved {row_count} data point(s) for {metric_text}{filter_text} "
            f"for {date_text}. The chart below uses the returned MySQL values. "
            "Dates absent from the source data are not filled or estimated."
        )
    elif view_mode == "table":
        answer = (
            f"Retrieved {row_count} row(s) for {metric_text}{filter_text} "
            f"for {date_text}. The results are shown in the table below."
        )
    else:
        # This is intentionally conservative: actual figures remain in tool_result/data_view.
        answer = (
            f"The query returned {row_count} row(s) for {metric_text}{filter_text} "
            f"for {date_text}. The verified values are included in the result details."
        )

    elapsed_ms = (perf_counter() - started) * 1000
    timings = dict(state.get("timings", {}))
    timings["generation_ms"] = 0.0
    trace = list(state.get("trace", []))
    trace.append({
        "step": "Deterministic analytics response",
        "status": "completed",
        "detail": "Skipped Ollama synthesis for structured SQL analytics; preserved verified rows for table/chart rendering.",
        "duration_ms": elapsed_ms,
    })
    return {
        "answer": answer,
        "trace": trace,
        "timings": timings,
        "status": "completed",
        "success": True,
        "error": None,
    }

def _format_mcp_error_node(state: MCPGraphState) -> dict[str, Any]:
    """Show the actual MCP-side error; do not disguise it as a generic SQL failure."""
    result = state.get("tool_result") or {}
    raw_error = result.get("error") or state.get("error") or "MCP tool failed without returning an error detail."
    safe_error = _safe_mcp_error(str(raw_error))
    arguments = state.get("sql_arguments", {})
    trace = list(state.get("trace", []))
    trace.append({
        "step": "MCP error diagnostics",
        "status": "failed",
        "detail": f"Tool={MCP_TOOL_NAME}; arguments={json.dumps(arguments, default=str)}; error={safe_error}",
        "duration_ms": 0.0,
    })
    timings = dict(state.get("timings", {}))
    timings["generation_ms"] = 0.0
    return {
        "answer": (
            "Tool Calling 4 reached the MCP query path, but the query did not succeed. "
            f"Exact MCP/tool error: {safe_error}"
        ),
        "trace": trace,
        "timings": timings,
        "status": "tool_error",
        "success": False,
        "component_failed": True,
        "tool_failed": True,
        "error": safe_error,
    }


def _build_graph():
    graph = StateGraph(MCPGraphState)
    graph.add_node("plan", _plan_node)
    # Reuse the same local PDF retrieval node as Tool Calling 2.
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("mcp_tool", _mcp_tool_node)
    graph.add_node("format_result", _format_mcp_result_node)
    graph.add_node("format_error", _format_mcp_error_node)
    graph.add_node("generate", _generate_node)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan", _after_plan,
        {"retrieve": "retrieve", "mcp_tool": "mcp_tool", "generate": "generate"},
    )
    graph.add_conditional_edges(
        "retrieve", _after_retrieval,
        {"mcp_tool": "mcp_tool", "generate": "generate"},
    )
    graph.add_conditional_edges(
        "mcp_tool", _after_mcp_tool,
        {"format_result": "format_result", "format_error": "format_error", "generate": "generate"},
    )
    graph.add_edge("format_result", END)
    graph.add_edge("format_error", END)
    graph.add_edge("generate", END)
    return graph.compile()


_MCP_GRAPH = _build_graph()


def run_mcp_langgraph_workflow(message: str, provider: str, model: str | None, use_rag: bool = True) -> dict[str, Any]:
    """Run Tool Calling 4 and return the shared chat response contract."""
    started = perf_counter()
    selected_provider = (provider or "ollama").lower()
    selected_model = model or (settings.ollama_model if selected_provider == "ollama" else settings.openai_model)
    initial_state: MCPGraphState = {
        "message": message,
        "provider": selected_provider,
        "model": selected_model,
        "use_rag": use_rag,
        "trace": [],
        "timings": {
            "retrieval_ms": None,
            "tool_selection_ms": None,
            "tool_execution_ms": None,
            "mcp_discovery_ms": None,
            "generation_ms": None,
            "server_total_ms": None,
        },
        "tool_calls": [],
        "passages": [],
        "component_failed": False,
        "tool_failed": False,
    }
    try:
        # FastAPI's /chat endpoint is synchronous (def), so asyncio.run is safe here.
        result = asyncio.run(_MCP_GRAPH.ainvoke(initial_state))
        timings = dict(result.get("timings", {}))
        timings["server_total_ms"] = (perf_counter() - started) * 1000
        query_result = result.get("tool_result") if result.get("selected_sql_tool") == MCP_TOOL_NAME else None
        return {
            "answer": result.get("answer", "The MCP graph returned no answer."),
            "success": bool(result.get("success", not result.get("component_failed", False))),
            "status": result.get("status", "completed"),
            "track": "mcp",
            "execution_path": "LangGraph StateGraph: plan -> optional PDF retrieval -> MCP tool discovery/invocation -> response",
            "provider": selected_provider,
            "model": selected_model,
            "sources": [
                {"source": item.get("source", "unknown"), "page": item.get("page", "unknown")}
                for item in result.get("passages", [])
            ],
            "tool_result": result.get("tool_result"),
            "data_view": query_result,
            "tool_calls": result.get("tool_calls", []),
            "trace": result.get("trace", []),
            "timings": timings,
            "error": result.get("error"),
        }
    except Exception as exc:
        elapsed = (perf_counter() - started) * 1000
        return {
            "answer": f"Tool Calling 4 failed before completing the MCP graph: {type(exc).__name__}: {exc}",
            "success": False,
            "status": "failed",
            "track": "mcp",
            "execution_path": "LangGraph StateGraph + MCP initialization/execution failed",
            "provider": selected_provider,
            "model": selected_model,
            "sources": [],
            "tool_result": None,
            "data_view": None,
            "tool_calls": [],
            "trace": [{"step": "LangGraph + MCP workflow", "status": "failed", "detail": str(exc)}],
            "timings": {"retrieval_ms": None, "tool_selection_ms": None, "tool_execution_ms": None, "mcp_discovery_ms": None, "generation_ms": None, "server_total_ms": elapsed},
            "error": str(exc),
        }
