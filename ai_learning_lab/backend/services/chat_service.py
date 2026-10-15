from time import perf_counter

from backend.llm.provider import generate
from backend.rag.pdf_indexer import retrieve
from backend.services.lesson_service import TRACKS
from tools.registry import available_tools, run_tool
from config.settings import settings


def _retrieval_query(message: str, track: str) -> str:
    """Expand short policy questions with common synonyms without forcing HR terminology elsewhere."""
    lower = message.lower()
    if track in ("rag", "mysql_rag") and any(word in lower for word in ("leave", "leaves", "vacation", "holiday", "sick")):
        return message + " HR human resources annual leave sick leave entitlement eligibility request approval"
    return message


def _is_policy_question(message: str) -> bool:
    lower = message.lower()
    return any(word in lower for word in (
        "policy", "policies", "check-in", "check in", "leave", "leaves",
        "vacation", "holiday", "sick", "entitlement", "eligibility", "approval",
    ))


def _is_call_volume_question(message: str) -> bool:
    words = set(message.lower().replace("-", " ").replace("?", " ").replace("_", " ").split())
    return bool(words.intersection({"call", "calls", "contact", "contacts"}) and words.intersection({
        "volume", "offered", "total", "count", "received", "handled", "number", "many",
    }))


def _extract_sql_filters(message: str, selected_sql_tool=None) -> dict:
    """Extract allowlisted channel/market/date filters for the selected SQL tool."""
    lower = message.lower()
    args = {}
    for value in ("voice", "chat", "email"):
        if value in lower:
            args["channel"] = value.title()
            break
    market_aliases = (
        (("qatar",), "Qatar"),
        (("uae", "united arab emirates", "emirates"), "UAE"),
        (("saudi arabia", "saudi", "ksa"), "Saudi Arabia"),
    )
    for aliases, canonical in market_aliases:
        if any(alias in lower for alias in aliases):
            args["market"] = canonical
            break

    # Only the interval-extreme tool accepts date_scope. Do not pass it to tools
    # whose function signatures use start_date/end_date.
    if selected_sql_tool == "lowest_sla_interval":
        if "yesterday" in lower or "previous day" in lower:
            args["date_scope"] = "yesterday"
        elif "today" in lower:
            args["date_scope"] = "today"
    return args


def _select_sql_tool(message: str):
    """Choose an existing read-only SQL tool based on clear contact-center data intent."""
    lower = message.lower()
    if _is_call_volume_question(message):
        return "total_calls_offered"

    # Interval-level extrema need their own SQL tool. A market summary is aggregated
    # across intervals and cannot answer which market/interval had the lowest SLA.
    lower_words = set(lower.replace("-", " ").replace("?", " ").replace("_", " ").split())
    asks_lowest_sla = (
        ("sla" in lower or "service level" in lower)
        and bool(lower_words.intersection({"lowest", "minimum", "min", "worst", "smallest", "least"}))
    )
    if asks_lowest_sla:
        return "lowest_sla_interval"

    data_terms = (
        "sla", "service level", "aht", "average handling", "asa", "speed of answer",
        "abandon", "answered", "kpi", "contact center data", "contact centre data",
        "market summary", "by market", "by channel", "daily summary", "show data",
        "performance metrics", "contact center metrics", "contact centre metrics",
    )
    if not any(term in lower for term in data_terms):
        return None
    if any(term in lower for term in ("daily", "by day", "per day", "each day", "daily summary")):
        return "daily_summary"
    return "market_summary"


def answer_question(message, track, provider, model, use_rag=True):
    # Tool Calling 4 uses a separate LangGraph workflow whose analytics tool is
    # discovered and invoked through an MCP client/server boundary.
    if track == "mcp":
        try:
            from backend.services.mcp_langgraph_service import run_mcp_langgraph_workflow
            return run_mcp_langgraph_workflow(message, provider, model, use_rag)
        except ImportError as exc:
            selected_provider = (provider or "ollama").lower()
            selected_model = model or (
                settings.ollama_model if selected_provider == "ollama" else settings.openai_model
            )
            return {
                "answer": "Tool Calling 4 needs LangGraph and MCP dependencies. Run `python -m pip install -r requirements.txt` in the project folder, then restart the backend. Details: " + str(exc),
                "success": False,
                "status": "missing_dependency",
                "track": "mcp",
                "execution_path": "LangGraph + MCP dependencies missing; MCP server was not called",
                "provider": selected_provider,
                "model": selected_model,
                "sources": [],
                "tool_result": None,
                "data_view": None,
                "tool_calls": [],
                "trace": [{"step": "LangGraph + MCP initialization", "status": "failed", "detail": str(exc)}],
                "timings": {"retrieval_ms": None, "tool_selection_ms": None, "tool_execution_ms": None, "mcp_discovery_ms": None, "generation_ms": None, "server_total_ms": 0.0},
                "error": str(exc),
            }

    # Tool Calling 2 uses its own compiled StateGraph workflow. Import lazily so
    # Basic Chat and the existing RAG/MySQL tracks remain usable independently.
    if track == "langgraph":
        try:
            from ai_learning_lab.backend.services.langgraph_service_OLD import run_langgraph_workflow
            return run_langgraph_workflow(message, provider, model, use_rag)
        except ImportError as exc:
            selected_provider = (provider or "ollama").lower()
            selected_model = model or (
                settings.ollama_model if selected_provider == "ollama" else settings.openai_model
            )
            return {
                "answer": "LangGraph is not installed in this Python environment. Run `python -m pip install -r requirements.txt` in the project folder, then restart the backend. Details: " + str(exc),
                "success": False,
                "status": "missing_dependency",
                "track": "langgraph",
                "execution_path": "LangGraph dependency missing; graph was not executed",
                "provider": selected_provider,
                "model": selected_model,
                "sources": [],
                "tool_result": None,
                "tool_calls": [],
                "trace": [{"step": "LangGraph initialization", "status": "failed", "detail": str(exc)}],
                "timings": {"retrieval_ms": None, "tool_selection_ms": None, "tool_execution_ms": None, "generation_ms": None, "server_total_ms": 0.0},
                "error": str(exc),
            }

    started = perf_counter()
    timings = {
        "retrieval_ms": None,
        "tool_selection_ms": None,
        "tool_execution_ms": None,
        "generation_ms": None,
        "server_total_ms": None,
    }
    trace = []
    passages = []
    tool_result = None
    tool_calls = []
    answer = ""
    error = None
    success = True
    component_failed = False
    status = "completed"
    selected_provider = (provider or "ollama").lower()
    selected_model = model or (
        settings.ollama_model if selected_provider == "ollama" else settings.openai_model
    )
    track_ids = {item["id"] for item in TRACKS}
    path_by_track = {
        "basic_llm": "direct prompt -> configured LLM",
        "rag": "PDF retrieval -> grounded prompt -> configured LLM",
        "mysql_rag": "keyword routing -> optional PDF retrieval and/or contact-center MySQL -> synthesis prompt -> configured LLM",
    }
    execution_path = path_by_track.get(track)

    try:
        if track not in track_ids:
            success = False
            status = "invalid_track"
            answer = f"Unknown learning track: {track}"
        elif track not in path_by_track:
            success = False
            status = "not_implemented"
            answer = (
                f"{next(item['title'] for item in TRACKS if item['id'] == track)} "
                "is a learning roadmap only; this architecture does not execute in chat yet."
            )
            execution_path = "not implemented; no LLM, graph, or MCP call made"
        else:
            policy_question = _is_policy_question(message)
            selection_started = perf_counter()
            selected_sql_tool = _select_sql_tool(message) if track == "mysql_rag" else None
            should_retrieve = track == "rag" or (
                track == "mysql_rag" and use_rag and policy_question
            )
            timings["tool_selection_ms"] = (perf_counter() - selection_started) * 1000 if track == "mysql_rag" else None

            if track == "mysql_rag":
                trace.append({
                    "step": "keyword tool routing",
                    "status": "selected" if selected_sql_tool else "skipped",
                    "detail": f"Selected {selected_sql_tool}" if selected_sql_tool else "No contact-center SQL intent matched",
                    "duration_ms": timings["tool_selection_ms"],
                })

            if should_retrieve:
                retrieval_started = perf_counter()
                passages = retrieve(_retrieval_query(message, track))
                timings["retrieval_ms"] = (perf_counter() - retrieval_started) * 1000
                trace.append({
                    "step": "PDF retrieval",
                    "status": "completed" if passages else "no_passages",
                    "detail": f"Retrieved {len(passages)} passage(s)",
                    "duration_ms": timings["retrieval_ms"],
                })

            if selected_sql_tool:
                tool_started = perf_counter()
                tool_arguments = _extract_sql_filters(message, selected_sql_tool)
                try:
                    tool_result = run_tool(selected_sql_tool, tool_arguments)
                    tool_status = "completed"
                except Exception as exc:
                    tool_result = f"Database tool unavailable: {exc}"
                    tool_status = "failed"
                    component_failed = True
                timings["tool_execution_ms"] = (perf_counter() - tool_started) * 1000
                tool_calls.append({
                    "name": selected_sql_tool,
                    "arguments": tool_arguments,
                    "result": tool_result,
                    "status": tool_status,
                    "duration_ms": timings["tool_execution_ms"],
                })
                trace.append({
                    "step": "MySQL tool execution",
                    "status": tool_status,
                    "detail": f"Executed {selected_sql_tool} against contact_center_demo",
                    "duration_ms": timings["tool_execution_ms"],
                })

            context = "\n\n".join(
                f"[Source: {item['source']}, page {item['page']}]\n{item['text']}"
                for item in passages
            )

            if track == "rag" and not passages:
                answer = (
                    "I couldn't find a relevant passage in the indexed PDF documents for that question. "
                    "Please confirm that the correct PDF was indexed, then try a specific question."
                )
            else:
                if track == "basic_llm":
                    prompt = f"Answer the user's question directly and clearly.\n\nQuestion: {message}"
                elif track == "rag":
                    prompt = f"""Answer the question using only the retrieved PDF passages. Do not invent document facts. If the passages do not support an answer, say so. Cite source filename and page.

Retrieved passages:
{context}

Question: {message}"""
                else:
                    prompt = f"""Answer the user's exact question clearly. Use PDF passages only for document/policy facts and SQL results only for contact-center data. Do not invent figures. The database table is first_application_ai.contact_center_demo. SLA is answered_in_threshold / answered; SLA results are supplied as a percentage when the field is sla_percent. AHT and ASA are weighted by answered contacts. State the date range and filters when present. If no SQL result was retrieved, say that the database was not queried. forecast_history retrieves historical observations only; it does not generate predictions.

Available read-only tools: {available_tools()}
Database tool selected: {selected_sql_tool or '(none)'}
Database result: {tool_result}
Retrieved PDF passages:
{context or "(No relevant indexed PDF passages were retrieved.)"}

Question: {message}"""

                generation_started = perf_counter()
                answer = generate(prompt, selected_provider, selected_model)
                timings["generation_ms"] = (perf_counter() - generation_started) * 1000
                trace.append({
                    "step": "LLM generation",
                    "status": "completed",
                    "detail": f"{selected_provider} / {selected_model}",
                    "duration_ms": timings["generation_ms"],
                })

    except Exception as exc:
        success = False
        status = "failed"
        error = str(exc)
        answer = f"The request could not be completed: {error}"
        trace.append({"step": "request", "status": "failed", "detail": error})
    finally:
        timings["server_total_ms"] = (perf_counter() - started) * 1000

    if component_failed and success:
        success = False
        status = "partial_failure"

    return {
        "answer": answer,
        "success": success,
        "status": status,
        "track": track,
        "execution_path": execution_path,
        "provider": selected_provider,
        "model": selected_model,
        "sources": [{"source": item["source"], "page": item["page"]} for item in passages],
        "tool_result": tool_result,
        "tool_calls": tool_calls,
        "trace": trace,
        "timings": timings,
        "error": error,
    }
