from time import perf_counter

from backend.llm.provider import generate
from backend.rag.pdf_indexer import retrieve
from backend.services.lesson_service import TRACKS
from tools.registry import available_tools, run_tool
from config.settings import settings


def _retrieval_query(message: str, track: str) -> str:
    """Expand short HR leave questions with common policy synonyms."""
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
    words = set(message.lower().replace("-", " ").replace("?", " ").split())
    return bool(words.intersection({"call", "calls"}) and words.intersection({
        "volume", "offered", "total", "count", "received", "handled", "number", "many",
    }))


def answer_question(message, track, provider, model, use_rag=True):
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
        "mysql_rag": "keyword routing -> optional PDF retrieval and/or MySQL -> synthesis prompt -> configured LLM",
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
            call_volume_question = _is_call_volume_question(message)
            should_retrieve = track == "rag" or (
                track == "mysql_rag" and use_rag and policy_question
            )
            should_query_calls = track == "mysql_rag" and call_volume_question

            if track == "mysql_rag":
                timings["tool_selection_ms"] = (perf_counter() - selection_started) * 1000
                trace.append({
                    "step": "keyword tool routing",
                    "status": "selected" if should_query_calls else "skipped",
                    "detail": "Selected total_calls_offered" if should_query_calls else "No call-volume tool trigger matched",
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

            if should_query_calls:
                tool_started = perf_counter()
                try:
                    tool_result = run_tool("total_calls_offered", {})
                    tool_status = "completed"
                except Exception as exc:
                    tool_result = f"Database tool unavailable: {exc}"
                    tool_status = "failed"
                    component_failed = True
                timings["tool_execution_ms"] = (perf_counter() - tool_started) * 1000
                tool_calls.append({
                    "name": "total_calls_offered",
                    "arguments": {},
                    "result": tool_result,
                    "status": tool_status,
                    "duration_ms": timings["tool_execution_ms"],
                })
                trace.append({
                    "step": "MySQL tool execution",
                    "status": tool_status,
                    "detail": "Executed total_calls_offered",
                    "duration_ms": timings["tool_execution_ms"],
                })

            sources = [
                {"source": item["source"], "page": item["page"]}
                for item in passages
            ]
            context = "\n\n".join(
                f"[Source: {item['source']}, page {item['page']}]\n{item['text']}"
                for item in passages
            )

            if track == "rag" and not passages:
                answer = (
                    "I couldn't find a relevant passage in the indexed PDF documents for that question. "
                    "Please confirm that the correct PDF was indexed, then try a specific policy question."
                )
            else:
                if track == "basic_llm":
                    prompt = f"Answer the user's question directly and clearly.\n\nQuestion: {message}"
                elif track == "rag":
                    prompt = f"""Answer the question using only the retrieved PDF passages. Do not invent policy facts. If the passages do not support an answer, say so. Cite source filename and page.

Retrieved passages:
{context}

Question: {message}"""
                else:
                    prompt = f"""Answer the user's exact question. Use PDF passages for policy facts and the database result for call-volume facts. Keep both evidence sources distinct; do not invent facts.

Available read-only tools: {available_tools()}
Database tool result: {tool_result}
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
        "sources": [
            {"source": item["source"], "page": item["page"]}
            for item in passages
        ],
        "tool_result": tool_result,
        "tool_calls": tool_calls,
        "trace": trace,
        "timings": timings,
        "error": error,
    }
