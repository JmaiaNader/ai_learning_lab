"""LangGraph workflow for Tool Calling 2.

The workflow uses deterministic, allowlisted routing for reliable execution:
plan -> optional PDF retrieval -> optional read-only SQL tool -> LLM synthesis.
It does not execute model-generated SQL.
"""
from __future__ import annotations

import json
import re
from time import perf_counter
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.llm.provider import generate
from backend.rag.pdf_indexer import retrieve
from config.settings import settings
from tools.registry import available_tools, run_tool


class LangGraphState(TypedDict, total=False):
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


def _is_policy_question(message: str) -> bool:
    lower = message.lower()
    return any(term in lower for term in (
        "policy", "policies", "check-in", "check in", "leave", "leaves",
        "vacation", "holiday", "sick", "entitlement", "eligibility", "approval",
    ))


def _is_call_volume_question(message: str) -> bool:
    words = set(message.lower().replace("-", " ").replace("?", " ").replace("_", " ").split())
    return bool(words.intersection({"call", "calls", "contact", "contacts"}) and words.intersection({
        "volume", "offered", "total", "count", "received", "handled", "number", "many",
    }))


def _has_contact_center_intent(message: str) -> bool:
    lower = message.lower()
    data_terms = (
        "sla", "service level", "aht", "average handling", "handling time",
        "asa", "speed of answer", "abandon", "answered", "offered", "calls",
        "contacts", "kpi", "contact center", "contact centre", "by market",
        "by channel", "daily summary", "performance metrics", "forecast",
        "trend", "interval", "volume",
    )
    return any(term in lower for term in data_terms)


def _select_sql_tool(message: str) -> str | None:
    """Route contact-center analytics to one generic, read-only data tool."""
    lower = message.lower()
    words = set(lower.replace("-", " ").replace("?", " ").replace("_", " ").split())

    # These tools serve nonstandard data sources/operations rather than the KPI table.
    if "agent" in lower and words.intersection({"count", "number", "active", "many", "total"}):
        return "agent_count"
    if any(term in lower for term in (
        "forecast history", "historical data for forecasting", "history for forecasting",
        "historical interval data", "time series history",
    )):
        return "forecast_history"

    if _has_contact_center_intent(message) or _is_call_volume_question(message):
        return "query_contact_center_data"
    return None


def _explicit_dates(message: str) -> list[str]:
    """Parse ISO and month-name dates from a question; no SQL is generated from text."""
    from datetime import date

    found: list[str] = []
    for value in re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", message):
        try:
            found.append(date.fromisoformat(value).isoformat())
        except ValueError:
            continue

    month_re = re.compile(
        r"\b(January|Jan|February|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|August|Aug|September|Sept|Sep|October|Oct|November|Nov|December|Dec)"
        r"\s+(\d{1,2})(?:,?\s+(20\d{2}))?\b",
        re.IGNORECASE,
    )
    current_year = datetime.now(ZoneInfo("Asia/Qatar")).year
    for match in month_re.finditer(message):
        token = f"{match.group(1)} {match.group(2)} {match.group(3) or current_year}"
        parsed = None
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                parsed = datetime.strptime(token, fmt).date()
                break
            except ValueError:
                pass
        if parsed:
            found.append(parsed.isoformat())
    return list(dict.fromkeys(found))


def _extract_sql_arguments(message: str, tool_name: str) -> dict[str, Any]:
    """Turn natural-language intent into a bounded query specification.

    Only the query specification is derived from the question. The actual SQL,
    identifiers, metrics and dimensions are built from allowlists in mysql_tools.py.
    """
    lower = message.lower()
    args: dict[str, Any] = {}

    for value in ("voice", "chat", "email"):
        if re.search(rf"\b{re.escape(value)}\b", lower):
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

    if tool_name != "query_contact_center_data":
        if tool_name == "forecast_history":
            args["days"] = 14
        return args

    # Requested metrics. These are semantic aliases, not raw SQL fragments.
    metrics = []
    if re.search(r"\b(sla|service level)\b", lower):
        metrics.append("sla")
    if any(term in lower for term in ("aht", "average handling", "handling time", "average handle time")):
        metrics.append("aht")
    if any(term in lower for term in ("asa", "speed of answer", "average speed of answer")):
        metrics.append("asa")
    if any(term in lower for term in ("abandon", "abandoned")):
        metrics.append("abandoned")
    if any(term in lower for term in ("offered", "call volume", "contact volume", "calls offered")):
        metrics.append("offered")
    if re.search(r"\b(answered|answered calls|handled contacts)\b", lower):
        metrics.append("answered")
    if not metrics:
        metrics = ["offered", "answered", "abandoned", "sla", "aht", "asa"]
    # Summary requests for one KPI commonly benefit from denominators/context.
    if "aht" in metrics and any(word in lower for word in ("summary", "compare", "comparison")):
        metrics.extend(item for item in ("answered", "offered") if item not in metrics)
    metrics = list(dict.fromkeys(metrics))
    args["metrics"] = metrics

    # Date interpretation is independent of metric/tool selection.
    date_words = re.sub(r"\s+", " ", lower)
    explicit = _explicit_dates(message)
    is_range = bool(re.search(r"\b(from|between)\b", lower) and re.search(r"\b(to|and)\b", lower))
    has_same_weekday_reference = any(
        phrase in lower for phrase in ("last week same day", "same weekday last week", "same day last week", "previous week same day")
    )
    if ("yesterday" in lower or "previous day" in lower) and has_same_weekday_reference:
        args["date_preset"] = "yesterday_and_last_week_same_day"
    elif has_same_weekday_reference:
        args["date_preset"] = "same_weekday_last_week"
    elif re.search(r"\b(last|past)\s+(7|seven)\s+days?\b", date_words):
        args["date_preset"] = "last_7_days"
    elif re.search(r"\b(last|past)\s+(14|fourteen)\s+days?\b", date_words):
        args["date_preset"] = "last_14_days"
    elif re.search(r"\b(last|past)\s+(30|thirty)\s+days?\b", date_words):
        args["date_preset"] = "last_30_days"
    elif re.search(r"\blast week\b", lower):
        args["date_preset"] = "last_week"
    elif re.search(r"\bthis week\b", lower):
        args["date_preset"] = "this_week"
    elif "yesterday" in lower or "previous day" in lower:
        args["date_preset"] = "yesterday"
    elif re.search(r"\btoday\b", lower):
        args["date_preset"] = "today"
    elif len(explicit) >= 2 and is_range:
        args["start_date"] = min(explicit)
        args["end_date"] = max(explicit)
    elif explicit:
        args["dates"] = explicit

    # Decide what rows should represent. A requested filter is not a grouping dimension.
    dimensions = []
    asks_single_extreme = bool(set(lower.split()).intersection({"lowest", "minimum", "min", "worst", "smallest", "least", "highest", "maximum", "max", "best"}))
    is_trend = any(term in lower for term in ("trend", "trending", "over time", "pattern", "graph", "chart", "plot"))
    interval_level = any(term in lower for term in ("interval", "half-hour", "half hour", "hourly", "throughout the day"))
    daily_level = any(term in lower for term in ("daily", "per day", "each day", "by day", "day by day"))
    weekly_level = any(term in lower for term in ("weekly", "per week", "each week", "by week", "week by week"))
    monthly_level = any(term in lower for term in ("monthly", "per month", "each month", "by month", "month by month"))
    multiple_dates = bool(args.get("dates") and len(args["dates"]) > 1) or bool(args.get("date_preset") == "yesterday_and_last_week_same_day")

    if interval_level or (is_trend and args.get("date_preset") == "yesterday"):
        dimensions.append("interval_start")
        single_date_presets = {"today", "yesterday", "same_weekday_last_week"}
        if "work_date" not in dimensions and (args.get("date_preset") not in single_date_presets or multiple_dates):
            dimensions.insert(0, "work_date")
        # If user asks which market, the result must retain market as a row dimension.
        if not args.get("market") or "which market" in lower or "by market" in lower or "across markets" in lower:
            dimensions.append("market")
        if not args.get("channel") or "by channel" in lower or "across channels" in lower:
            dimensions.append("channel")
    elif weekly_level:
        dimensions.append("week_start")
        if not args.get("market") or "by market" in lower or "across markets" in lower or "which market" in lower:
            dimensions.append("market")
        if not args.get("channel") or "by channel" in lower or "across channels" in lower:
            dimensions.append("channel")
    elif monthly_level:
        dimensions.append("month_start")
        if not args.get("market") or "by market" in lower or "across markets" in lower or "which market" in lower:
            dimensions.append("market")
        if not args.get("channel") or "by channel" in lower or "across channels" in lower:
            dimensions.append("channel")
    elif daily_level or is_trend or multiple_dates:
        dimensions.append("work_date")
        if not args.get("market") or "by market" in lower or "across markets" in lower or "which market" in lower:
            dimensions.append("market")
        if not args.get("channel") or "by channel" in lower or "across channels" in lower:
            dimensions.append("channel")
    else:
        if "by market" in lower or "across markets" in lower or "which market" in lower:
            dimensions.append("market")
        if "by channel" in lower or "across channels" in lower:
            dimensions.append("channel")
        if asks_single_extreme and ("interval" in lower or "which interval" in lower):
            dimensions.extend(item for item in ("work_date", "interval_start") if item not in dimensions)
            if "which market" in lower and "market" not in dimensions:
                dimensions.append("market")
    # An extreme (lowest/highest/worst/best) must compare detailed groups rather
    # than ordering one already-aggregated grand total. Include interval grain.
    if asks_single_extreme:
        dimensions.extend(item for item in ("work_date", "interval_start") if item not in dimensions)
        if (not args.get("market") or "which market" in lower or "by market" in lower or "across markets" in lower) and "market" not in dimensions:
            dimensions.append("market")
        if (not args.get("channel") or "by channel" in lower or "across channels" in lower) and "channel" not in dimensions:
            dimensions.append("channel")
    args["dimensions"] = list(dict.fromkeys(dimensions))

    # Extremes are performed by SQL over the exact metric, not guessed by the LLM.
    requested_metric = metrics[0]
    if asks_single_extreme:
        args["order_by"] = requested_metric
        args["sort_direction"] = "ASC" if set(lower.split()).intersection({"lowest", "minimum", "min", "worst", "smallest", "least"}) else "DESC"
        args["limit"] = 1

    if any(term in lower for term in ("graph", "chart", "plot")) or is_trend:
        args["view_mode"] = "chart"
    elif any(term in lower for term in ("table", "tabular", "show rows")):
        args["view_mode"] = "table"
    elif any(term in lower for term in ("only result", "just the result", "just the number", "only the number", "only answer", "just answer")) or asks_single_extreme:
        args["view_mode"] = "summary"
    else:
        args["view_mode"] = "auto"
    return args

def _with_trace(state: LangGraphState, entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [*state.get("trace", []), entry]


def _plan_node(state: LangGraphState) -> dict[str, Any]:
    started = perf_counter()
    message = state["message"]
    selected_tool = _select_sql_tool(message)
    policy_question = _is_policy_question(message)
    should_retrieve = bool(state.get("use_rag", True) and policy_question)

    if selected_tool and should_retrieve:
        intent = "sql_and_pdf"
    elif selected_tool:
        intent = "sql_tool"
    elif should_retrieve:
        intent = "pdf_retrieval"
    else:
        intent = "llm_only"

    elapsed = (perf_counter() - started) * 1000
    timings = dict(state.get("timings", {}))
    timings["tool_selection_ms"] = elapsed
    return {
        "selected_sql_tool": selected_tool,
        "sql_arguments": _extract_sql_arguments(message, selected_tool) if selected_tool else {},
        "should_retrieve": should_retrieve,
        "passages": [],
        "tool_result": None,
        "tool_calls": [],
        "trace": _with_trace(state, {
            "step": "LangGraph planner node",
            "status": "selected",
            "detail": f"Intent={intent}; SQL tool={selected_tool or 'none'}; PDF retrieval={'yes' if should_retrieve else 'no'}",
            "duration_ms": elapsed,
        }),
        "timings": timings,
        "component_failed": False,
        "tool_failed": False,
        "error": None,
    }


def _after_plan(state: LangGraphState) -> str:
    if state.get("should_retrieve"):
        return "retrieve"
    if state.get("selected_sql_tool"):
        return "execute_tool"
    return "generate"


_PDF_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "based", "can", "could", "document",
    "documents", "for", "from", "give", "help", "how", "i", "in", "is", "it",
    "me", "my", "of", "on", "or", "our", "please", "policy", "policies", "pdf",
    "relevant", "rule", "rules", "summarize", "tell", "that", "the", "their", "them",
    "there", "these", "this", "to", "using", "what", "which", "with", "would", "you",
    "your", "applicable", "requirement", "requirements", "explain", "describe", "about",
}

_PDF_TERM_ALIASES = {
    "call": {"call", "caller", "contact", "telephone", "phone", "queue"},
    "handling": {"handling", "handle", "handled", "response", "respond", "answer", "support", "service", "wait", "waiting", "resolution", "resolve", "agent", "queue"},
    "service": {"service", "support", "sla", "response", "answer", "target", "threshold", "within", "wait", "waiting"},
    "level": {"level", "target", "threshold", "percentage", "percent", "within", "time", "second", "minute", "sla"},
    "refund": {"refund", "refundable", "refunded", "reimburse", "reimbursement"},
    "change": {"change", "changes", "changed", "amend", "amendment"},
    "ticket": {"ticket", "tickets", "fare", "fares", "booking", "bookings"},
    "leave": {"leave", "vacation", "holiday", "absence"},
    "entitlement": {"entitlement", "entitled", "allowance", "quota"},
}


def _normalise_pdf_token(token: str) -> str:
    token = token.lower().strip("-")
    aliases = {
        "calls": "call", "callers": "caller", "contacts": "contact", "tickets": "ticket",
        "policies": "policy", "requirements": "requirement", "answered": "answer",
        "answers": "answer", "handling": "handling", "handled": "handling",
        "refundable": "refundable", "refunds": "refund", "changes": "change",
        "minutes": "minute", "seconds": "second", "hours": "hour", "leaves": "leave",
        "entitlements": "entitlement", "classes": "class", "responses": "response",
    }
    return aliases.get(token, token)


def _policy_retrieval_query(message: str) -> str:
    """Remove a trailing analytics/chart request before querying the PDF index."""
    split_pattern = r"(?i)\b(?:also\s+)?(?:show|plot|display|chart|graph|compare|calculate|retrieve)\b"
    for match in re.finditer(split_pattern, message):
        prefix = message[:match.start()].strip(" \t\r\n.,;:-")
        suffix = message[match.start():]
        prefix = re.sub(r"(?i)\b(?:and\s+)?also\s*$", "", prefix).strip(" \t\r\n.,;:-")
        if prefix and _is_policy_question(prefix) and _has_contact_center_intent(suffix):
            return prefix
    return message.strip()


def _passage_relevant_to_policy_query(question: str, passage_text: str) -> bool:
    """A conservative lexical evidence check after vector retrieval.

    It is intentionally a guard against obviously off-topic chunks, not a replacement
    for semantic retrieval. If a query has no useful content terms, trust Chroma's rank.
    """
    question_tokens = {
        _normalise_pdf_token(token)
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9]*", question.lower())
        if _normalise_pdf_token(token) not in _PDF_QUERY_STOPWORDS
        and len(_normalise_pdf_token(token)) > 2
    }
    if not question_tokens:
        return True

    passage_tokens = {
        _normalise_pdf_token(token)
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9]*", passage_text.lower())
    }
    matched_terms = 0
    for term in question_tokens:
        aliases = _PDF_TERM_ALIASES.get(term, {term})
        if aliases.intersection(passage_tokens):
            matched_terms += 1

    # For multi-concept questions such as "call handling / service-level", require
    # evidence for at least two requested concepts. This avoids presenting ticket/fare
    # rules as call-center SLA policy merely because both come from an airline PDF.
    required_matches = 2 if len(question_tokens) >= 2 else 1
    return matched_terms >= required_matches


def _filter_relevant_pdf_passages(question: str, passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        passage for passage in passages
        if _passage_relevant_to_policy_query(question, str(passage.get("text", "")))
    ]


def _retrieve_node(state: LangGraphState) -> dict[str, Any]:
    started = perf_counter()
    message = state["message"]
    retrieval_query = _policy_retrieval_query(message)
    try:
        candidates = retrieve(retrieval_query)
        passages = _filter_relevant_pdf_passages(retrieval_query, candidates)
        status = "completed" if passages else ("no_relevant_passages" if candidates else "no_passages")
        detail = (
            f"Retrieved {len(candidates)} candidate PDF passage(s); kept {len(passages)} relevant passage(s) "
            f"for policy query: {retrieval_query[:180]}"
        )
        error = None
        failed = False
    except Exception as exc:
        passages = []
        candidates = []
        status = "failed"
        detail = f"PDF retrieval failed: {exc}"
        error = str(exc)
        failed = True

    elapsed = (perf_counter() - started) * 1000
    timings = dict(state.get("timings", {}))
    timings["retrieval_ms"] = elapsed
    trace = _with_trace(state, {
        "step": "PDF retrieval node",
        "status": status,
        "detail": detail,
        "duration_ms": elapsed,
    })
    return {
        "passages": passages,
        "trace": trace,
        "timings": timings,
        "component_failed": bool(state.get("component_failed") or failed),
        "error": error or state.get("error"),
    }


def _after_retrieval(state: LangGraphState) -> str:
    if state.get("selected_sql_tool"):
        return "execute_tool"
    return "generate"


def _execute_tool_node(state: LangGraphState) -> dict[str, Any]:
    started = perf_counter()
    tool_name = state["selected_sql_tool"]
    arguments = state.get("sql_arguments", {})
    try:
        result = run_tool(tool_name, arguments)
        tool_status = "completed"
        detail = f"Executed allowlisted tool '{tool_name}'"
        error = None
        failed = False
    except Exception as exc:
        result = {"error": str(exc), "message": "The database tool failed."}
        tool_status = "failed"
        detail = f"Tool '{tool_name}' failed: {exc}"
        error = str(exc)
        failed = True

    elapsed = (perf_counter() - started) * 1000
    timings = dict(state.get("timings", {}))
    timings["tool_execution_ms"] = elapsed
    call = {
        "name": tool_name,
        "arguments": arguments,
        "result": result,
        "status": tool_status,
        "duration_ms": elapsed,
    }
    trace = _with_trace(state, {
        "step": "Read-only SQL tool node",
        "status": tool_status,
        "detail": detail,
        "duration_ms": elapsed,
    })
    return {
        "tool_result": result,
        "tool_calls": [*state.get("tool_calls", []), call],
        "trace": trace,
        "timings": timings,
        "component_failed": bool(state.get("component_failed") or failed),
        "tool_failed": failed,
        "error": error or state.get("error"),
    }


def _format_daily_aht_comparison(result: Any) -> str:
    """Format the verified SQL comparison without asking an LLM to infer dates/numbers."""
    if not isinstance(result, dict):
        return "The AHT comparison tool returned an unexpected result format; check the SQL tool trace."

    expected = result.get("expected_dates") or {}
    comparisons = result.get("comparisons") or []
    filters = result.get("filters") or {}
    channel = filters.get("channel") or "all channels"
    market = filters.get("market") or "all markets"
    if not comparisons:
        return (
            f"No contact-center rows were found for {channel} in {market} on either comparison date. "
            f"Expected dates: yesterday={expected.get('yesterday', 'unknown')}; "
            f"same weekday last week={expected.get('same_weekday_last_week', 'unknown')}. "
            "The SQL lookup completed but returned no matching records."
        )

    lines = [f"Daily weighted AHT comparison — {channel}, {market}"]
    for item in comparisons:
        item_channel = item.get("channel") or channel
        item_market = item.get("market") or market
        if len(comparisons) > 1:
            lines.append(f"\n{item_channel} — {item_market}")
        yesterday = item.get("yesterday")
        previous = item.get("same_weekday_last_week")
        for label, row, expected_date in (
            ("Yesterday", yesterday, expected.get("yesterday")),
            ("Same weekday last week", previous, expected.get("same_weekday_last_week")),
        ):
            if row is None:
                lines.append(f"- {label} ({expected_date or 'expected date unknown'}): no matching rows returned.")
                continue
            aht = row.get("weighted_avg_aht_seconds")
            if aht is None:
                aht_text = "not calculable (no answered contacts)"
            else:
                aht_text = f"{float(aht):.2f} seconds"
            lines.append(
                f"- {label} ({row.get('date', expected_date)}): weighted average AHT = {aht_text}; "
                f"answered = {row.get('answered_total', 0):,}; "
                f"offered = {row.get('offered_total', 0):,}; "
                f"intervals = {row.get('interval_count', 0)}."
            )

        current_aht = yesterday.get("weighted_avg_aht_seconds") if yesterday else None
        previous_aht = previous.get("weighted_avg_aht_seconds") if previous else None
        if current_aht is not None and previous_aht is not None:
            change = item.get("aht_change_seconds")
            change_pct = item.get("aht_change_percent")
            direction = "increased" if change > 0 else "decreased" if change < 0 else "did not change"
            if change_pct is None:
                lines.append(f"- Difference: AHT {direction} by {abs(change):.2f} seconds; percentage change unavailable.")
            else:
                lines.append(
                    f"- Difference: AHT {direction} by {abs(change):.2f} seconds "
                    f"({abs(change_pct):.2f}% relative to last week's value)."
                )
        elif yesterday is not None or previous is not None:
            missing = item.get("missing_periods") or []
            labels = {
                "yesterday": f"yesterday ({expected.get('yesterday', 'unknown')})",
                "same_weekday_last_week": f"same weekday last week ({expected.get('same_weekday_last_week', 'unknown')})",
            }
            lines.append(
                "- Partial comparison: the available period above is reported; missing data for "
                + ", ".join(labels.get(name, name) for name in missing)
                + "."
            )
    return "\n".join(lines)


def _format_generic_summary(result: Any) -> str:
    """Render a concise direct answer for requests that ask for the result only."""
    if not isinstance(result, dict):
        return "The contact-center query returned an unexpected result format."
    rows = result.get("rows") or []
    if not rows:
        return result.get("message") or "No rows matched the selected dates and filters."
    if len(rows) > 1:
        return f"Retrieved {len(rows)} matching rows. The full result is displayed below."
    row = rows[0]
    labels = {
        "work_date": "Date",
        "interval_start": "Interval",
        "channel": "Channel",
        "market": "Market",
        "offered_total": "Offered calls",
        "answered_total": "Answered calls",
        "abandoned_total": "Abandoned calls",
        "sla_percent": "SLA",
        "weighted_avg_aht_seconds": "Weighted AHT",
        "weighted_avg_asa_seconds": "Weighted ASA",
        "source_interval_count": "Source intervals",
    }
    pieces = []
    for key in result.get("columns", []):
        if key not in row or key == "source_interval_count":
            continue
        value = row.get(key)
        if value is None:
            shown = "not calculable"
        elif key == "sla_percent":
            shown = f"{float(value):.2f}%"
        elif key in ("weighted_avg_aht_seconds", "weighted_avg_asa_seconds"):
            shown = f"{float(value):.2f} seconds"
        else:
            shown = str(value)
        pieces.append(f"{labels.get(key, key)}: {shown}")
    return " | ".join(pieces) if pieces else "The query completed, but there were no displayable metric values."


def _generate_node(state: LangGraphState) -> dict[str, Any]:
    timings = dict(state.get("timings", {}))
    message = state["message"]
    passages = state.get("passages", [])
    selected_tool = state.get("selected_sql_tool")
    tool_result = state.get("tool_result")
    context = "\n\n".join(
        f"[Source: {item.get('source', 'unknown')}, page {item.get('page', 'unknown')}]\n{item.get('text', '')}"
        for item in passages
    )

    if state.get("tool_failed") and selected_tool:
        answer = (
            f"The SQL tool '{selected_tool}' failed, so I cannot provide a verified database answer. "
            "Check the database connection and tool error in the execution trace."
        )
        timings["generation_ms"] = None
        generation_trace = {
            "step": "Response node",
            "status": "skipped_after_tool_failure",
            "detail": "Returned a safe failure response; no unsupported database facts were generated.",
            "duration_ms": 0.0,
        }
    elif state.get("should_retrieve") and selected_tool and not passages:
        query_result = tool_result if isinstance(tool_result, dict) else {}
        row_count = int(query_result.get("row_count", len(query_result.get("rows") or [])) or 0)
        answer = (
            "I retrieved the SQL analytics result, but the indexed PDFs did not return passages relevant to the policy part "
            "of your question. I won't summarize unrelated document text as policy evidence. "
            "The SQL result is still available below. "
            f"The database returned {row_count} row(s); any chart or table uses those verified rows only."
        )
        timings["generation_ms"] = 0.0
        generation_trace = {
            "step": "Policy evidence guard",
            "status": "no_relevant_policy_evidence",
            "detail": "SQL branch is preserved, but no LLM policy summary was generated because no relevant PDF passages survived the relevance check.",
            "duration_ms": 0.0,
        }
    elif selected_tool == "compare_daily_aht":
        answer = _format_daily_aht_comparison(tool_result)
        timings["generation_ms"] = 0.0
        generation_trace = {
            "step": "Deterministic KPI comparison formatter",
            "status": "completed",
            "detail": "Formatted database-returned AHT values and partial results directly; no LLM date or metric inference.",
            "duration_ms": 0.0,
        }
    elif selected_tool == "query_contact_center_data" and isinstance(tool_result, dict) and tool_result.get("requested_view") == "summary":
        answer = _format_generic_summary(tool_result)
        timings["generation_ms"] = 0.0
        generation_trace = {
            "step": "Deterministic result formatter",
            "status": "completed",
            "detail": "Presented verified SQL values directly; the LLM did not calculate or infer metric values.",
            "duration_ms": 0.0,
        }
    elif state.get("should_retrieve") and not passages and not selected_tool:
        answer = (
            "I couldn't find relevant passages in the indexed PDF documents for this question. "
            "Check that the intended PDF files have been indexed, then try a more specific question."
        )
        timings["generation_ms"] = None
        generation_trace = {
            "step": "Response node",
            "status": "no_evidence",
            "detail": "No relevant PDF passages were retrieved; the model was not asked to invent an answer.",
            "duration_ms": 0.0,
        }
    else:
        provider = state["provider"]
        model = state["model"]
        if selected_tool:
            prompt_result = tool_result
            if selected_tool == "query_contact_center_data" and isinstance(tool_result, dict):
                # Keep model context bounded. The full rows stay in `data_view` for the UI.
                prompt_result = dict(tool_result)
                all_rows = tool_result.get("rows") or []
                prompt_result["rows"] = all_rows[:25]
                prompt_result["rows_in_prompt"] = len(prompt_result["rows"])
                prompt_result["total_rows_available"] = len(all_rows)
            prompt = f"""You are the response node of a LangGraph workflow. Answer the user's exact question using the verified SQL result below, not guesses. Never invent metrics, dates, intervals or markets. If `has_data` is false, say no matching rows were returned and state which filters were applied. SLA is answered_in_threshold / answered; `sla_percent` is already a percentage. AHT and ASA are weighted averages when provided. The structured result is also displayed separately as the requested table/chart when relevant, so keep your explanation concise. For a trend, describe only visible direction or comparisons supported by the returned rows; do not claim statistical significance. The database is first_application_ai.contact_center_demo.

Available analytics capability: {available_tools()}
Selected tool: {selected_tool}
Tool arguments: {json.dumps(state.get('sql_arguments', {}), default=str)}
SQL result (first 25 rows maximum): {json.dumps(prompt_result, default=str)}
PDF context (use only for policy/document claims): {context or '(no PDF passages)'}

User question: {message}"""
        elif passages:
            prompt = f"""Answer the question using only the retrieved PDF passages. Do not invent document facts. Cite the source filename and page. If the passages do not support an answer, say so.

Retrieved passages:
{context}

Question: {message}"""
        else:
            prompt = f"Answer the user's question directly and clearly. Do not claim that you queried tools or documents unless the execution trace shows that they ran.\n\nQuestion: {message}"

        started = perf_counter()
        try:
            answer = generate(prompt, state["provider"], state["model"])
            elapsed = (perf_counter() - started) * 1000
            timings["generation_ms"] = elapsed
            generation_trace = {
                "step": "LLM response node",
                "status": "completed",
                "detail": f"{provider} / {model} synthesized the graph result",
                "duration_ms": elapsed,
            }
        except Exception as exc:
            answer = f"The LangGraph workflow completed its retrieval/tool steps, but response generation failed: {exc}"
            elapsed = (perf_counter() - started) * 1000
            timings["generation_ms"] = elapsed
            generation_trace = {
                "step": "LLM response node",
                "status": "failed",
                "detail": str(exc),
                "duration_ms": elapsed,
            }
            return {
                "answer": answer,
                "trace": _with_trace(state, generation_trace),
                "timings": timings,
                "status": "partial_failure",
                "success": False,
                "error": str(exc),
            }

    failed = bool(state.get("component_failed"))
    return {
        "answer": answer,
        "trace": _with_trace(state, generation_trace),
        "timings": timings,
        "status": "partial_failure" if failed else "completed",
        "success": not failed,
        "error": state.get("error"),
    }


def _build_graph():
    graph = StateGraph(LangGraphState)
    graph.add_node("plan", _plan_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("execute_tool", _execute_tool_node)
    graph.add_node("generate", _generate_node)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan",
        _after_plan,
        {"retrieve": "retrieve", "execute_tool": "execute_tool", "generate": "generate"},
    )
    graph.add_conditional_edges(
        "retrieve",
        _after_retrieval,
        {"execute_tool": "execute_tool", "generate": "generate"},
    )
    graph.add_edge("execute_tool", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


# Compile once at import; each request gets a fresh state.
_LANGGRAPH_APP = _build_graph()


def run_langgraph_workflow(message: str, provider: str, model: str | None, use_rag: bool = True) -> dict[str, Any]:
    """Run the actual LangGraph graph and return the same shape as other chat tracks."""
    started = perf_counter()
    selected_provider = (provider or "ollama").lower()
    selected_model = model or (
        settings.ollama_model if selected_provider == "ollama" else settings.openai_model
    )
    initial_state: LangGraphState = {
        "message": message,
        "provider": selected_provider,
        "model": selected_model,
        "use_rag": use_rag,
        "trace": [],
        "timings": {
            "retrieval_ms": None,
            "tool_selection_ms": None,
            "tool_execution_ms": None,
            "generation_ms": None,
            "server_total_ms": None,
        },
        "tool_calls": [],
        "passages": [],
        "component_failed": False,
        "tool_failed": False,
    }
    try:
        result = _LANGGRAPH_APP.invoke(initial_state)
        timings = dict(result.get("timings", {}))
        timings["server_total_ms"] = (perf_counter() - started) * 1000
        generic_tool_result = result.get("tool_result") if result.get("selected_sql_tool") == "query_contact_center_data" else None
        return {
            "answer": result.get("answer", "The LangGraph workflow returned no answer."),
            "success": bool(result.get("success", True)),
            "status": result.get("status", "completed"),
            "track": "langgraph",
            "execution_path": "LangGraph StateGraph: plan -> conditional PDF retrieval / allowlisted SQL tool -> response (LLM synthesis or deterministic KPI formatter)",
            "provider": selected_provider,
            "model": selected_model,
            "sources": [
                {"source": item.get("source", "unknown"), "page": item.get("page", "unknown")}
                for item in result.get("passages", [])
            ],
            "tool_result": result.get("tool_result"),
            "data_view": generic_tool_result,
            "tool_calls": result.get("tool_calls", []),
            "trace": result.get("trace", []),
            "timings": timings,
            "error": result.get("error"),
        }
    except Exception as exc:
        elapsed = (perf_counter() - started) * 1000
        return {
            "answer": f"The LangGraph workflow failed: {exc}",
            "success": False,
            "status": "failed",
            "track": "langgraph",
            "execution_path": "LangGraph StateGraph initialization or execution failed",
            "provider": selected_provider,
            "model": selected_model,
            "sources": [],
            "tool_result": None,
            "tool_calls": [],
            "trace": [{"step": "LangGraph workflow", "status": "failed", "detail": str(exc)}],
            "timings": {
                "retrieval_ms": None,
                "tool_selection_ms": None,
                "tool_execution_ms": None,
                "generation_ms": None,
                "server_total_ms": elapsed,
            },
            "error": str(exc),
        }
