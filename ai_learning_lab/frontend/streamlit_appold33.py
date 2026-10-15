import sys
from time import perf_counter
from pathlib import Path
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from config.settings import settings
from backend.services.lesson_service import TRACKS



def render_data_view(data_view):
    """Render generic SQL results according to the user's requested presentation."""
    if not isinstance(data_view, dict) or not data_view.get("tool") == "query_contact_center_data":
        return

    rows = data_view.get("rows") or []
    view_mode = data_view.get("requested_view", "table")
    spec = data_view.get("query_spec") or {}
    if not rows:
        st.info(data_view.get("message") or "No matching data was returned for these filters.")
        return

    try:
        import pandas as pd
    except ImportError:
        st.dataframe(rows, use_container_width=True)
        return

    frame = pd.DataFrame(rows)
    dimensions = list(spec.get("dimensions") or [])
    columns = list(data_view.get("columns") or frame.columns.tolist())
    metric_columns = [
        col for col in columns
        if col not in dimensions and col != "source_interval_count" and col in frame.columns
    ]

    labels = {
        "offered_total": "Offered calls",
        "answered_total": "Answered calls",
        "abandoned_total": "Abandoned calls",
        "sla_percent": "SLA (%)",
        "weighted_avg_aht_seconds": "Weighted AHT (seconds)",
        "weighted_avg_asa_seconds": "Weighted ASA (seconds)",
    }

    if view_mode == "summary":
        if len(frame) == 1:
            row = frame.iloc[0]
            dim_summary = [f"{dim}: {row[dim]}" for dim in dimensions if dim in frame.columns and pd.notna(row[dim])]
            if dim_summary:
                st.caption(" | ".join(dim_summary))
            display_metrics = [m for m in metric_columns if pd.notna(row.get(m))]
            if display_metrics:
                cards = st.columns(min(4, len(display_metrics)))
                for idx, metric in enumerate(display_metrics):
                    value = row[metric]
                    label = labels.get(metric, metric)
                    if metric == "sla_percent":
                        rendered = f"{float(value):.2f}%"
                    elif metric in ("weighted_avg_aht_seconds", "weighted_avg_asa_seconds"):
                        rendered = f"{float(value):,.2f}"
                    elif isinstance(value, (int, float)):
                        rendered = f"{value:,.0f}"
                    else:
                        rendered = str(value)
                    cards[idx % len(cards)].metric(label, rendered)
            else:
                st.write(row.to_dict())
        else:
            st.dataframe(frame, use_container_width=True, hide_index=True)
        return

    if view_mode == "table":
        st.dataframe(frame, use_container_width=True, hide_index=True)
        return

    if view_mode != "chart":
        # Auto mode: a single aggregate is clearer as metrics; grouped results as a table.
        if len(frame) == 1 and not dimensions:
            row = frame.iloc[0]
            visible_metrics = [m for m in metric_columns if pd.notna(row.get(m))]
            if visible_metrics:
                cards = st.columns(min(4, len(visible_metrics)))
                for idx, metric in enumerate(visible_metrics):
                    value = row[metric]
                    rendered = f"{float(value):.2f}%" if metric == "sla_percent" else (f"{float(value):,.2f}" if "seconds" in metric else f"{value:,.0f}" if isinstance(value, (int, float)) else str(value))
                    cards[idx % len(cards)].metric(labels.get(metric, metric), rendered)
            else:
                st.dataframe(frame, use_container_width=True, hide_index=True)
        else:
            st.dataframe(frame, use_container_width=True, hide_index=True)
        return

    # Create a stable x-axis. If date and interval are both present, combine them.
    if "work_date" in frame.columns and "interval_start" in frame.columns:
        frame["_x_axis"] = frame["work_date"].astype(str) + " " + frame["interval_start"].astype(str)
        x_label = "Date / interval"
    elif "work_date" in frame.columns:
        frame["_x_axis"] = frame["work_date"].astype(str)
        x_label = "Date"
    elif "interval_start" in frame.columns:
        frame["_x_axis"] = frame["interval_start"].astype(str)
        x_label = "Interval"
    else:
        x_candidates = [dim for dim in dimensions if dim in frame.columns]
        if not x_candidates:
            st.info("This query returned a single aggregate value, so there is no time/category axis to plot. Ask for daily or interval-level data for a chart.")
            st.dataframe(frame, use_container_width=True, hide_index=True)
            return
        frame["_x_axis"] = frame[x_candidates[0]].astype(str)
        x_label = x_candidates[0].replace("_", " ").title()

    grouping_dims = [dim for dim in dimensions if dim in frame.columns and dim not in ("work_date", "interval_start")]
    if grouping_dims:
        frame["_series"] = frame[grouping_dims].astype(str).agg(" / ".join, axis=1)
    else:
        frame["_series"] = "Value"

    numeric_metrics = []
    for metric in metric_columns:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        if frame[metric].notna().any():
            numeric_metrics.append(metric)
    if not numeric_metrics:
        st.info("The returned rows do not contain numeric metrics to chart.")
        st.dataframe(frame.drop(columns=["_x_axis", "_series"], errors="ignore"), use_container_width=True, hide_index=True)
        return

    st.subheader("Chart from retrieved MySQL data")
    try:
        import altair as alt
    except ImportError:
        st.warning("Altair is not installed, so the chart cannot display point labels. Run `python -m pip install altair`.")
        for metric in numeric_metrics:
            st.caption(labels.get(metric, metric))
            if grouping_dims:
                chart_frame = frame.pivot_table(index="_x_axis", columns="_series", values=metric, aggfunc="first")
            else:
                chart_frame = frame.drop_duplicates(subset=["_x_axis"]).set_index("_x_axis")[[metric]]
            chart_frame = chart_frame.sort_index()
            if not chart_frame.empty and len(chart_frame) >= 2:
                st.line_chart(chart_frame, use_container_width=True)
            elif not chart_frame.empty:
                st.bar_chart(chart_frame, use_container_width=True)
        return

    def format_point_label(metric_name, value):
        if pd.isna(value):
            return ""
        if metric_name == "sla_percent":
            return f"{float(value):.1f}%"
        if metric_name in ("weighted_avg_aht_seconds", "weighted_avg_asa_seconds"):
            return f"{float(value):,.1f}"
        if metric_name in ("offered_total", "answered_total", "abandoned_total"):
            return f"{float(value):,.0f}"
        return f"{float(value):,.2f}" if abs(float(value)) % 1 else f"{float(value):,.0f}"

    st.caption("Point labels show the exact retrieved value; hover over a point for date, series and metric details.")
    for metric in numeric_metrics:
        st.markdown(f"**{labels.get(metric, metric)}**")
        plot_frame = frame[["_x_axis", "_series", metric]].copy()
        plot_frame = plot_frame.rename(columns={metric: "value"})
        plot_frame["value"] = pd.to_numeric(plot_frame["value"], errors="coerce")
        plot_frame = plot_frame.dropna(subset=["value"]).sort_values(["_x_axis", "_series"])
        plot_frame["value_label"] = plot_frame["value"].apply(lambda value: format_point_label(metric, value))
        if plot_frame.empty:
            st.info(f"No plottable values for {metric}.")
            continue
        if len(plot_frame) == 1:
            st.metric(labels.get(metric, metric), plot_frame.iloc[0]["value_label"])
            st.dataframe(plot_frame.rename(columns={"_x_axis": x_label, "_series": "Series", "value": labels.get(metric, metric)})[[x_label, "Series", labels.get(metric, metric)]], use_container_width=True, hide_index=True)
            continue

        x_encoding = alt.X("_x_axis:N", title=x_label, sort=None, axis=alt.Axis(labelAngle=-35, labelOverlap=False))
        y_encoding = alt.Y("value:Q", title=labels.get(metric, metric), scale=alt.Scale(zero=False))
        color_encoding = alt.Color("_series:N", title="Series")
        tooltip_encoding = [
            alt.Tooltip("_x_axis:N", title=x_label),
            alt.Tooltip("_series:N", title="Series"),
            alt.Tooltip("value:Q", title=labels.get(metric, metric), format=",.2f"),
        ]
        base = alt.Chart(plot_frame).encode(
            x=x_encoding,
            y=y_encoding,
            color=color_encoding,
            tooltip=tooltip_encoding,
        )
        lines = base.mark_line(strokeWidth=2.5)
        points = base.mark_point(filled=True, size=75)
        data_labels = alt.Chart(plot_frame).mark_text(
            align="center", baseline="bottom", dy=-9, fontSize=11, fontWeight="bold"
        ).encode(
            x=x_encoding,
            y=y_encoding,
            text=alt.Text("value_label:N"),
            color=alt.Color("_series:N", legend=None),
            tooltip=tooltip_encoding,
        )
        chart = (lines + points + data_labels).properties(height=360).interactive()
        st.altair_chart(chart, use_container_width=True)


def render_response_details(response):
    timings = response.get("timings", {})
    st.caption(
        f"Architecture: {response.get('track', 'N/A')} | "
        f"Path: {response.get('execution_path') or 'N/A'} | "
        f"Model: {response.get('provider', 'N/A')} / {response.get('model') or 'N/A'} | "
        f"Status: {response.get('status', 'N/A')}"
    )
    with st.expander("Performance and execution trace"):
        client_ms = response.get("client_total_ms")
        st.write({
            "End-to-end (UI request, includes API wait)": f"{client_ms:.1f} ms ({client_ms / 1000:.3f} s)" if client_ms is not None else "N/A",
            "Server orchestration": f"{timings['server_total_ms']:.1f} ms" if timings.get("server_total_ms") is not None else "N/A",
            "PDF retrieval": f"{timings['retrieval_ms']:.1f} ms" if timings.get("retrieval_ms") is not None else "N/A",
            "Tool selection": f"{timings['tool_selection_ms']:.1f} ms" if timings.get("tool_selection_ms") is not None else "N/A",
            "Tool execution": f"{timings['tool_execution_ms']:.1f} ms" if timings.get("tool_execution_ms") is not None else "N/A",
            "LLM generation": f"{timings['generation_ms']:.1f} ms" if timings.get("generation_ms") is not None else "N/A",
            "Retrieved passages": len(response.get("sources", [])),
            "Tool calls": len(response.get("tool_calls", [])),
        })
        if response.get("trace"):
            st.markdown("**Steps**")
            st.json(response["trace"])
        for call in response.get("tool_calls", []):
            with st.expander(f"Tool: {call['name']} ({call['status']})"):
                st.write("Arguments", call["arguments"])
                st.write("Execution time", f"{call['duration_ms']:.1f} ms")
                st.write("Result", call["result"])

st.set_page_config(page_title="AI Learning Lab", page_icon="🧠", layout="wide")
st.title("🧠 AI Learning Lab")
st.caption("Learn step by step. In the RAG track, ask questions about your indexed PDF documents.")

by_title = {t["title"]: t for t in TRACKS}
with st.sidebar:
    st.header("Learning menu")
    title = st.radio("Choose a track", list(by_title))
    track = by_title[title]
    st.divider()
    st.subheader("Dynamic LLM")
    provider = st.selectbox("Provider", ["ollama", "openai"], index=0)
    default_model = settings.ollama_model if provider == "ollama" else settings.openai_model
    model = st.text_input("Model", value=default_model)
    st.caption("Ollama starts as the default. OpenAI requires OPENAI_API_KEY in .env.")
    use_rag = st.checkbox("Retrieve PDF context", value=True)
    st.divider()
    st.subheader("PDF knowledge base")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    def post(path, **kwargs):
        try:
            r = requests.post(settings.api_base_url + path, timeout=180, **kwargs)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            st.error(f"API request failed. Start backend with `uvicorn backend.main:app --reload --port 8000`. Details: {e}")
            return None
    if st.button("Upload and index PDF", disabled=uploaded is None, use_container_width=True):
        result = post("/upload-pdf", files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")})
        if result: st.success(f"Indexed {result.get('indexed_chunks', 0)} chunks.")
    if st.button("Index PDFs in data/pdfs", use_container_width=True):
        result = post("/index")
        if result: st.json(result)

st.header(track["title"])
st.write(track["goal"])
lessons_tab, code_tab, chat_tab = st.tabs(["📚 Training lessons", "💻 Code & definitions", "💬 AI chat"])

with lessons_tab:
    for lesson in track["lessons"]:
        with st.expander(lesson["title"]):
            st.markdown("**Definition**")
            st.write(lesson["definition"])
            st.markdown("**Step-by-step**")
            for step in lesson["steps"]: st.markdown(f"- {step}")
            st.caption(f"Code file: `{lesson['file']}`")
            if st.button("Ask tutor about this lesson", key="ask_" + track["id"] + lesson["title"]):
                st.session_state["question_" + track["id"]] = f"Teach me {lesson['title']} step by step. Explain the definition and walk through the code."
                st.session_state["switch_chat"] = True

with code_tab:
    st.markdown("""
| Topic | File | Responsibility |
|---|---|---|
| Frontend | `frontend/streamlit_app.py` | Menu, lessons, chat, provider selection |
| API | `backend/main.py` | FastAPI routes |
| LLM adapter | `backend/llm/provider.py` | Ollama/OpenAI abstraction |
| PDF RAG | `backend/rag/pdf_indexer.py` | Extract, chunk, index, retrieve |
| Chat orchestration | `backend/services/chat_service.py` | Combines prompt, retrieval, and tools |
| DB config | `config/database.py` | MySQL connection |
| SQL tools | `tools/mysql_tools.py` | Approved read-only queries |
| Tool allowlist | `tools/registry.py` | Registered tools |
| MCP | `tools/mcp_tools.py` | MCP learning placeholder |
""")
    for lesson in track["lessons"]:
        st.markdown("**" + lesson["title"] + " — definition**")
        st.write(lesson["definition"])
        st.code("\n".join(lesson["steps"]), language="text")
    st.info("Trace a request through Streamlit → FastAPI → service → LLM/tool → response.")

with chat_tab:
    history_key = "history_" + track["id"]
    history = st.session_state.setdefault(history_key, [])
    for msg in history:
        with st.chat_message(msg["role"]):
            response = msg.get("response") or {}
            data_view = response.get("data_view") or {}
            compact_summary = (
                msg["role"] == "assistant"
                and data_view.get("requested_view") == "summary"
                and len(data_view.get("rows") or []) == 1
            )
            if not compact_summary:
                st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption("Sources: " + ", ".join(f"{s['source']} p.{s['page']}" for s in msg["sources"]))
            if msg["role"] == "assistant" and msg.get("response"):
                render_data_view(data_view)
                render_response_details(response)
    placeholder = "Ask about your PDF / HR policy" if track["id"] == "rag" else "Ask a question for this architecture"
    question = st.chat_input(placeholder)
    pending_question_key = "question_" + track["id"]
    if st.session_state.get(pending_question_key):
        question = st.session_state.pop(pending_question_key)
    if question:
        history.append({"role":"user", "content":question})
        with st.chat_message("user"): st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner(f"Using {provider} / {model}..."):
                request_started = perf_counter()
                result = post("/chat", json={"message":question, "track":track["id"], "provider":provider, "model":model, "use_rag":use_rag})
                client_total_ms = (perf_counter() - request_started) * 1000
            if result is None:
                result = {"answer": "The API request failed; see the error above.", "success": False, "status": "request_failed", "track": track["id"], "provider": provider, "model": model, "timings": {}, "sources": [], "tool_calls": [], "trace": []}
            result["client_total_ms"] = client_total_ms
            data_view = result.get("data_view") or {}
            compact_summary = (
                data_view.get("requested_view") == "summary"
                and len(data_view.get("rows") or []) == 1
            )
            if not compact_summary:
                st.markdown(result["answer"])
            render_data_view(data_view)
            if result.get("sources"):
                st.caption("PDF sources: " + ", ".join(f"{s['source']} p.{s['page']}" for s in result["sources"]))
            render_response_details(result)
            history.append({"role":"assistant", "content":result["answer"], "sources":result.get("sources", []), "response":result})
    if st.button("Clear chat", key="clear_" + track["id"]): st.session_state[history_key] = []; st.rerun()
    if st.button("Quiz me", key="quiz_" + track["id"]): st.session_state[pending_question_key] = f"Quiz me on {track['title']}. Ask one question and wait for my answer."; st.rerun()

with st.expander("Troubleshooting"):
    st.write(f"API URL: `{settings.api_base_url}`")
    if st.button("Check API health"):
        try: st.json(requests.get(settings.api_base_url + "/health", timeout=5).json())
        except requests.RequestException as e: st.error(str(e))
    st.markdown("- Backend: `uvicorn backend.main:app --reload --port 8000`")
    st.markdown("- Frontend: `streamlit run frontend/streamlit_app.py`")
    st.markdown("- Ollama: start the service and pull the selected model.")
    st.markdown("- MySQL: check `.env`, database access, and exact schema names.")
