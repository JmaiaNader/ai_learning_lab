import sys
from time import perf_counter
from pathlib import Path
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from config.settings import settings
from backend.services.lesson_service import TRACKS


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
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption("Sources: " + ", ".join(f"{s['source']} p.{s['page']}" for s in msg["sources"]))
            if msg["role"] == "assistant" and msg.get("response"):
                render_response_details(msg["response"])
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
            st.markdown(result["answer"])
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
