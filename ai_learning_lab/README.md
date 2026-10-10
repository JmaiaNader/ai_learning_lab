# AI Learning Lab

Four Streamlit learning tracks:
1. RAG application (PDFs)
2. Tool Calling 1 — RAG + MySQL
3. Tool Calling 2 — LangGraph + Tools
4. Tool Calling 4 — LangGraph + Tools + MCP

Each track includes training lessons, code locations/definitions, and an AI tutor chat. The default provider is Ollama; OpenAI is also supported through a separate provider adapter.

## Run in VS Code
1. Extract this ZIP and open the folder.
2. Create and activate a virtual environment:
   - Windows: `py -m venv .venv` then `.venv\Scripts\Activate.ps1`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
3. Install: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env`.
5. Install/start Ollama, then run `ollama pull llama3.2`.
6. Terminal 1: `uvicorn backend.main:app --reload --port 8000`
7. Terminal 2: `streamlit run frontend/streamlit_app.py`

UI: http://localhost:8501 | API health: http://localhost:8000/health

## PDFs
Upload a PDF in the sidebar or place PDFs in `data/pdfs/`, then click Index PDFs. Text is chunked and stored in local ChromaDB at `data/chroma/`.

## MySQL schema from your screenshots
Database: `first_application_ai`
- `agentdetails`: agentid, name, site, supervisorname, officename, managername, active_status, doj, resignation_date, seperation_type, team, primary_language, nationality
- `aht_agent`: id, date, agentID, aht, handled, hold, acw, channel, site
- `marketleveldataset`: id, offered, handled, aht, sla, language, market, date, interval, channel

Set MYSQL_* variables in `.env`. Sample tools use read-only aggregate queries and an allowlist. Review before production use.

## Architecture
`frontend/` Streamlit UI; `backend/` FastAPI, LLM, RAG, services; `config/` app/database/provider settings; `tools/` isolated tool implementations and registry. The LangGraph and MCP tracks are guided learning scaffolds, not a production-ready connected MCP deployment. Add authentication, authorization, secrets management and stricter database controls before exposing this app.
