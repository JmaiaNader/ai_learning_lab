# Tool Calling 4 — Stage 2

This stage enables `track="mcp"` in chat and adds the verification tests. It changes three files:
- `backend/services/chat_service.py`
- `tools/mcp_tools.py` (status now describes the configured local MCP path; it does not fake a live connection)
- `tests/test_mcp_langgraph_service.py` (new)

After Stage 1 and Stage 2 are installed, run `python -m pip install -r requirements.txt` and `python -m unittest discover -s tests -p "test_mcp_langgraph_service.py" -v` from the project root. The tests mock the MCP boundary so they do not require live MySQL or Ollama. Then send a real query in Tool Calling 4 to verify the configured `.env`, MySQL, and Ollama connection.
