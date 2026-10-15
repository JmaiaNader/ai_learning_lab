# Tool Calling 4 — Stage 1

This stage adds the MCP server, the LangGraph+MCP workflow service, and dependencies. It changes at most three project files:
- `requirements.txt`
- `tools/mcp_server.py` (new)
- `backend/services/mcp_langgraph_service.py` (new)

Do not run the MCP server manually; the LangGraph MCP client starts it as a local stdio subprocess. MCP stdout must remain reserved for protocol messages; safe settings diagnostics are redirected to stderr.

Install Stage 1, then apply Stage 2 to enable the `mcp` track in chat.
