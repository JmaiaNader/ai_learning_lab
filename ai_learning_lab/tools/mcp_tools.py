"""Status helpers for the local MCP analytics integration."""
from pathlib import Path


def mcp_status():
    """Describe the configured local MCP boundary used by Tool Calling 4.

    This reports configuration only. A successful tool discovery/invocation in the
    Tool Calling 4 execution trace is the actual runtime health check.
    """
    project_root = Path(__file__).resolve().parent.parent
    server_path = project_root / "tools" / "mcp_server.py"
    configured = server_path.is_file()
    return {
        "configured": configured,
        "connected": None,
        "transport": "stdio",
        "server": "tools.mcp_server",
        "tool": "query_contact_center_data",
        "message": (
            "The local MCP server is configured and is started on demand by the Tool Calling 4 LangGraph workflow. "
            "Use the execution trace to confirm live discovery and tool invocation."
            if configured else
            "The MCP server file is missing. Install the Tool Calling 4 patch and restart the backend."
        ),
    }
