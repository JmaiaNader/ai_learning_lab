def mcp_status():
    """Return the current MCP learning adapter status without contacting a server."""
    return {"configured": False, "message": "MCP client/server wiring is a learning exercise. Configure an approved server, transport, authentication, and compatible SDK before enabling remote tools."}

# Exercise: select an MCP server, use its documented transport, discover tools,
# validate tool names/arguments, and expose only an allowlisted subset.
# Keep credentials outside prompts and require approval for consequential actions.
