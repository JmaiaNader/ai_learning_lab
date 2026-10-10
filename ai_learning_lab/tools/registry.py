from tools.mysql_tools import market_summary, agent_count, total_calls_offered
from tools.mcp_tools import mcp_status
TOOLS = {"market_summary": market_summary, "agent_count": agent_count, "total_calls_offered": total_calls_offered, "mcp_status": mcp_status}

def available_tools():
    return [{"name": n, "description": fn.__doc__ or ""} for n, fn in TOOLS.items()]

def run_tool(name, arguments):
    if name not in TOOLS:
        raise ValueError(f"Tool '{name}' is not allowlisted.")
    return TOOLS[name](**arguments)
