"""Allowlisted tool registry.

LangGraph's contact-center analytics uses one generic query tool. Older named
functions remain registered for the existing direct Tool Calling 1 track and
backward compatibility, but they are not advertised as separate analytics tools.
"""
from tools.mysql_tools import (
    agent_count,
    compare_daily_aht,
    daily_summary,
    forecast_history,
    lowest_sla_interval,
    market_summary,
    query_contact_center_data,
    sla_trend,
    total_calls_offered,
)
from tools.mcp_tools import mcp_status

# These specialized functions are retained to avoid breaking the current direct
# SQL track. New LangGraph contact-center requests should always use the generic tool.
LEGACY_TOOLS = {
    "market_summary": market_summary,
    "compare_daily_aht": compare_daily_aht,
    "lowest_sla_interval": lowest_sla_interval,
    "sla_trend": sla_trend,
    "daily_summary": daily_summary,
    "total_calls_offered": total_calls_offered,
}

TOOLS = {
    "query_contact_center_data": query_contact_center_data,
    "forecast_history": forecast_history,
    "agent_count": agent_count,
    "mcp_status": mcp_status,
    **LEGACY_TOOLS,
}

# Advertise one generic contact-center query capability to new agent planners.
# This avoids encouraging per-question tool selection while old routes remain usable.
PUBLIC_TOOLS = {
    "query_contact_center_data": query_contact_center_data,
    "forecast_history": forecast_history,
    "agent_count": agent_count,
    "mcp_status": mcp_status,
}


def available_tools():
    return [{"name": name, "description": fn.__doc__ or ""} for name, fn in PUBLIC_TOOLS.items()]


def run_tool(name, arguments):
    if name not in TOOLS:
        raise ValueError(f"Tool '{name}' is not allowlisted.")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise TypeError("Tool arguments must be a dictionary.")
    return TOOLS[name](**arguments)
