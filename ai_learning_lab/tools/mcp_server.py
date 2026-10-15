"""Local stdio MCP server exposing the shared, read-only contact-center query tool.

Run as a subprocess of the LangGraph MCP client:
    python -m tools.mcp_server

Only the validated query function is exposed. Raw SQL is never accepted.
"""
from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from typing import Any

from mcp.server.fastmcp import FastMCP

# config.settings currently prints safe startup diagnostics. MCP stdio reserves
# stdout for protocol messages, so redirect those import-time prints to stderr.
with redirect_stdout(sys.stderr):
    from tools.mysql_tools import query_contact_center_data as _query_contact_center_data

mcp = FastMCP("AI Learning Lab Contact Center Analytics")


@mcp.tool()
def query_contact_center_data(
    metrics: list[str] | None = None,
    dimensions: list[str] | None = None,
    channel: str | None = None,
    market: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    dates: list[str] | None = None,
    date_preset: str | None = None,
    order_by: str | None = None,
    sort_direction: str = "ASC",
    limit: int = 500,
    view_mode: str = "auto",
) -> str:
    """Run a bounded, read-only KPI query on contact_center_demo.

    Allowed metrics: offered, answered, abandoned, sla, aht, asa.
    Allowed grouping dimensions: work_date, interval_start, channel, market,
    week_start, month_start. Filters and all identifiers are validated by the
    existing shared SQL query function. Use dates OR date range OR preset.
    """
    try:
        result: dict[str, Any] = _query_contact_center_data(
            metrics=metrics,
            dimensions=dimensions,
            channel=channel,
            market=market,
            start_date=start_date,
            end_date=end_date,
            dates=dates,
            date_preset=date_preset,
            order_by=order_by,
            sort_direction=sort_direction,
            limit=limit,
            view_mode=view_mode,
        )
    except Exception as exc:
        result = {
            "tool": "query_contact_center_data",
            "has_data": False,
            "rows": [],
            "row_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "message": "The contact-center query failed validation or execution.",
        }
    # JSON string output is portable across MCP clients and retains structured rows.
    return json.dumps(result, ensure_ascii=False, default=str)


if __name__ == "__main__":
    # Do not print banners or diagnostics to stdout with stdio transport.
    mcp.run(transport="stdio")
