"""Regression tests for fast, deterministic Tool Calling 4 analytics responses."""
import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from backend.services import mcp_langgraph_service as service


class MCPAnalyticsFastResponseTests(unittest.TestCase):
    def test_chart_query_routes_to_deterministic_formatter(self):
        state = {
            "selected_sql_tool": service.MCP_TOOL_NAME,
            "should_retrieve": False,
            "tool_result": {
                "tool": service.MCP_TOOL_NAME,
                "requested_view": "chart",
                "query_spec": {
                    "metrics": ["answered"],
                    "dimensions": ["work_date"],
                    "filters": {"channel": "Voice", "market": "Qatar"},
                    "date_preset": "last_7_days",
                },
                "columns": ["work_date", "answered_total"],
                "rows": [
                    {"work_date": f"2026-10-{day:02d}", "answered_total": 100 + day}
                    for day in range(9, 15)
                ],
                "row_count": 6,
                "has_data": True,
            },
        }
        self.assertEqual(service._after_mcp_tool(state), "format_result")

    def test_chart_formatter_does_not_call_llm_and_preserves_success(self):
        state = {
            "tool_result": {
                "requested_view": "chart",
                "query_spec": {
                    "metrics": ["answered"],
                    "dimensions": ["work_date"],
                    "filters": {"channel": "Voice", "market": "Qatar"},
                    "date_preset": "last_7_days",
                },
                "rows": [{"work_date": "2026-10-14", "answered_total": 123}],
                "row_count": 1,
            },
            "timings": {},
            "trace": [],
            "component_failed": False,
        }
        with patch.object(service, "generate", side_effect=AssertionError("LLM should be skipped")):
            response = service._format_mcp_result_node(state)
        self.assertTrue(response["success"])
        self.assertEqual(response["timings"]["generation_ms"], 0.0)
        self.assertIn("1 data point", response["answer"])


class MCPTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_timeout_is_actionable(self):
        class SlowClient:
            async def get_tools(self):
                await asyncio.sleep(1)

        with patch.object(service, "_mcp_client", return_value=SlowClient()), \
             patch.object(service, "MCP_DISCOVERY_TIMEOUT_SECONDS", 0.01):
            with self.assertRaisesRegex(RuntimeError, "discovery timed out"):
                await service._invoke_mcp_tool({})


if __name__ == "__main__":
    unittest.main()
