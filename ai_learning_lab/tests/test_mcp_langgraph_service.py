"""Mocked tests for the real Tool Calling 4 LangGraph route and MCP execution node.

These tests do not require a live MySQL server or Ollama. The local MCP server is
verified separately by running the chat track with the project .env configured.
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from backend.services import mcp_langgraph_service as service


class FakeMCPTool:
    name = "query_contact_center_data"

    async def ainvoke(self, arguments):
        self.arguments = arguments
        return json.dumps({
            "tool": "query_contact_center_data",
            "query_spec": {"metrics": ["answered"], "dimensions": [], "view_mode": "summary"},
            "columns": ["answered_total", "source_interval_count"],
            "rows": [{"answered_total": 1234, "source_interval_count": 144}],
            "row_count": 1,
            "has_data": True,
            "message": None,
            "requested_view": "summary",
        })


class FakeMCPClient:
    def __init__(self, connections):
        self.connections = connections

    async def get_tools(self):
        return [FakeMCPTool()]


class ToolCalling4Tests(unittest.TestCase):
    @patch.object(service, "MultiServerMCPClient", FakeMCPClient)
    def test_mcp_graph_discovers_and_invokes_generic_query_tool(self):
        result = service.run_mcp_langgraph_workflow(
            "What is the total answered yesterday? Give only the result.",
            "ollama",
            "llama3.2",
            use_rag=False,
        )
        self.assertEqual(result["track"], "mcp")
        self.assertTrue(result["success"])
        self.assertEqual(result["data_view"]["rows"][0]["answered_total"], 1234)
        self.assertTrue(any(item["step"] == "MCP server discovery" for item in result["trace"]))
        self.assertTrue(any(item["step"] == "MCP tool invocation" for item in result["trace"]))
        self.assertEqual(result["tool_calls"][0]["name"], "query_contact_center_data")

    def test_mcp_result_decoder_restores_structured_json(self):
        decoded = service._decode_tool_result('{"has_data":true,"rows":[{"answered_total":7}]}')
        self.assertTrue(decoded["has_data"])
        self.assertEqual(decoded["rows"][0]["answered_total"], 7)

    def test_generic_query_arguments_are_derived_from_user_question(self):
        from ai_learning_lab.backend.services.langgraph_service_OLD import _extract_sql_arguments, _select_sql_tool
        question = "Show the last 7 days of total answered calls for Voice in Qatar as a line chart with data labels."
        selected = _select_sql_tool(question)
        self.assertEqual(selected, "query_contact_center_data")
        args = _extract_sql_arguments(question, selected)
        self.assertEqual(args.get("channel"), "Voice")
        self.assertEqual(args.get("market"), "Qatar")
        self.assertEqual(args.get("date_preset"), "last_7_days")
        self.assertEqual(args.get("view_mode"), "chart")


if __name__ == "__main__":
    unittest.main()
