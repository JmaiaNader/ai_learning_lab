"""Regression tests for MCP result types returned by MCP client adapters."""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

# Keep the fixtures compatible with MCP SDK versions where content blocks are
# Pydantic objects, as well as adapter versions that return dictionaries.
from backend.services import mcp_langgraph_service as service


class MCPResultListDecodeTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "tool": "query_contact_center_data",
            "has_data": True,
            "row_count": 1,
            "columns": ["work_date", "answered"],
            "rows": [{"work_date": "2026-10-14", "answered": 123}],
            "requested_view": "chart",
        }

    def test_decodes_list_of_mcp_text_content_objects(self):
        raw = [SimpleNamespace(type="text", text=json.dumps(self.payload))]
        decoded = service._decode_tool_result(raw)
        self.assertTrue(decoded["has_data"])
        self.assertEqual(decoded["rows"][0]["answered"], 123)

    def test_decodes_list_of_mcp_text_dicts(self):
        raw = [{"type": "text", "text": json.dumps(self.payload)}]
        decoded = service._decode_tool_result(raw)
        self.assertTrue(decoded["has_data"])
        self.assertEqual(decoded["requested_view"], "chart")

    def test_decodes_call_tool_result_with_content_list(self):
        raw = SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(self.payload))])
        decoded = service._decode_tool_result(raw)
        self.assertEqual(decoded["row_count"], 1)

    def test_preserves_direct_structured_payload(self):
        decoded = service._decode_tool_result(self.payload)
        self.assertEqual(decoded["rows"], self.payload["rows"])

    def test_unexpected_list_returns_actionable_error(self):
        decoded = service._decode_tool_result([object()])
        self.assertFalse(decoded["has_data"])
        self.assertIn("list", decoded["error"].lower())


if __name__ == "__main__":
    unittest.main()
