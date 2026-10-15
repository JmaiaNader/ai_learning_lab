"""Regression tests: surface MCP query failures rather than hiding root cause."""
import unittest
from unittest.mock import patch

from backend.services import mcp_langgraph_service as service


class MCPErrorVisibilityTests(unittest.TestCase):
    def test_tool_error_routes_to_diagnostic_node(self):
        state = {
            "selected_sql_tool": service.MCP_TOOL_NAME,
            "tool_failed": True,
            "tool_result": {"error": "ProgrammingError: Unknown column 'foo'"},
        }
        self.assertEqual(service._after_mcp_tool(state), "format_error")

    def test_diagnostic_contains_underlying_error_and_arguments(self):
        state = {
            "tool_result": {"error": "ProgrammingError: Unknown column 'foo'"},
            "error": "ProgrammingError: Unknown column 'foo'",
            "sql_arguments": {"channel": "Voice", "market": "Qatar", "date_preset": "last_7_days"},
            "trace": [],
            "timings": {},
        }
        result = service._format_mcp_error_node(state)
        self.assertFalse(result["success"])
        self.assertIn("Unknown column", result["answer"])
        self.assertIn("last_7_days", result["trace"][-1]["detail"])

    def test_configured_password_is_redacted(self):
        with patch.object(service.settings, "mysql_password", "secret-db-pass"):
            safe = service._safe_mcp_error("Access denied using secret-db-pass")
        self.assertNotIn("secret-db-pass", safe)
        self.assertIn("[REDACTED]", safe)


if __name__ == "__main__":
    unittest.main()
