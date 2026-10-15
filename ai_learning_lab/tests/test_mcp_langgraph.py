import unittest
from unittest.mock import patch

from backend.workflows import mcp_langgraph


class McpLangGraphTests(unittest.TestCase):
    def test_hybrid_question_routes_to_both_allowlisted_tools(self):
        calls = []

        def fake_call(selected):
            calls.extend(selected)
            return [
                {
                    "name": item["name"],
                    "arguments": item["arguments"],
                    "result": (
                        {"query": "policy", "passages": [{"source": "hr.pdf", "page": 2, "text": "Policy."}]}
                        if item["name"] == "search_hr_policy"
                        else {"total_calls_offered": 42}
                    ),
                    "status": "completed",
                    "duration_ms": 1.0,
                }
                for item in selected
            ], 0.5

        with patch.object(mcp_langgraph, "_call_mcp_tools", side_effect=fake_call), patch.object(
            mcp_langgraph, "generate", return_value="Combined answer"
        ):
            result = mcp_langgraph.run_mcp_workflow(
                "What is the check-in policy and total offered call count?", "ollama", "mock"
            )

        self.assertEqual(
            [item["name"] for item in calls],
            ["search_hr_policy", "get_total_calls_offered"],
        )
        self.assertEqual(result["sources"], [{"source": "hr.pdf", "page": 2}])
        self.assertEqual(result["tool_result"]["get_total_calls_offered"]["total_calls_offered"], 42)
        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["timings"]["graph_total_ms"], 0)
        self.assertEqual(len(result["trace"]), 3)

    def test_tool_error_is_returned_as_partial_failure(self):
        selected = [{"name": "get_total_calls_offered", "arguments": {}}]
        with patch.object(
            mcp_langgraph,
            "_call_mcp_tools",
            return_value=([{
                "name": selected[0]["name"],
                "arguments": {},
                "result": "database unavailable",
                "status": "failed",
                "duration_ms": 1.0,
            }], 0.5),
        ), patch.object(mcp_langgraph, "generate", return_value="Database tool failed."):
            result = mcp_langgraph.run_mcp_workflow(
                "What is the total offered call count?", "ollama", "mock"
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "partial_failure")
        self.assertIn("database unavailable", result["error"])
        self.assertEqual(result["tool_calls"][0]["status"], "failed")

    def test_unrelated_question_does_not_select_tools(self):
        with patch.object(mcp_langgraph, "_call_mcp_tools") as call_tools, patch.object(
            mcp_langgraph, "generate", return_value="Direct explanation"
        ):
            result = mcp_langgraph.run_mcp_workflow(
                "Explain what a language model is.", "ollama", "mock"
            )

        call_tools.assert_not_called()
        self.assertEqual(result["tool_calls"], [])
        self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()