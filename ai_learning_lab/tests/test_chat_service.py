import unittest
from unittest.mock import patch

from backend.services import chat_service
from ai_learning_lab.backend.services import langgraph_service_OLD


class AnswerQuestionTests(unittest.TestCase):
    def setUp(self):
        self.retrieve_patcher = patch.object(
            chat_service,
            "retrieve",
            return_value=[{"source": "policy.pdf", "page": 3, "text": "Policy evidence."}],
        )
        self.tool_patcher = patch.object(
            chat_service, "run_tool", return_value={"total_calls_offered": 123}
        )
        self.generate_patcher = patch.object(
            chat_service, "generate", return_value="mock answer"
        )
        self.retrieve = self.retrieve_patcher.start()
        self.run_tool = self.tool_patcher.start()
        self.generate = self.generate_patcher.start()
        self.addCleanup(self.retrieve_patcher.stop)
        self.addCleanup(self.tool_patcher.stop)
        self.addCleanup(self.generate_patcher.stop)

    def test_basic_llm_skips_retrieval_and_tools(self):
        result = chat_service.answer_question(
            "Explain an LLM", "basic_llm", "ollama", "mock-model"
        )

        self.assertTrue(result["success"])
        self.retrieve.assert_not_called()
        self.run_tool.assert_not_called()
        self.assertIn("direct prompt", result["execution_path"])
        self.assertGreaterEqual(result["timings"]["generation_ms"], 0)

    def test_rag_retrieves_and_returns_sources(self):
        result = chat_service.answer_question(
            "What is the check-in policy?", "rag", "ollama", "mock-model"
        )

        self.retrieve.assert_called_once()
        self.run_tool.assert_not_called()
        self.assertEqual(result["sources"], [{"source": "policy.pdf", "page": 3}])
        self.assertGreaterEqual(result["timings"]["retrieval_ms"], 0)

    def test_hybrid_question_runs_retrieval_and_mysql_tool(self):
        result = chat_service.answer_question(
            "What is the check-in policy and total offered call count?",
            "mysql_rag",
            "ollama",
            "mock-model",
        )

        self.retrieve.assert_called_once()
        self.run_tool.assert_called_once_with("total_calls_offered", {})
        self.assertEqual(result["tool_result"], {"total_calls_offered": 123})
        self.assertGreaterEqual(result["timings"]["tool_execution_ms"], 0)

    def test_langgraph_track_invokes_graph_workflow_and_mcp_remains_a_placeholder(self):
        graph_response = {
            "answer": "mock LangGraph answer",
            "success": True,
            "status": "completed",
            "track": "langgraph",
            "execution_path": "LangGraph StateGraph: plan -> tool -> response",
            "provider": "ollama",
            "model": "mock-model",
            "sources": [],
            "tool_result": {"total_calls_offered": 123},
            "tool_calls": [{"name": "total_calls_offered", "status": "completed", "arguments": {}}],
            "trace": [{"step": "LangGraph planner node", "status": "selected"}],
            "timings": {"server_total_ms": 1.0},
            "error": None,
        }
        with patch("backend.services.langgraph_service.run_langgraph_workflow", return_value=graph_response) as graph_run:
            result = chat_service.answer_question(
                "What is total offered call count?", "langgraph", "ollama", "mock-model"
            )
            graph_run.assert_called_once_with(
                "What is total offered call count?", "ollama", "mock-model", True
            )
            self.assertEqual(result["status"], "completed")
            self.assertIn("LangGraph StateGraph", result["execution_path"])
            self.assertEqual(result["tool_calls"][0]["name"], "total_calls_offered")

        mcp_result = chat_service.answer_question(
            "Explain this architecture", "mcp", "ollama", "mock-model"
        )
        self.assertEqual(mcp_result["status"], "not_implemented")
        self.generate.assert_not_called()

    def test_generation_failure_reports_failure_and_elapsed_time(self):
        self.generate.side_effect = RuntimeError("mock provider failure")

        result = chat_service.answer_question(
            "Explain an LLM", "basic_llm", "ollama", "mock-model"
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "failed")
        self.assertGreaterEqual(result["timings"]["server_total_ms"], 0)

    def test_database_failure_is_not_reported_as_full_success(self):
        self.run_tool.side_effect = RuntimeError("mock database failure")

        result = chat_service.answer_question(
            "What is the total offered call count?",
            "mysql_rag",
            "ollama",
            "mock-model",
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "partial_failure")
        self.assertEqual(result["tool_calls"][0]["status"], "failed")


class LangGraphWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tool_patcher = patch.object(
            langgraph_service_OLD,
            "run_tool",
            return_value={
                "work_date": "2026-10-14",
                "interval_start": "10:30:00",
                "channel": "Voice",
                "market": "UAE",
                "sla_percent": 12.5,
            },
        )
        self.generate_patcher = patch.object(
            langgraph_service_OLD, "generate", return_value="Lowest SLA was 12.5% in UAE at 10:30."
        )
        self.retrieve_patcher = patch.object(langgraph_service_OLD, "retrieve", return_value=[])
        self.run_tool = self.tool_patcher.start()
        self.generate = self.generate_patcher.start()
        self.retrieve = self.retrieve_patcher.start()
        self.addCleanup(self.tool_patcher.stop)
        self.addCleanup(self.generate_patcher.stop)
        self.addCleanup(self.retrieve_patcher.stop)

    def test_lowest_sla_yesterday_uses_interval_tool_through_real_graph(self):
        result = chat_service.answer_question(
            "Give me the lowest SLA for Voice in which market and which interval for yesterday",
            "langgraph", "ollama", "mock-model", False,
        )
        self.run_tool.assert_called_once_with(
            "lowest_sla_interval", {"channel": "Voice", "date_scope": "yesterday"}
        )
        self.generate.assert_called_once()
        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["tool_calls"][0]["status"], "completed")
        self.assertIn("LangGraph StateGraph", result["execution_path"])
        self.assertTrue(any(item["step"] == "LangGraph planner node" for item in result["trace"]))

    def test_tool_failure_returns_safe_error_without_fabricated_answer(self):
        self.run_tool.side_effect = RuntimeError("database offline")
        result = chat_service.answer_question(
            "Show SLA by market", "langgraph", "ollama", "mock-model", False
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "partial_failure")
        self.generate.assert_not_called()
        self.assertIn("cannot provide a verified database answer", result["answer"])

    def test_general_question_uses_graph_without_calling_sql(self):
        result = chat_service.answer_question(
            "Explain what a contact center is", "langgraph", "ollama", "mock-model", False
        )
        self.run_tool.assert_not_called()
        self.generate.assert_called_once()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["tool_calls"], [])

    def test_yesterday_filter_is_added_to_regular_summary_tool(self):
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        args = langgraph_service_OLD._extract_sql_arguments(
            "What is the total number of calls offered yesterday?", "total_calls_offered"
        )
        expected = (datetime.now(ZoneInfo("Asia/Qatar")).date() - timedelta(days=1)).isoformat()
        self.assertEqual(args["start_date"], expected)
        self.assertEqual(args["end_date"], expected)


if __name__ == "__main__":
    unittest.main()