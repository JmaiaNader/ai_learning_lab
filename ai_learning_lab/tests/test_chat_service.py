import unittest
from unittest.mock import patch

from backend.services import chat_service


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

    def test_langgraph_and_mcp_are_not_silently_run_as_basic_chat(self):
        for track in ("langgraph", "mcp"):
            with self.subTest(track=track):
                result = chat_service.answer_question(
                    "Explain this architecture", track, "ollama", "mock-model"
                )

                self.assertEqual(result["status"], "not_implemented")

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


if __name__ == "__main__":
    unittest.main()