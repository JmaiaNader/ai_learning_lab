import unittest
from unittest.mock import patch

from tools import mcp_server


class McpServerTests(unittest.TestCase):
    def test_hr_policy_search_excludes_non_hr_documents(self):
        candidates = [
            {"text": "Airline check-in policy", "source": "airline_policy_v1.pdf", "page": 1},
            {"text": "Employee attendance and working hours", "source": "the_nestle_hr_policy_pdf_2012.pdf", "page": 5},
        ]
        with patch.object(mcp_server, "retrieve", return_value=candidates) as retrieve:
            result = mcp_server.search_hr_policy("check-in policy")

        retrieve.assert_called_once_with("HR human resources policy check-in policy", count=12, strict=True)
        self.assertEqual(len(result["passages"]), 1)
        self.assertEqual(result["passages"][0]["source"], "the_nestle_hr_policy_pdf_2012.pdf")

    def test_hr_search_reports_no_matching_policy_source(self):
        with patch.object(
            mcp_server,
            "retrieve",
            return_value=[
                {"text": "Airline policy", "source": "airline_policy.pdf", "page": 1},
                {"text": "Working conditions", "source": "hr_policy.pdf", "page": 2},
            ],
        ):
            result = mcp_server.search_hr_policy("check-in policy")

        self.assertEqual(result["passages"], [])
        self.assertIn("No matching passages", result["message"])

    def test_hr_search_surfaces_retrieval_errors(self):
        with patch.object(mcp_server, "retrieve", side_effect=RuntimeError("Chroma unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Chroma unavailable"):
                mcp_server.search_hr_policy("check-in policy")


if __name__ == "__main__":
    unittest.main()