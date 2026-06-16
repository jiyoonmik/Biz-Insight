import unittest

from src.context.queries import get_knowledge_context, search_company_by_name


class ContextQueryTests(unittest.TestCase):
    def test_company_search_and_knowledge_context(self) -> None:
        results = search_company_by_name("AJ네트웍스", limit=1)
        self.assertEqual(results[0]["stock_code"], "095570")
        context = get_knowledge_context("095570", year=2022)
        self.assertEqual(context["company"]["stock_code"], "095570")
        self.assertTrue(context["financial_observations"])
        self.assertIn("criteria_evidence", context)


if __name__ == "__main__":
    unittest.main()
