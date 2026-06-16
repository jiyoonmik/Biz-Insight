import unittest

from fastapi.testclient import TestClient

from src.api.main import app


class ApiCompanyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_company_profile_endpoint(self) -> None:
        response = self.client.get("/companies/095570")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["stock_code"], "095570")

    def test_knowledge_context_endpoint(self) -> None:
        response = self.client.get("/companies/095570/knowledge-context", params={"year": 2022})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["year"], 2022)


if __name__ == "__main__":
    unittest.main()

