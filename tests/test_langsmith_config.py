import os
import unittest

from src.config import configure_langsmith_environment, langsmith_tracing_enabled


class LangSmithConfigTests(unittest.TestCase):
    ENV_KEYS = (
        "LANGSMITH_TRACING",
        "LANGSMITH_TRACING_V2",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_TRACING",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGCHAIN_PROJECT",
        "LANGSMITH_ENDPOINT",
        "LANGCHAIN_ENDPOINT",
    )

    def setUp(self) -> None:
        self.previous = {key: os.environ.get(key) for key in self.ENV_KEYS}
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)
            if self.previous[key] is not None:
                os.environ[key] = self.previous[key]
        configure_langsmith_environment()

    def test_langsmith_env_aliases_are_normalized(self) -> None:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = "test-key"
        os.environ["LANGSMITH_PROJECT"] = "biz-insight-test"
        os.environ["LANGSMITH_ENDPOINT"] = "https://example.test"

        configure_langsmith_environment()

        self.assertEqual(os.environ["LANGCHAIN_TRACING_V2"], "true")
        self.assertEqual(os.environ["LANGSMITH_TRACING_V2"], "true")
        self.assertEqual(os.environ["LANGCHAIN_TRACING"], "true")
        self.assertEqual(os.environ["LANGCHAIN_API_KEY"], "test-key")
        self.assertEqual(os.environ["LANGCHAIN_PROJECT"], "biz-insight-test")
        self.assertEqual(os.environ["LANGCHAIN_ENDPOINT"], "https://example.test")
        self.assertTrue(langsmith_tracing_enabled())

    def test_explicit_false_tracing_is_preserved(self) -> None:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGSMITH_API_KEY"] = "test-key"

        configure_langsmith_environment()

        self.assertEqual(os.environ["LANGCHAIN_TRACING_V2"], "false")
        self.assertEqual(os.environ["LANGSMITH_TRACING_V2"], "false")
        self.assertEqual(os.environ["LANGCHAIN_TRACING"], "false")
        self.assertFalse(langsmith_tracing_enabled())


if __name__ == "__main__":
    unittest.main()
