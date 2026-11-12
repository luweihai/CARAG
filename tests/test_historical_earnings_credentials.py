"""Offline regressions for retrieval using the already configured Neo4j driver.

Run with: python -m unittest discover -s tests -v
Only the dependencies already listed in README.md are required.
"""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from agents import historicalEarningsAgent as module


class HistoricalEarningsCredentialsTests(unittest.TestCase):
    def configured_agent(self, read_text, with_uri=False):
        credentials = {
            "openai_api_key": "synthetic-test-key",
            "neo4j_uri": "bolt://fixture.invalid:7687",
            "neo4j_username": "fixture-user",
            "neo4j_password": "synthetic-test-password",
        }
        path = Path("custom-config") / "fixture.json"

        def configured_file_only(requested, *args, **kwargs):
            if requested != path:
                raise FileNotFoundError("No default credentials file in this test")
            return json.dumps(credentials)

        read_text.side_effect = configured_file_only
        session = MagicMock()
        session.__enter__.return_value = session
        driver = SimpleNamespace(session=Mock(return_value=session))
        if with_uri:
            driver._uri = credentials["neo4j_uri"]
        client = MagicMock()
        with (
            patch.object(module, "OpenAI", return_value=client),
            patch.object(module.GraphDatabase, "driver", return_value=driver) as factory,
            patch.object(module, "OpenAIEmbeddings"),
        ):
            agent = module.HistoricalEarningsAgent(str(path))
        factory.assert_called_once_with(
            credentials["neo4j_uri"],
            auth=(credentials["neo4j_username"], credentials["neo4j_password"]),
        )
        return agent, driver, session, client, path

    @staticmethod
    def record(quarter):
        value = {"quarter": quarter, "metric": "Revenue", "embedding": [1.0, 0.0]}
        return SimpleNamespace(data=lambda: value.copy())

    def test_vector_lookup_reuses_configured_driver_with_or_without_private_uri(self):
        for with_uri in (False, True):
            with self.subTest(with_uri=with_uri), patch.object(Path, "read_text", autospec=True) as read_text:
                agent, driver, session, client, path = self.configured_agent(read_text, with_uri)
                session.run.side_effect = [
                    [self.record("2025-Q1"), self.record("2025-Q2"), self.record("2025-Q3")],
                    [self.record("2024-Q2")],
                ]

                result = agent.get_similar_facts_by_embedding(
                    {"embedding": [1.0, 0.0]}, "FIXTURE", "2025-Q2", top_n=3
                )

                self.assertIsNotNone(result)
                self.assertEqual([item["quarter"] for item in result], ["2025-Q1", "2024-Q2"])
                read_text.assert_called_once_with(path)
                driver.session.assert_called_once_with()
                self.assertEqual(session.run.call_count, 2)
                session.__exit__.assert_called_once()
                client.embeddings.create.assert_not_called()
                self.assertEqual(len(agent.kg_retrieval_times), 1)

    def test_empty_lookup_returns_empty_list_without_reopening_credentials(self):
        with patch.object(Path, "read_text", autospec=True) as read_text:
            agent, _, session, _, path = self.configured_agent(read_text)
            session.run.side_effect = [[], []]

            result = agent.get_similar_facts_by_embedding(
                {"embedding": [1.0, 0.0]}, "FIXTURE", "2025-Q2"
            )

            self.assertEqual(result, [])
            read_text.assert_called_once_with(path)
            session.__exit__.assert_called_once()

    def test_vector_failure_still_uses_the_existing_python_fallback(self):
        with patch.object(Path, "read_text", autospec=True) as read_text:
            agent, _, session, _, path = self.configured_agent(read_text, with_uri=True)
            session.run.side_effect = [
                RuntimeError("Synthetic vector index failure"),
                [self.record("2025-Q1"), self.record("2025-Q2"), self.record("2025-Q3")],
            ]

            result = agent.get_similar_facts_by_embedding(
                {"embedding": [1.0, 0.0]}, "FIXTURE", "2025-Q2", top_n=1
            )

            self.assertIsNotNone(result)
            self.assertEqual([item["quarter"] for item in result], ["2025-Q1"])
            read_text.assert_called_once_with(path)
            self.assertEqual(session.run.call_count, 2)
            session.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
