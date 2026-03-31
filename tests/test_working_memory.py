import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from services.ai.providers.agent_guardrails import sanitize_memory_args
from services.ai.shared_memory_pool import SharedMemoryPool
from services.ai.working_memory import WorkingMemory


class WorkingMemoryAdaptiveTests(unittest.TestCase):
    def test_similar_facts_merge_without_increasing_count(self) -> None:
        memory = WorkingMemory(
            max_facts=5,
            adaptive_max_facts=8,
            fact_similarity_threshold=0.6,
            fact_similarity_window=5,
        )

        memory.add_facts(
            [
                "Revenue grew 20% in Q4 due to export sales.",
                "Q4 revenue grew by 20% because export sales increased.",
            ],
            importance=6,
            tags=["financials"],
        )

        self.assertEqual(memory.summary_counts()["facts"], 1)
        fact = memory.to_dict()["facts_learned"][0]
        self.assertGreaterEqual(fact["importance"], 6)
        self.assertGreaterEqual(fact["access_count"], 1)

    def test_adjust_limits_expands_fact_capacity_under_pressure(self) -> None:
        memory = WorkingMemory(
            max_facts=10,
            adaptive_max_facts=20,
            adaptive_usage_step=0.25,
        )

        limits = memory.adjust_limits(1.6)

        self.assertGreater(limits["max_facts"], 10)
        self.assertLessEqual(limits["max_facts"], 20)

    def test_similarity_density_can_trigger_consolidation_early(self) -> None:
        memory = WorkingMemory(
            max_facts=10,
            consolidation_threshold=8,
            fact_similarity_threshold=0.45,
            fact_similarity_window=6,
            consolidation_similarity_ratio=0.3,
        )

        memory.from_dict(
            {
                "facts_learned": [
                    {"text": "Net debt fell to 10B TRY after asset sales.", "importance": 7},
                    {"text": "After asset sales, net debt fell to TRY 10B.", "importance": 6},
                    {"text": "EBITDA margin improved to 18%.", "importance": 6},
                    {"text": "Quarterly EBITDA margin reached 18 percent.", "importance": 5},
                ]
            }
        )

        self.assertTrue(memory.needs_facts_consolidation())


class FakeRAGService:
    def search(self, **kwargs):
        return [
            {
                "id": "doc-1",
                "collection": "analysis",
                "content": "Revenue increased 18% while net debt fell to 10B TRY.",
                "metadata": {
                    "doc_type": "news_analysis",
                    "timestamp": "2026-03-01T10:00:00",
                    "doc_group_id": "grp-1",
                    "chunk_index": 0,
                },
                "quality_adjusted_score": 0.88,
            }
        ]


class CapturingRAGService(FakeRAGService):
    def __init__(self) -> None:
        self.last_search_kwargs = {}

    def search(self, **kwargs):
        self.last_search_kwargs = dict(kwargs)
        return super().search(**kwargs)

    def fetch_hits(self, hit_ids, context_window=0):
        _ = (hit_ids, context_window)

        return [
            {
                "id": "doc-1",
                "collection": "analysis",
                "content": "Revenue increased 18% while net debt fell to 10B TRY. Management expects margin recovery.",
                "metadata": {
                    "doc_type": "news_analysis",
                    "timestamp": "2026-03-01T10:00:00",
                    "doc_group_id": "grp-1",
                    "chunk_index": 0,
                },
                "quality_adjusted_score": 0.91,
            }
        ]


class WorkingMemoryRefreshTests(unittest.TestCase):
    def test_refresh_context_loads_rag_hits_into_memory(self) -> None:
        memory = WorkingMemory(max_facts=6)

        entries = memory.refresh_context(
            "sample query",
            rag_service=FakeRAGService(),
            top_k=3,
            importance=7,
        )

        self.assertEqual(len(entries), 1)
        self.assertEqual(memory.summary_counts()["facts"], 1)
        facts = memory.to_dict()["facts_learned"]
        self.assertIn("[RAG:news_analysis@2026-03-01T10:00:00]", facts[0]["text"])
        self.assertEqual(facts[0]["importance"], 7)

    def test_refresh_context_passes_query_hypothesis(self) -> None:
        memory = WorkingMemory(max_facts=6)
        rag = CapturingRAGService()

        memory.refresh_context(
            "sample query",
            rag_service=rag,
            query_hypothesis="Hypothetical research note about sample query.",
        )

        self.assertEqual(
            rag.last_search_kwargs.get("query_hypothesis"),
            "Hypothetical research note about sample query.",
        )


class WorkingMemoryDynamicImportanceTests(unittest.TestCase):
    def test_get_facts_tracks_access_and_increases_importance(self) -> None:
        memory = WorkingMemory(max_facts=6, importance_recalc_interval_seconds=5)
        memory.add_facts(["Backlog coverage improved to 2.4x sales."], importance=5)

        before = memory.to_dict()["facts_learned"][0]
        memory.get_facts()
        after = memory.to_dict()["facts_learned"][0]

        self.assertEqual(after["access_count"], before["access_count"] + 1)
        self.assertGreaterEqual(after["importance"], before["importance"])

    def test_shared_reference_count_affects_importance(self) -> None:
        memory = WorkingMemory(max_facts=6, importance_recalc_interval_seconds=5)
        memory.from_dict(
            {
                "facts_learned": [
                    {
                        "text": "Export order pipeline remains strong.",
                        "importance": 5,
                        "base_importance": 5,
                        "shared_reference_count": 6,
                        "access_count": 0,
                    }
                ]
            }
        )

        memory.recalculate_importance_scores(force=True)

        fact = memory.to_dict()["facts_learned"][0]
        self.assertGreater(fact["importance"], fact["base_importance"])


class WorkingMemoryInputNormalizationTests(unittest.TestCase):
    def test_sanitize_memory_args_keeps_dict_facts_with_string_provenance(self) -> None:
        sanitized = sanitize_memory_args(
            {
                "new_facts": {
                    "XU100_current_price_18Mar2026": "13,195.27 TRY",
                    "XU100_1M_return": "-7.47%",
                },
                "fact_provenance": "yfinance_index_data, yfinance_search",
                "fact_tags": ["market_data"],
            }
        )

        self.assertEqual(
            sanitized["fact_provenance"],
            {"source_id": "yfinance_index_data, yfinance_search"},
        )
        self.assertEqual(
            sanitized["new_facts"],
            [
                {"text": "XU100_current_price_18Mar2026: 13,195.27 TRY"},
                {"text": "XU100_1M_return: -7.47%"},
            ],
        )

    def test_update_from_args_accepts_dict_facts_and_string_provenance(self) -> None:
        memory = WorkingMemory(max_facts=10)

        memory.update_from_args(
            {
                "new_facts": {
                    "TUPRS_current_price": "253.5 TRY",
                    "TUPRS_analyst_consensus": "Hold (rating score 2.6)",
                },
                "fact_provenance": "yfinance_overview, yfinance_analyst",
                "research_milestones": "Validated TUPRS snapshot.",
            }
        )

        snapshot = memory.to_dict()
        self.assertEqual(memory.summary_counts()["facts"], 2)
        self.assertEqual(snapshot["facts_learned"][0]["provenance"]["source_id"], "yfinance_overview, yfinance_analyst")
        self.assertEqual(memory.summary_counts()["milestones"], 1)


class WorkingMemoryConsolidationFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_consolidation_falls_back_when_callback_returns_empty(self) -> None:
        async def empty_callback(facts):
            return []

        memory = WorkingMemory(
            consolidation_callback=empty_callback,
            max_facts=10,
            consolidation_threshold=3,
            fact_similarity_threshold=0.6,
        )
        memory.from_dict(
            {
                "facts_learned": [
                    {
                        "text": "Revenue grew 20% in Q4 due to export sales.",
                        "importance": 6,
                        "tags": ["finance"],
                    },
                    {
                        "text": "Q4 revenue grew by 20% because export sales increased.",
                        "importance": 7,
                        "tags": ["finance"],
                    },
                    {
                        "text": "Operating margin held at 18%.",
                        "importance": 5,
                        "tags": ["finance"],
                    },
                ]
            }
        )

        await memory._run_consolidation()

        self.assertLessEqual(memory.summary_counts()["facts"], 2)


class WorkingMemorySharedPoolTests(unittest.TestCase):
    def test_shared_pool_sync_and_refresh_between_memories(self) -> None:
        with TemporaryDirectory() as tmpdir:
            pool = SharedMemoryPool(storage_path=Path(tmpdir) / "pool.json")

            producer = WorkingMemory(
                max_facts=6,
                shared_memory_pool=pool,
                shared_memory_scope="symbol:akbnk",
                shared_memory_agent_name="producer_agent",
            )
            producer.add_facts(
                ["Net interest margin improved to 5.2% in the latest quarter."],
                importance=8,
                tags=["banking"],
            )
            producer.update(
                source_summary="Quarterly banking update reviewed",
                contradictions=["Loan growth slowed while margin improved"],
            )

            consumer = WorkingMemory(
                max_facts=6,
                shared_memory_pool=pool,
                shared_memory_scope="symbol:akbnk",
                shared_memory_agent_name="consumer_agent",
            )
            entries = consumer.refresh_from_shared_pool(
                "akbnk margin latest quarter",
                top_k=3,
            )

            self.assertEqual(len(entries), 1)
            self.assertEqual(consumer.summary_counts()["facts"], 1)
            self.assertIn("[Shared:symbol:akbnk@", consumer.get_facts()[0])

    def test_shared_pool_scope_isolated(self) -> None:
        with TemporaryDirectory() as tmpdir:
            pool = SharedMemoryPool(storage_path=Path(tmpdir) / "pool.json")

            producer = WorkingMemory(
                max_facts=6,
                shared_memory_pool=pool,
                shared_memory_scope="symbol:thyao",
                shared_memory_agent_name="producer_agent",
            )
            producer.add_facts(["Passenger yield improved despite weaker load factor."])

            consumer = WorkingMemory(
                max_facts=6,
                shared_memory_pool=pool,
                shared_memory_scope="symbol:asels",
                shared_memory_agent_name="consumer_agent",
            )
            entries = consumer.refresh_from_shared_pool("passenger yield load factor")

            self.assertEqual(entries, [])
            self.assertEqual(consumer.summary_counts()["facts"], 0)


if __name__ == "__main__":
    unittest.main()