import unittest

from services.ai.memory import RAGMemory


class _FakeReranker:
    def score(self, query, documents):
        self.last_query = query
        self.last_documents = list(documents)
        return [0.1, 0.9]


class RAGMemoryChunkingTests(unittest.TestCase):
    def test_structured_chunking_preserves_table_and_section_path(self) -> None:
        rag = RAGMemory.__new__(RAGMemory)

        markdown = """
# Overview
Revenue improved during the quarter due to export demand.

## Balance Sheet
| Metric | Value |
| --- | --- |
| Net Debt | 10B TRY |
| EBITDA | 18B TRY |
"""

        records = rag._build_chunk_records(markdown, chunk_size=70, overlap=10)

        table_records = [
            record for record in records if record.get("content_type") == "table"
        ]
        self.assertEqual(len(table_records), 1)
        self.assertIn("| Net Debt | 10B TRY |", table_records[0]["content"])
        self.assertEqual(table_records[0]["section"], "Overview / Balance Sheet")

    def test_parent_document_expansion_overrides_child_excerpt(self) -> None:
        rag = RAGMemory.__new__(RAGMemory)
        rag._get_parent_document = lambda collection_key, parent_id: {
            "id": parent_id,
            "content": "Full parent document with surrounding context.",
            "metadata": {"doc_type": "final_analysis"},
            "collection": collection_key,
        }

        expanded = rag._expand_with_context_windows(
            [
                {
                    "id": "child-1",
                    "collection": "analysis",
                    "content": "Matched child excerpt.",
                    "metadata": {
                        "parent_id": "parent-1",
                        "chunk_index": 2,
                        "section": "Overview / Risks",
                    },
                }
            ],
            context_window=0,
        )

        self.assertEqual(len(expanded), 1)
        self.assertEqual(
            expanded[0]["content"], "Full parent document with surrounding context."
        )
        self.assertEqual(expanded[0]["metadata"]["matched_chunk_index"], 2)
        self.assertEqual(expanded[0]["matched_chunk_content"], "Matched child excerpt.")


class RAGMemoryRerankTests(unittest.TestCase):
    def test_reranker_can_promote_more_relevant_result(self) -> None:
        rag = RAGMemory.__new__(RAGMemory)
        rag._rerank_weight = 0.7
        rag._rerank_candidate_pool = 10
        rag._reranker_failed = False
        rag._get_reranker = lambda: _FakeReranker()

        results = [
            {"content": "Candidate A", "retrieval_score": 0.9},
            {"content": "Candidate B", "retrieval_score": 0.7},
        ]

        rag._rerank_results("which candidate is better", results)
        ordered = sorted(results, key=lambda item: item["final_score"], reverse=True)

        self.assertEqual(ordered[0]["content"], "Candidate B")
        self.assertGreater(ordered[0]["final_score"], ordered[1]["final_score"])


class RAGMemoryQueryVariantTests(unittest.TestCase):
    def test_query_variants_include_explicit_hypothesis(self) -> None:
        rag = RAGMemory.__new__(RAGMemory)
        rag._query_expansion_enabled = True
        rag._query_expansion_weight = 0.35
        rag._query_expansion_template = "Fallback for {query}"
        rag._query_expansion_max_chars = 200

        variants = rag._build_query_variants(
            "akbank latest margin outlook",
            query_hypothesis="Akbank margin outlook, risks, and catalysts in recent quarters.",
        )

        labels = [item["label"] for item in variants]
        self.assertEqual(labels[0], "query")
        self.assertIn("hyde", labels)
        self.assertNotIn("template_hyde", labels)