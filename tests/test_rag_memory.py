import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from services.ai.memory import RAGMemory


class _FakeCacheManager:
    def get_ttl_cache(self, **kwargs):
        self.last_ttl_cache_kwargs = kwargs
        return self

    def get_many(self, _namespace, _keys):
        return {}

    def set_many(self, _namespace, _values):
        return None


class _FakeEncodedBatch:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeSentenceTransformer:
    instances = []

    def __init__(self, model_name):
        self.model_name = model_name
        self.calls = []
        self.__class__.instances.append(self)

    def encode(self, texts, **kwargs):
        captured_texts = list(texts)
        self.calls.append((captured_texts, kwargs))
        vectors = [
            [float(len(text)), float(len(text)) + 0.5]
            for text in captured_texts
        ]
        return _FakeEncodedBatch(vectors)


class _FakePersistentClient:
    def __init__(self, path):
        self.path = path
        self.collections = []

    def get_or_create_collection(self, name, embedding_function, metadata):
        config = embedding_function.get_config()
        rebuilt = embedding_function.build_from_config(config)
        self.collections.append(
            {
                "name": name,
                "embedding_name": embedding_function.name(),
                "rebuilt_name": rebuilt.name(),
                "legacy": embedding_function.is_legacy(),
                "config": config,
                "metadata": metadata,
            }
        )
        return object()


class _FakeReranker:
    def score(self, query, documents):
        self.last_query = query
        self.last_documents = list(documents)
        return [0.1, 0.9]


class RAGMemoryEmbeddingProtocolTests(unittest.TestCase):
    def test_rag_initialization_uses_embedding_protocol(self) -> None:
        fake_config = {
            "ai": {
                "rag": {
                    "enabled": True,
                    "embedding_model": "intfloat/multilingual-e5-large",
                    "normalize_embeddings": True,
                    "top_k": 5,
                    "candidate_pool": 18,
                    "huggingface": {"suppress_model_load_report": True},
                    "reranker": {"enabled": False},
                    "query_expansion": {"enabled": False},
                }
            },
            "cache": {"rag_embeddings": {"max_entries": 16, "ttl_seconds": 60}},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            fake_cache = _FakeCacheManager()
            _FakeSentenceTransformer.instances = []
            with (
                patch("services.ai.memory.get_config", return_value=fake_config),
                patch("services.ai.memory.get_runtime_dir", return_value=Path(temp_dir)),
                patch("services.ai.memory.get_unified_cache", return_value=fake_cache),
                patch("services.ai.memory.SentenceTransformer", _FakeSentenceTransformer),
                patch(
                    "services.ai.memory.chromadb",
                    SimpleNamespace(PersistentClient=_FakePersistentClient),
                ),
            ):
                rag = RAGMemory()

        wrapper = rag._embedding_function
        self.assertIsNotNone(wrapper)

        document_vectors = wrapper.embed_documents(["Quarterly revenue outlook"])
        query_vectors = wrapper.embed_query(["Quarterly revenue outlook"])

        self.assertTrue(rag.is_ready())
        self.assertEqual(len(rag._client.collections), 6)
        self.assertTrue(
            all(
                item["embedding_name"] == "sentence_transformer"
                for item in rag._client.collections
            )
        )
        self.assertTrue(
            all(item["embedding_name"] == item["rebuilt_name"] for item in rag._client.collections)
        )
        self.assertTrue(all(item["legacy"] is False for item in rag._client.collections))

        model = _FakeSentenceTransformer.instances[0]
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(model.calls[0][0][0].startswith("passage: "))
        self.assertTrue(model.calls[1][0][0].startswith("query: "))
        self.assertEqual(len(document_vectors), 1)
        self.assertEqual(len(query_vectors), 1)
        self.assertNotEqual(document_vectors, query_vectors)

    def test_semantic_similarity_scores_uses_embedding_vectors(self) -> None:
        class _FakeEmbeddingFunction:
            def encode(self, texts, *, kind):
                captured = list(texts)
                if kind == "query":
                    return [[1.0, 0.0] for _ in captured]
                mapping = {
                    "strong": [1.0, 0.0],
                    "weak": [0.0, 1.0],
                }
                return [mapping.get(text, [0.5, 0.5]) for text in captured]

        rag = RAGMemory.__new__(RAGMemory)
        rag._embedding_function = _FakeEmbeddingFunction()

        scores = rag.semantic_similarity_scores("query", ["strong", "weak"])

        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0], scores[1])
        self.assertAlmostEqual(scores[0], 1.0)

    def test_has_document_url_checks_collection_lookup(self) -> None:
        class _FakeCollection:
            def __init__(self):
                self.last_where = None

            def get(self, where=None, limit=None):
                self.last_where = where
                _ = limit
                return {"ids": ["doc-1"]}

        rag = RAGMemory.__new__(RAGMemory)
        fake_collection = _FakeCollection()
        rag._collections = {"pre_research": fake_collection}

        exists = rag.has_document_url("pre_research", "https://example.com/report")

        self.assertTrue(exists)
        self.assertEqual(
            fake_collection.last_where,
            {"url": "https://example.com/report"},
        )


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

    def test_index_pre_research_adds_symbol_and_url_metadata(self) -> None:
        rag = RAGMemory.__new__(RAGMemory)
        rag.is_ready = lambda: True
        rag.has_document_url = lambda collection_key, url: False
        rag._clamp_confidence = lambda value, default: value if value is not None else default
        rag._default_confidence_score = 0.55
        rag._safe_int = lambda value, default=0: int(value if value is not None else default)
        rag._doc_group_id = lambda *args: "group-1"
        rag._to_unix_ts = lambda value: 1234567890
        rag._pre_research_chunk_size = 1800
        rag._pre_research_chunk_overlap = 220

        captured = {}

        def _capture(collection_key, markdown_content, metadata, chunk_size, overlap, id_prefix):
            captured["collection_key"] = collection_key
            captured["markdown_content"] = markdown_content
            captured["metadata"] = metadata
            captured["chunk_size"] = chunk_size
            captured["overlap"] = overlap
            captured["id_prefix"] = id_prefix
            return 2

        rag._index_parent_child_document = _capture

        indexed = rag.index_pre_research(
            exchange="BIST",
            markdown_content="Digest body",
            symbol="SAHOL.IS",
            url="https://example.com/sahol",
            doc_type="fetch_digest",
        )

        self.assertEqual(indexed, 2)
        self.assertEqual(captured["collection_key"], "pre_research")
        self.assertEqual(captured["metadata"]["symbol"], "SAHOL.IS")
        self.assertEqual(captured["metadata"]["url"], "https://example.com/sahol")
        self.assertEqual(captured["metadata"]["doc_type"], "fetch_digest")


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


class RAGMemorySearchFilterTests(unittest.TestCase):
    def test_symbol_filter_applies_to_pre_research_collection(self) -> None:
        rag = RAGMemory.__new__(RAGMemory)
        rag.enabled = True
        rag.top_k = 5
        rag._candidate_pool = 5
        rag._rerank_candidate_pool = 5
        rag._collections = {"pre_research": object()}
        rag._build_query_variants = lambda query, query_hypothesis=None, query_variants=None: [
            {"label": "query", "text": query, "weight": 1.0}
        ]
        rag._embed_query = lambda query_text: [0.1]
        captured_where = []

        def _query_collection(collection_key, query, n_results, where=None, query_embedding=None):
            _ = (collection_key, query, n_results, query_embedding)
            captured_where.append(where)
            return [
                {
                    "id": "doc-1",
                    "content": "Digest",
                    "metadata": {"symbol": "SAHOL.IS", "doc_group_id": "grp-1", "chunk_index": 0},
                    "distance": 0.1,
                    "collection": "pre_research",
                }
            ]

        rag._query_collection = _query_collection
        rag._normalize_collection_scores = lambda results: None
        rag._apply_confidence_weighting = lambda results: None
        rag._rerank_results = lambda query, results: results
        rag._expand_with_context_windows = lambda items, context_window: items

        rag.search(
            "screening query",
            collection="pre_research",
            symbol_filter="SAHOL.IS",
        )

        self.assertEqual(captured_where[0], {"symbol": "SAHOL.IS"})