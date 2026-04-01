import unittest
from unittest.mock import patch

from services.tools import SummarizeUrlContentTool


class _ReadyRAGStub:
    def is_ready(self):
        return True

    def semantic_similarity_scores(self, query_text, documents):
        _ = query_text
        scores = []
        for document in documents:
            if "Revenue jumped" in document:
                scores.append(0.98)
            else:
                scores.append(0.05)
        return scores


class UrlContentPruningTests(unittest.TestCase):
    def _build_tool(self) -> SummarizeUrlContentTool:
        tool = SummarizeUrlContentTool.__new__(SummarizeUrlContentTool)
        tool._chunk_size_chars = 40
        tool._max_selected_chunks = 1
        tool._max_expanded_chunks = 3
        tool._lead_chunk_count = 1
        tool._min_selected_chars = 10
        tool._min_focus_score = 2.2
        tool._min_salience_score = 0.45
        tool._fallback_fulltext_chars = 5
        tool._coverage_ratio_floor = 0.0
        tool._semantic_enabled = True
        tool._semantic_weight = 0.35
        tool._semantic_threshold = 0.2
        tool._context_window_size = 1
        return tool

    def test_expand_chunk_indices_respects_context_window_first(self) -> None:
        tool = self._build_tool()

        indices = tool._expand_chunk_indices([3], total_chunks=8, max_chunks=3)

        self.assertEqual(indices, [2, 3, 4])

    def test_prune_content_merges_query_hypothesis_into_focus(self) -> None:
        tool = self._build_tool()
        content = "\n\n".join(
            [
                "Intro text.",
                "Revenue jumped 24% and EBITDA margin expanded.",
                "Closing note.",
            ]
        )

        with patch("services.tools.get_rag_service", return_value=_ReadyRAGStub()):
            result = tool._prune_content(
                content=content,
                focus_query="quarterly metrics",
                query_hypothesis="revenue margin expansion",
            )

        stats = result["selection_stats"]
        self.assertIn("quarterly metrics", stats["focus_query"])
        self.assertIn("revenue margin expansion", stats["focus_query"])
        self.assertGreater(stats["top_semantic_score"], 0.9)
        self.assertIn("Revenue jumped 24%", result["content"])


if __name__ == "__main__":
    unittest.main()