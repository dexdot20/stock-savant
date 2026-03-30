import logging
import unittest

from services.ai.providers.base_provider import ProviderContext
from services.ai.providers.openrouter_provider import (
    _prepare_openrouter_payload,
    _process_openrouter_response,
)
from services.ai.providers.research_agent_support import ResearchAgentSupportMixin
from services.ai.working_memory import WorkingMemory


class _StubExecutor:
    async def send_async(self, prompt, request_type="summarizer", timeout_override=None):
        del prompt, request_type, timeout_override
        return {"content": "Compressed summary"}


class _HistorySummaryHarness(ResearchAgentSupportMixin):
    def __init__(self) -> None:
        self.logger = logging.getLogger("history-summary-harness")
        self._history_token_limit = 32
        self._token_encoding = "cl100k_base"
        self._prompt_resolver = lambda key: "Summarize in {output_language}"
        self._output_lang = "English"
        self._config = {"ai": {"output_language": "English"}}
        self._request_executor = _StubExecutor()
        self.memory = WorkingMemory(max_facts=6)
        self._history_archives = []
        self._tool_journal = []


class OpenRouterPromptCacheTests(unittest.TestCase):
    def test_openrouter_payload_preserves_sticky_routing_by_removing_manual_order(self) -> None:
        context = ProviderContext(
            logger=logging.getLogger("openrouter-payload-test"),
            providers_cfg={
                "openrouter": {
                    "api_key": "openrouter-key",
                    "default_model": "anthropic/claude-sonnet-4.5",
                    "provider": {
                        "order": ["anthropic", "google-vertex"],
                        "require_parameters": True,
                    },
                    "prompt_caching": {
                        "enabled": True,
                        "preserve_sticky_routing": True,
                        "cache_control": {"type": "ephemeral"},
                    },
                }
            },
            model_configs={},
            request_timeout=30,
        )

        _, _, payload, _ = _prepare_openrouter_payload(
            [{"role": "user", "content": "hello"}],
            "news",
            context,
        )

        self.assertEqual(payload["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("order", payload["provider"])
        self.assertTrue(payload["provider"]["require_parameters"])

    def test_openrouter_response_exposes_prompt_cache_metrics(self) -> None:
        context = ProviderContext(
            logger=logging.getLogger("openrouter-cache-usage-test"),
            providers_cfg={"openrouter": {"prompt_caching": {"log_usage": True}}},
            model_configs={},
            request_timeout=30,
        )

        response = _process_openrouter_response(
            {
                "choices": [
                    {
                        "message": {"content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 25,
                    "total_tokens": 1025,
                    "cache_discount": 1.25,
                    "prompt_tokens_details": {
                        "cached_tokens": 768,
                        "cache_write_tokens": 512,
                    },
                },
            },
            context,
            "anthropic/claude-sonnet-4.5",
            "news",
        )

        self.assertEqual(response["usage"]["cached_tokens"], 768)
        self.assertEqual(response["usage"]["cache_write_tokens"], 512)
        self.assertEqual(response["usage"]["cache_discount"], 1.25)


class HistorySummaryCacheComplianceTests(unittest.IsolatedAsyncioTestCase):
    async def test_reflection_prompt_includes_tool_journal(self) -> None:
        harness = _HistorySummaryHarness()
        harness._tool_journal = [
            {
                "step": 3,
                "tools": [
                    {"name": "search_web", "args": {"query": "AKBNK outlook"}},
                    {"name": "fetch_url_content", "args": {"url": "https://example.com"}},
                ],
            }
        ]

        prompt = harness._build_self_reflection_prompt(step=3)

        self.assertIn("Chronological step trace in this run:", prompt)
        self.assertIn("search_web", prompt)
        self.assertIn("fetch_url_content", prompt)

    async def test_history_summary_preserves_first_non_system_anchor(self) -> None:
        harness = _HistorySummaryHarness()
        long_chunk = "token " * 400
        history = [
            {"role": "system", "content": "You are a research assistant."},
            {"role": "user", "content": "Investigate company ABC and keep the initial task stable."},
            {"role": "assistant", "content": f"Planning the research path. {long_chunk}"},
            {"role": "user", "content": f"Gather source one. {long_chunk}"},
            {"role": "assistant", "content": f"Source one collected. {long_chunk}"},
            {"role": "user", "content": f"Gather source two. {long_chunk}"},
            {"role": "assistant", "content": f"Source two collected. {long_chunk}"},
            {"role": "user", "content": f"Continue with conclusions. {long_chunk}"},
        ]

        summarized = await harness._summarize_history(history)

        self.assertEqual(summarized[0], history[0])
        self.assertEqual(summarized[1], history[1])
        self.assertIn("PREVIOUS CONTEXT SUMMARY", summarized[2]["content"])


if __name__ == "__main__":
    unittest.main()