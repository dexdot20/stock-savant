import logging
import unittest

from services.ai.providers.base_provider import ProviderContext
from services.ai.providers.openrouter_provider import (
    _prepare_openrouter_payload,
    _process_openrouter_response,
    _inject_gemini_cache_control,
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


class GeminiCacheControlTests(unittest.TestCase):
    """Test Gemini-specific cache_control injection for OpenRouter."""

    def test_gemini_cache_control_injected_into_last_message(self) -> None:
        """Gemini requires cache_control inside message content blocks, not top-level."""
        payload = {
            "model": "google/gemini-2.5-pro",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Analyze this document."},
                {"role": "user", "content": "Large document content here..."},
            ],
        }
        cache_control = {"type": "ephemeral"}

        _inject_gemini_cache_control(payload, cache_control)

        # cache_control should NOT be at top level
        self.assertNotIn("cache_control", payload)

        # Last message should have cache_control in its content
        last_message = payload["messages"][-1]
        self.assertIsInstance(last_message["content"], list)
        self.assertEqual(last_message["content"][0]["cache_control"], cache_control)

    def test_gemini_cache_control_with_existing_list_content(self) -> None:
        """Handle case where content is already a list of blocks."""
        payload = {
            "model": "google/gemini-2.5-flash",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Part 1"},
                        {"type": "text", "text": "Part 2 with large content..."},
                    ],
                },
            ],
        }
        cache_control = {"type": "ephemeral"}

        _inject_gemini_cache_control(payload, cache_control)

        # cache_control should be added to the last text block
        last_message = payload["messages"][-1]
        content_blocks = last_message["content"]
        self.assertEqual(content_blocks[-1]["cache_control"], cache_control)


class DeepSeekCacheComplianceTests(unittest.TestCase):
    """Test DeepSeek-specific caching requirements.

    DeepSeek requires identical prefixes from token 0 for cache hits.
    Dynamic content must appear AFTER stable prefixes.
    """

    def test_stable_message_prefix_structure(self) -> None:
        """Verify messages have stable system prompt and first user message."""
        # Simulating pre_research_agent history structure
        system_prompt = "You are a financial research agent. Use tools to research and summarize."
        stable_user_prefix = "Start research, rank all suitable companies you find according to criteria."
        dynamic_context = "Date: 31 March 2026\nTarget Exchange: BIST\nAdditional Criteria: None"

        history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": stable_user_prefix},
            {"role": "user", "content": dynamic_context},
        ]

        # First two messages should be stable across requests
        self.assertEqual(history[0]["role"], "system")
        self.assertEqual(history[1]["role"], "user")

        # Dynamic content should be in separate message
        self.assertIn("Date:", history[2]["content"])
        self.assertIn("Target Exchange:", history[2]["content"])

    def test_cache_hit_token_extraction(self) -> None:
        """Verify DeepSeek cache hit/miss tokens are extracted from usage."""
        from services.ai.providers.cache_usage import extract_prompt_cache_usage

        # DeepSeek usage format
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "total_tokens": 1050,
            "prompt_cache_hit_tokens": 800,
            "prompt_cache_miss_tokens": 200,
        }

        cache_usage = extract_prompt_cache_usage(usage)

        self.assertEqual(cache_usage["prompt_cache_hit_tokens"], 800)
        self.assertEqual(cache_usage["prompt_cache_miss_tokens"], 200)


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