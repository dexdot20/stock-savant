import logging
import unittest
from io import StringIO
from rich.console import Console

from services.ai.providers.agent_guardrails import (
    build_tool_plan_preview,
    get_pre_research_pivot_notice,
    looks_like_bulk_list_fact,
    normalize_tool_args,
    sanitize_memory_args,
    validate_tool_args,
)
from services.ai.providers.deepseek_provider import (
    _ConsoleStreamRenderer,
    _process_deepseek_response,
)
from services.ai.providers.openrouter_provider import _process_openrouter_response
from services.ai.providers.base_provider import ProviderContext
from services.ai.providers.pre_research_agent import PreResearchAgent
from services.ai.providers.prompt_store import PromptStore


class AgentToolPlanPreviewTests(unittest.TestCase):
    def test_memory_updates_do_not_consume_parallel_slots(self) -> None:
        tool_calls = [
            {"name": "update_working_memory", "args": {"new_facts": ["A"]}},
            {"name": "search_web", "args": {"query": "one"}},
            {"name": "search_news", "args": {"query": "two"}},
            {"name": "search_google_news", "args": {"query": "three"}},
            {"name": "yfinance_overview", "args": {"ticker": "EREGL.IS"}},
            {"name": "yfinance_price_history", "args": {"ticker": "EREGL.IS"}},
        ]

        preview = build_tool_plan_preview(
            tool_calls,
            max_parallel_tools=4,
            non_dedup_tools={"update_working_memory"},
        )

        self.assertEqual(preview[0]["status"], "memory_update")
        executable_statuses = [row["status"] for row in preview[1:]]
        self.assertEqual(executable_statuses[:4], ["execute_now"] * 4)
        self.assertEqual(executable_statuses[4], "deferred")


class PromptStoreAgentLimitTests(unittest.TestCase):
    def test_prompt_uses_agent_parallel_limit_and_failure_guidance(self) -> None:
        store = PromptStore(
            config={
                "ai": {
                    "agent_tool_limits": {"max_parallel_tools": 4},
                    "output_language": "English",
                },
                "network": {"smart_retry": {}},
                "files": {},
            },
            logger=logging.getLogger("prompt-store-test"),
        )

        prompt = store.get("pre_research_agent")

        self.assertIn("at most **4** tools may run in parallel per step", prompt)
        self.assertIn("does NOT consume a parallel external-tool slot", prompt)
        self.assertIn("treat that source fetch as failed", prompt)
        self.assertIn("Do NOT write failed fetch attempts", prompt)


class AgentGuardrailNormalizationTests(unittest.TestCase):
    def test_normalize_search_memory_alias_and_validate(self) -> None:
        normalized = normalize_tool_args(
            "search_memory", {"query": "BIST", "max_hits": 10}
        )

        self.assertEqual(normalized["limit"], 10)
        self.assertNotIn("max_hits", normalized)
        self.assertIsNone(validate_tool_args("search_memory", normalized))

    def test_normalize_yfinance_search_aliases_and_validate(self) -> None:
        normalized = normalize_tool_args(
            "yfinance_search",
            {
                "query": "BIST",
                "count": 20,
                "asset_type": "stock",
                "region": "TR",
            },
        )

        self.assertEqual(normalized["max_results"], 20)
        self.assertEqual(normalized["type_filter"], "stock")
        self.assertNotIn("region", normalized)
        self.assertNotIn("count", normalized)
        self.assertNotIn("asset_type", normalized)
        self.assertIsNone(validate_tool_args("yfinance_search", normalized))

    def test_validate_unknown_yfinance_search_arg(self) -> None:
        error = validate_tool_args(
            "yfinance_search", {"query": "BIST", "foo": "bar"}
        )

        self.assertIn("unsupported argument(s): foo", error)

    def test_validate_unknown_search_memory_arg(self) -> None:
        error = validate_tool_args(
            "search_memory", {"query": "BIST", "foo": "bar"}
        )

        self.assertIn("unsupported argument(s): foo", error)

    def test_bulk_list_fact_is_filtered_from_verified_facts(self) -> None:
        args = sanitize_memory_args(
            {
                "new_facts": [
                    {
                        "fact": "List of constituents includes: AEFES, AGHOL, AGROT, AHGAZ, AKSA, AKSEN, ALARK, ALFAS, ALTNY, ANHYT, ANSGR, ARCLK, ARDYZ, ASELS, ASTOR, AVPGY, BIMAS, CCOLA, EREGL, GARAN.",
                        "source": "secondary-source",
                    }
                ]
            }
        )

        self.assertEqual(args["new_facts"], [])
        self.assertTrue(
            any(
                "bulk list-like" in item
                for item in args.get("research_milestones", [])
            )
        )
        self.assertTrue(looks_like_bulk_list_fact("AEFES, AGHOL, AGROT, AHGAZ, AKSA, AKSEN, ALARK, ALFAS, ALTNY, ANHYT, ANSGR, ARCLK, ARDYZ, ASELS"))

    def test_pre_research_pivot_notice_after_many_relative_low_failures(self) -> None:
        notice = get_pre_research_pivot_notice(
            {
                "rejected_hypotheses": [
                    "GARAN.IS is not at a relative low.",
                    "THYAO.IS is not at a relative low.",
                    "BIMAS.IS is not at a relative low.",
                    "TCELL.IS is not at a relative low.",
                    "ENJSA.IS is not near the lower end of the range.",
                ]
            },
            [
                "yfinance_price_history",
                "yfinance_price_history",
                "yfinance_price_history",
            ],
        )

        self.assertIn("Screening drift detected", notice)


class ReasoningUxTests(unittest.TestCase):
    def test_deepseek_response_preserves_streaming_metadata(self) -> None:
        context = ProviderContext(
            logger=logging.getLogger("deepseek-response-test"),
            providers_cfg={},
            model_configs={},
            request_timeout=30,
        )

        response = _process_deepseek_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Final answer",
                            "reasoning_content": "Step by step",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "_streaming": {
                    "reasoning_streamed": True,
                    "content_streamed": True,
                },
            },
            context,
            "deepseek-reasoner",
            "pre_research",
        )

        self.assertTrue(response["reasoning_streamed"])
        self.assertTrue(response["content_streamed"])

    def test_openrouter_response_reads_reasoning_details(self) -> None:
        context = ProviderContext(
            logger=logging.getLogger("openrouter-response-test"),
            providers_cfg={},
            model_configs={},
            request_timeout=30,
        )

        response = _process_openrouter_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Final answer",
                            "reasoning_details": [
                                {
                                    "type": "reasoning.summary",
                                    "summary": "Summarized chain",
                                },
                                {
                                    "type": "reasoning.text",
                                    "text": "Detailed chain",
                                },
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
            context,
            "anthropic/claude-sonnet-4.5",
            "reasoner",
        )

        self.assertEqual(response["content"], "Final answer")
        self.assertIn("Summarized chain", response["reasoning"])
        self.assertIn("Detailed chain", response["reasoning"])

    def test_openrouter_response_preserves_streaming_metadata(self) -> None:
        context = ProviderContext(
            logger=logging.getLogger("openrouter-stream-test"),
            providers_cfg={},
            model_configs={},
            request_timeout=30,
        )

        response = _process_openrouter_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Final answer",
                            "reasoning": "Step by step",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "_streaming": {
                    "reasoning_streamed": True,
                    "content_streamed": True,
                },
            },
            context,
            "openai/gpt-5.2",
            "reasoner",
        )

        self.assertTrue(response["reasoning_streamed"])
        self.assertTrue(response["content_streamed"])

    def test_console_stream_renderer_uses_compact_headings(self) -> None:
        stream = StringIO()
        console = Console(file=stream, force_terminal=False, color_system=None)
        renderer = _ConsoleStreamRenderer(
            console,
            request_type="pre_research",
            model="deepseek-reasoner",
        )

        renderer.emit_reasoning("Thinking...")
        renderer.emit_content("Answer...")
        renderer.finalize()

        output = stream.getvalue()
        self.assertIn("AI Stream | pre_research | deepseek-reasoner", output)
        self.assertIn("Reasoning", output)
        self.assertIn("Answer", output)
        self.assertNotIn("[thinking]", output)
        self.assertNotIn("[answer]", output)

    def test_pre_research_reflection_skips_reasoning_panel_when_streamed(self) -> None:
        agent = PreResearchAgent.__new__(PreResearchAgent)
        agent._reflection_enabled = True
        agent._suppress_reasoning_panel_after_stream = True
        agent.logger = logging.getLogger("pre-research-reflection-test")

        stream = StringIO()
        console = Console(file=stream, force_terminal=False, color_system=None)

        agent._emit_reflection(
            "Short reasoning",
            [],
            console,
            reasoning_streamed=True,
        )

        output = stream.getvalue()
        self.assertNotIn("Thinking Process", output)


if __name__ == "__main__":
    unittest.main()