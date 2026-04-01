import asyncio
import logging
import unittest
from io import StringIO
from unittest.mock import AsyncMock, patch

from rich.console import Console

from services.ai.providers.agent_guardrails import (
    build_tool_plan_preview,
    get_pre_research_pivot_notice,
    looks_like_bulk_list_fact,
    normalize_tool_args,
    sanitize_memory_args,
    should_block_bist_yfinance_search,
    validate_tool_args,
)
from services.ai.providers.deepseek_provider import (
    _ConsoleStreamRenderer,
    _prepare_deepseek_payload,
    _process_deepseek_response,
)
from services.ai.providers.native_tooling import (
    build_native_tool_request_kwargs,
    build_tool_result_history_message,
)
from services.ai.providers.openrouter_provider import (
    _emit_stream_delta as _emit_openrouter_stream_delta,
    _process_openrouter_response,
    _resolve_stream_state as _resolve_openrouter_stream_state,
)
from services.ai.providers.base_provider import ProviderContext
from services.ai.providers.pre_research_agent import PreResearchAgent
from services.ai.providers.prompt_store import PromptStore
from services.ai.providers.system_prompt_utils import augment_system_prompt
from services.ai.providers.tool_call_parser import (
    content_has_tool_call_markup,
    normalize_tool_calls,
    parse_tool_calls_from_content,
)


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
        self.assertIn("Use exact tool names and argument names", prompt)
        self.assertIn("Do NOT invent, rename, or simulate unavailable tools", prompt)
        self.assertIn("For BIST broad discovery, do not start with `yfinance_search`", prompt)


class SystemPromptAugmentationTests(unittest.TestCase):
    def test_augment_system_prompt_adds_runtime_capability_guidance(self) -> None:
        prompt = augment_system_prompt(
            "Base prompt.",
            config={
                "ai": {
                    "python_exec": {"enabled": True},
                    "report_tool": {"enabled": True},
                }
            },
            output_language="English",
        )

        self.assertIn("Tool availability is request-scoped", prompt)
        self.assertIn("call tools natively instead of printing JSON", prompt)
        self.assertIn("If a tool fails or returns empty data", prompt)
        self.assertIn("python_exec(code, input_data?, timeout_seconds?)", prompt)
        self.assertIn("report(title, category, severity, summary", prompt)


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

    def test_bist_broad_yfinance_search_is_blocked(self) -> None:
        reason = should_block_bist_yfinance_search(
            exchange="BIST",
            query="BIST 100 stocks low price",
            type_filter="stock",
        )

        self.assertIsNotNone(reason)
        self.assertIn("not reliable in yfinance", reason)

    def test_bist_specific_ticker_yfinance_search_is_allowed(self) -> None:
        reason = should_block_bist_yfinance_search(
            exchange="BIST",
            query="FROTO.IS",
            type_filter="stock",
        )

        self.assertIsNone(reason)

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


class ToolCallParserCompatibilityTests(unittest.TestCase):
    def test_normalize_native_tool_calls_preserves_id(self) -> None:
        tool_calls = normalize_tool_calls(
            [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "search_memory",
                        "arguments": '{"query": "AKBNK"}',
                    },
                }
            ]
        )

        self.assertEqual(tool_calls[0]["id"], "call_123")
        self.assertEqual(tool_calls[0]["type"], "function")
        self.assertEqual(tool_calls[0]["name"], "search_memory")
        self.assertEqual(tool_calls[0]["args"], {"query": "AKBNK"})

    def test_parse_legacy_tool_call_markup(self) -> None:
        content = (
            "I'll help you check prior context.\n"
            "<tool_call><function=search_memory>"
            "<parameter=query>BIST stocks relative low safe dividend</parameter>"
            "<parameter=detail_level>standard</parameter>"
            "</function></tool_call>"
        )

        tool_calls, cleaned = parse_tool_calls_from_content(content)

        self.assertEqual(
            tool_calls,
            [
                {
                    "name": "search_memory",
                    "args": {
                        "query": "BIST stocks relative low safe dividend",
                        "detail_level": "standard",
                    },
                }
            ],
        )
        self.assertEqual(cleaned, "I'll help you check prior context.")
        self.assertTrue(content_has_tool_call_markup(content))

    def test_parse_legacy_tool_call_markup_coerces_scalars(self) -> None:
        content = (
            "<tool_call><function=yfinance_search>"
            "<parameter=max_results>10</parameter>"
            "<parameter=include_etfs>false</parameter>"
            "</function></tool_call>"
        )

        tool_calls, cleaned = parse_tool_calls_from_content(content)

        self.assertEqual(tool_calls[0]["args"]["max_results"], 10)
        self.assertFalse(tool_calls[0]["args"]["include_etfs"])
        self.assertEqual(cleaned, "")


class ReasoningUxTests(unittest.TestCase):
    def test_native_tool_request_kwargs_include_tools_and_auto_choice(self) -> None:
        kwargs = build_native_tool_request_kwargs(
            "pre_research",
            max_parallel_tools=4,
        )

        self.assertTrue(kwargs["tools"])
        self.assertEqual(kwargs["tool_choice"], "auto")
        self.assertTrue(kwargs["parallel_tool_calls"])

    def test_build_tool_result_history_message_uses_role_tool_for_native_call(self) -> None:
        message = build_tool_result_history_message(
            tool_name="search_memory",
            result={"status": "ok"},
            tool_call={"id": "call_1", "name": "search_memory"},
        )

        self.assertEqual(message["role"], "tool")
        self.assertEqual(message["tool_call_id"], "call_1")
        self.assertIn('"status": "ok"', message["content"])

    def test_deepseek_payload_forwards_native_tools(self) -> None:
        context = ProviderContext(
            logger=logging.getLogger("deepseek-payload-test"),
            providers_cfg={"deepseek": {"api_key": "deepseek-key"}},
            model_configs={},
            request_timeout=30,
        )

        _, _, payload, _ = _prepare_deepseek_payload(
            [{"role": "user", "content": "hello"}],
            "news",
            context,
            tools=[{"type": "function", "function": {"name": "search_memory"}}],
            tool_choice="auto",
        )

        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["tools"][0]["function"]["name"], "search_memory")

    def test_openrouter_response_adds_missing_tool_call_ids(self) -> None:
        context = ProviderContext(
            logger=logging.getLogger("openrouter-native-tool-test"),
            providers_cfg={},
            model_configs={},
            request_timeout=30,
        )

        response = _process_openrouter_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "search_memory",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            context,
            "openai/gpt-5.2",
            "pre_research",
        )

        self.assertEqual(response["tool_calls"][0]["id"], "tool_call_0")

    def test_deepseek_response_adds_missing_tool_call_ids(self) -> None:
        context = ProviderContext(
            logger=logging.getLogger("deepseek-native-tool-test"),
            providers_cfg={},
            model_configs={},
            request_timeout=30,
        )

        response = _process_deepseek_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "search_memory",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            context,
            "deepseek-chat",
            "news",
        )

        self.assertEqual(response["tool_calls"][0]["id"], "tool_call_0")

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

    def test_openrouter_stream_preserves_leading_spaces_between_deltas(self) -> None:
        stream = StringIO()
        console = Console(file=stream, force_terminal=False, color_system=None)
        stream_state = _resolve_openrouter_stream_state(
            "pre_research",
            "stepfun/step-3.5-flash",
            console=console,
        )

        _emit_openrouter_stream_delta(stream_state, {"reasoning": "The"})
        _emit_openrouter_stream_delta(stream_state, {"reasoning": " user"})
        _emit_openrouter_stream_delta(stream_state, {"reasoning": " wants"})
        _emit_openrouter_stream_delta(stream_state, {"content": "I'll"})
        _emit_openrouter_stream_delta(stream_state, {"content": " help"})
        if stream_state.finalizer:
            stream_state.finalizer()

        output = stream.getvalue()
        self.assertIn("The user wants", output)
        self.assertIn("I'll help", output)
        self.assertNotIn("Theuserwants", output)
        self.assertNotIn("I'llhelp", output)

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


class PreResearchAgentAdaptiveFetchTests(unittest.TestCase):
    def _build_agent(self) -> PreResearchAgent:
        agent = PreResearchAgent.__new__(PreResearchAgent)
        agent.logger = logging.getLogger("pre-research-adaptive-test")
        agent._fetch_dedupe_window_hours = 24
        agent._fetched_urls = {}
        agent._adaptive_digest_enabled = True
        agent._adaptive_simple_threshold_chars = 3000
        agent._adaptive_complex_threshold_chars = 15000
        agent._adaptive_pruned_target_chars = 7000
        agent._current_exchange = "BIST"
        agent._output_lang = "English"
        agent._request_executor = AsyncMock()
        agent._request_executor.send_async = AsyncMock(return_value={"content": ""})
        agent._config = {"ai": {"rag": {"query_expansion": {"enabled": True}}}}
        agent.memory = type(
            "MemoryStub",
            (),
            {
                "to_dict": lambda self: {
                    "unanswered_questions": ["What are the near-term risks for SAHOL.IS?"],
                    "sources_consulted": ["SAHOL.IS investor presentation"],
                },
                "refresh_context": lambda self, *args, **kwargs: [{"id": "doc-1"}],
            },
        )()
        agent._url_pruning_tool = type(
            "PrunerStub",
            (),
            {
                "_prune_content": lambda self, **kwargs: {
                    "content": "Selected chunk",
                    "index_content": "Selected chunk",
                    "selection_stats": {
                        "mode": "query_pruned",
                        "selected_chars": 14,
                    },
                },
                "_format_pruned_content": lambda self, **kwargs: "RAG_PRUNED_CONTENT\nSelected chunk",
            },
        )()
        agent._index_fetch_result_to_rag = AsyncMock()
        agent._fetch_url_with_digest = AsyncMock(
            return_value={"url": "https://example.com", "digest": "LLM digest", "digest_strategy": "llm_digest"}
        )
        return agent

    def test_duplicate_fetch_is_detected_within_window(self) -> None:
        agent = self._build_agent()

        agent._mark_url_fetched("https://example.com")

        self.assertTrue(agent._is_duplicate_fetch("https://example.com"))

    def test_estimate_fetch_complexity_marks_large_structured_content_complex(self) -> None:
        agent = self._build_agent()

        complexity = agent._estimate_fetch_complexity(
            {
                "content": ("# Section\nRevenue 10\n" * 1200),
            },
            "Revenue outlook and debt profile",
        )

        self.assertEqual(complexity, "complex")

    def test_execute_tool_safe_skips_duplicate_fetch_before_tool_call(self) -> None:
        agent = self._build_agent()
        agent._update_working_memory = lambda *args, **kwargs: None
        agent._current_depth_score = lambda: 0
        agent._mark_url_fetched("https://example.com")

        result = asyncio.run(
            agent._execute_tool_safe(
                "fetch_url_content",
                {"url": "https://example.com"},
                console=None,
            )
        )

        self.assertIn("duplicate_fetch", result)
        self.assertIn("dedupe_skip", result)

    def test_bootstrap_memory_from_rag_loads_prior_news_context(self) -> None:
        agent = self._build_agent()
        rag = type("RAGStub", (), {"is_ready": lambda self: True})()

        with patch(
            "services.ai.providers.pre_research_agent.get_rag_service",
            return_value=rag,
        ):
            asyncio.run(
                agent._bootstrap_memory_from_rag(
                    exchange="BIST",
                    criteria="banking screen",
                    console=None,
                )
            )

        self.assertTrue(agent._index_fetch_result_to_rag.await_count == 0)


if __name__ == "__main__":
    unittest.main()