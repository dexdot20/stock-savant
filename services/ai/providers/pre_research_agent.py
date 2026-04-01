import json
import asyncio
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

from rich.console import Console

from config import get_config
from services.factories import get_rag_service, get_shared_memory_pool
from services.tools import SummarizeUrlContentTool, execute_tool
from services.tools import get_tool_quality_metrics, to_standard_tool_result
from services.ai.memory_formatter import format_working_memory_evidence_pack
from services.ai.providers.token_utils import estimate_tokens_for_messages
from services.ai.providers.memory_rag_utils import (
    derive_spill_confidence,
    derive_spill_data_gap_count,
    format_fact_for_rag,
    generate_hypothetical_rag_document,
)
from services.ai.providers.context_preservation_utils import (
    extract_tool_name_from_tool_result,
    normalize_history_archives,
)
from services.ai.providers.agent_session_utils import (
    generate_agent_session_id,
    load_agent_session,
    save_agent_session,
)
from services.ai.providers.system_prompt_utils import augment_system_prompt
from services.ai.providers.native_tooling import (
    build_native_tool_request_kwargs,
    build_tool_result_history_message,
)
from services.ai.providers.tool_call_parser import (
    content_has_tool_call_markup,
    normalize_tool_calls,
    parse_tool_calls_from_content,
)
from services.ai.providers.agent_guardrails import (
    VERIFIED_MEMORY_TOOLS,
    build_tool_plan_preview,
    canonicalize_bist_market_tool_call,
    is_bist_market_context,
    get_pre_research_pivot_notice,
    make_tool_signature,
    normalize_tool_args,
    should_block_bist_yfinance_search,
    validate_tool_args,
)
from services.ai.working_memory import WorkingMemory
from services.ai.providers.research_agent_support import ResearchAgentSupportMixin

# Search tools that produce large raw output and need early cleanup from history.
_EPHEMERAL_SEARCH_TOOLS: frozenset = frozenset(
    {"search_web", "search_news", "search_google_news"}
)

from domain.utils import safe_int_strict as _safe_int


class PreResearchAgent(ResearchAgentSupportMixin):
    """
    Autonomous pre-research agent.
    Performs web research for target exchange and lists found companies.
    """

    def __init__(self, logger, prompt_resolver, request_executor) -> None:
        self.logger = logger
        self._prompt_resolver = prompt_resolver
        self._request_executor = request_executor
        config = get_config()
        token_cfg = config.get("ai", {}).get("token_budget", {})
        self._token_budget_enabled = bool(token_cfg.get("enabled", True))
        self._history_token_limit = _safe_int(
            token_cfg.get("history_token_limit", 48000), 48000
        )
        self._token_encoding = str(token_cfg.get("encoding", "cl100k_base"))
        reflection_cfg = config.get("ai", {}).get("agent_reflection", {})
        self._reflection_enabled = bool(reflection_cfg.get("enabled", True))
        self._suppress_reasoning_panel_after_stream = bool(
            reflection_cfg.get("suppress_reasoning_panel_after_stream", True)
        )

        # Get max_steps from config or use default
        agent_steps_cfg = config.get("ai", {}).get("agent_steps", {})
        self.max_steps = _safe_int(agent_steps_cfg.get("pre_research", 50), 50)

        tool_limit_cfg = config.get("ai", {}).get("agent_tool_limits", {})
        raw_limit = _safe_int(tool_limit_cfg.get("max_parallel_tools", 0), 0)
        self._max_parallel_tools: Optional[int] = raw_limit if raw_limit > 0 else None

        # Cache config slice used in methods to avoid repeated deepcopy.
        self._output_lang: str = config.get("ai", {}).get("output_language", "English")
        self._config = config
        self._screening_pivot_notified = False
        summarizer_cfg = config.get("ai", {}).get("summarizer_settings", {})
        self._summarizer_timeout = _safe_int(
            summarizer_cfg.get("summarizer_timeout_seconds", 60), 60
        )
        network_cfg = config.get("network", {})
        self._step_request_timeout = float(
            network_cfg.get("request_timeout_seconds", 45) or 45
        )
        if self._step_request_timeout <= 0:
            self._step_request_timeout = 45.0

        working_memory_cfg = config.get("ai", {}).get("working_memory", {})
        shared_pool = (
            get_shared_memory_pool(config)
            if bool(working_memory_cfg.get("shared_pool_enabled", True))
            else None
        )

        self.memory = WorkingMemory(
            max_facts=_safe_int(working_memory_cfg.get("max_facts", 60), 60),
            adaptive_max_facts=_safe_int(
                working_memory_cfg.get("adaptive_max_facts", 90), 90
            ),
            adaptive_usage_step=float(
                working_memory_cfg.get("adaptive_usage_step", 0.2) or 0.2
            ),
            max_sources=_safe_int(working_memory_cfg.get("max_sources", 30), 30),
            max_questions=_safe_int(
                working_memory_cfg.get("max_questions", 20), 20
            ),
            max_contradictions=_safe_int(
                working_memory_cfg.get("max_contradictions", 20), 20
            ),
            max_evidence_records=_safe_int(
                working_memory_cfg.get("max_evidence_records", 24), 24
            ),
            consolidation_threshold=_safe_int(
                working_memory_cfg.get("consolidation_threshold", 45), 45
            ),
            fact_similarity_threshold=float(
                working_memory_cfg.get("fact_similarity_threshold", 0.82) or 0.82
            ),
            fact_similarity_window=_safe_int(
                working_memory_cfg.get("fact_similarity_window", 12), 12
            ),
            consolidation_similarity_ratio=float(
                working_memory_cfg.get("consolidation_similarity_ratio", 0.35)
                or 0.35
            ),
            consolidation_cache_size=_safe_int(
                working_memory_cfg.get("consolidation_cache_size", 32), 32
            ),
            importance_recalc_interval_seconds=_safe_int(
                working_memory_cfg.get("importance_recalc_interval_seconds", 120),
                120,
            ),
            shared_memory_pool=shared_pool,
            shared_memory_agent_name="pre_research_agent",
        )
        # fetch_url_content + summarize_url_content outputs: last 2 steps kept in history.
        self._fetch_indices: List[int] = []
        # search_web / search_news / search_google_news outputs: last 1 step kept.
        self._search_indices: List[int] = []
        # Progress message pruning: previous [Progress: ...] message replaced with stub.
        self._progress_msg_index: Optional[int] = None
        # Consolidation: tracks the step when consolidation was last performed.
        self._last_consolidation_step: int = 0
        self._current_exchange: str = ""
        self._history_archives: List[Dict[str, Any]] = []
        self._tool_journal: List[Dict[str, Any]] = []
        adaptive_digest_cfg = config.get("ai", {}).get("pre_research", {}).get(
            "adaptive_digest", {}
        )
        self._adaptive_digest_enabled = bool(adaptive_digest_cfg.get("enabled", True))
        self._adaptive_simple_threshold_chars = _safe_int(
            adaptive_digest_cfg.get("simple_threshold_chars", 3000), 3000
        )
        self._adaptive_complex_threshold_chars = _safe_int(
            adaptive_digest_cfg.get("complex_threshold_chars", 15000), 15000
        )
        self._adaptive_pruned_target_chars = _safe_int(
            adaptive_digest_cfg.get("pruned_target_chars", 7000), 7000
        )
        self._fetch_dedupe_window_hours = max(
            1,
            _safe_int(
                config.get("ai", {})
                .get("pre_research", {})
                .get("fetch_dedupe_window_hours", 24),
                24,
            ),
        )
        self._fetched_urls: Dict[str, str] = {}
        self._url_pruning_tool = SummarizeUrlContentTool()

    def _prune_recent_fetches(self) -> None:
        if not self._fetched_urls:
            return

        cutoff = datetime.now(timezone.utc) - timedelta(
            hours=self._fetch_dedupe_window_hours
        )
        retained: Dict[str, str] = {}
        for url, raw_timestamp in self._fetched_urls.items():
            try:
                fetched_at = datetime.fromisoformat(str(raw_timestamp))
            except Exception:
                continue
            if fetched_at >= cutoff:
                retained[url] = str(raw_timestamp)
        self._fetched_urls = retained

    def _is_duplicate_fetch(self, url: str) -> bool:
        normalized_url = str(url or "").strip()
        if not normalized_url:
            return False
        self._prune_recent_fetches()
        last_fetch = self._fetched_urls.get(normalized_url)
        if not last_fetch:
            return False
        try:
            fetched_at = datetime.fromisoformat(last_fetch)
        except Exception:
            return False
        return fetched_at >= (
            datetime.now(timezone.utc)
            - timedelta(hours=self._fetch_dedupe_window_hours)
        )

    def _mark_url_fetched(self, url: str) -> None:
        normalized_url = str(url or "").strip()
        if not normalized_url:
            return
        self._prune_recent_fetches()
        self._fetched_urls[normalized_url] = datetime.now(timezone.utc).isoformat()

    def _build_fetch_focus_query(self, explicit_focus_query: Optional[str]) -> str:
        explicit = str(explicit_focus_query or "").strip()
        if explicit:
            return explicit

        questions = self.memory.to_dict().get("unanswered_questions", [])
        normalized = [str(item).strip() for item in questions if str(item).strip()]
        if normalized:
            return "\n".join(normalized[:8])

        if self._current_exchange:
            return (
                f"{self._current_exchange} market catalysts, risks, financial metrics, "
                "disclosures"
            )
        return "key financial metrics, catalysts, risks, disclosures"

    async def _generate_fetch_query_hypothesis(self, focus_query: str) -> str:
        normalized_focus = str(focus_query or "").strip()
        if not normalized_focus:
            return ""

        try:
            return await generate_hypothetical_rag_document(
                normalized_focus,
                request_executor=self._request_executor,
                language=self._output_lang,
                timeout_seconds=10.0,
                max_chars=500,
            )
        except Exception as exc:
            self.logger.debug("Fetch HyDE hypothesis skipped: %s", exc)
            return ""

    def _estimate_fetch_complexity(
        self,
        raw_result: Dict[str, Any],
        focus_query: str,
    ) -> str:
        content = str(raw_result.get("content") or "")
        content_length = len(content)
        heading_count = len(
            re.findall(r"(^|\n)(#{1,6}\s|[A-Z][^\n]{0,80}:)", content)
        )
        numeric_hits = len(
            re.findall(r"\b\d[\d,.:/%$EURTRYTLMBKmbk-]*\b", content)
        )
        question_count = len(self.memory.to_dict().get("unanswered_questions", []))

        complexity_score = 0
        if content_length >= self._adaptive_complex_threshold_chars:
            complexity_score += 3
        elif content_length <= self._adaptive_simple_threshold_chars:
            complexity_score -= 1
        else:
            complexity_score += 1
        if heading_count >= 8:
            complexity_score += 1
        if numeric_hits >= 80:
            complexity_score += 1
        if question_count >= 5:
            complexity_score += 1
        if len(str(focus_query or "").strip()) >= 160:
            complexity_score += 1

        if complexity_score <= 0:
            return "simple"
        if complexity_score >= 4:
            return "complex"
        return "moderate"

    def _detect_fetch_symbol(self, payload: Dict[str, Any]) -> Optional[str]:
        memory_state = self.memory.to_dict()
        candidates: List[str] = []
        for key in ("symbol", "ticker"):
            value = memory_state.get(key)
            if value:
                candidates.append(str(value))
        for source in memory_state.get("sources_consulted", []):
            source_str = str(source or "").strip()
            if source_str:
                candidates.append(source_str)
        title = str(payload.get("title") or payload.get("headline") or "").strip()
        if title:
            candidates.append(title)

        for candidate in candidates:
            normalized = str(candidate).strip().upper()
            if not normalized:
                continue
            bist_match = re.search(r"\b([A-Z]{2,10})\.IS\b", normalized)
            if bist_match:
                return f"{bist_match.group(1)}.IS"
            us_match = re.search(r"\b([A-Z]{2,5})\b", normalized)
            if us_match and us_match.group(1) not in {"HTTP", "HTTPS", "BIST"}:
                return us_match.group(1)
        return None

    def _build_fetch_index_content(
        self,
        payload: Dict[str, Any],
        strategy: str,
        focus_query: str,
    ) -> str:
        if payload.get("index_content"):
            base = str(payload.get("index_content") or "").strip()
        elif payload.get("digest"):
            base = str(payload.get("digest") or "").strip()
        else:
            base = str(payload.get("content") or "").strip()
        if not base:
            return ""

        questions = [
            str(item).strip()
            for item in self.memory.to_dict().get("unanswered_questions", [])
            if str(item).strip()
        ]
        parts = [
            f"Fetch strategy: {strategy}",
            f"Exchange: {self._current_exchange or 'UNKNOWN'}",
        ]
        if focus_query:
            parts.append(f"Focus query: {focus_query}")
        if questions:
            parts.append("Open questions:\n" + "\n".join(f"- {item}" for item in questions[:8]))
        parts.append(base)
        return "\n\n".join(part for part in parts if part).strip()

    async def _index_fetch_result_to_rag(
        self,
        *,
        url: str,
        payload: Dict[str, Any],
        strategy: str,
        focus_query: str,
        console: Optional[Console],
    ) -> None:
        normalized_url = str(url or "").strip()
        if not normalized_url:
            return

        index_content = self._build_fetch_index_content(payload, strategy, focus_query)
        if not index_content:
            return

        try:
            rag = get_rag_service()
        except Exception as exc:
            self.logger.debug("Fetch RAG indexing skipped: %s", exc)
            return

        if not rag.is_ready() or rag.has_document_url("pre_research", normalized_url):
            return

        indexed_count = rag.index_pre_research(
            exchange=self._current_exchange or "UNKNOWN",
            markdown_content=index_content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence_score=0.62 if strategy == "llm_digest" else 0.56,
            symbol=self._detect_fetch_symbol(payload),
            url=normalized_url,
            doc_type="fetch_digest",
        )
        if indexed_count > 0:
            self.logger.info(
                "Fetch result indexed into RAG via %s: %s (%d chunks)",
                strategy,
                normalized_url,
                indexed_count,
            )
            if console:
                console.print(
                    f"[dim]📚 Fetch RAG record: {indexed_count} chunk ({strategy})[/dim]"
                )

    def _build_skip_fetch_payload(self, url: str) -> Dict[str, Any]:
        return {
            "url": str(url or "").strip(),
            "content": "",
            "is_summarized": True,
            "is_retrieval_pruned": False,
            "skip_reason": "duplicate_fetch",
            "digest_strategy": "dedupe_skip",
        }

    def _apply_rag_pruning_to_fetch_result(
        self,
        *,
        raw_result: Dict[str, Any],
        url: str,
        focus_query: str,
        query_hypothesis: str,
    ) -> Dict[str, Any]:
        content = str(raw_result.get("content") or "").strip()
        if not content:
            return dict(raw_result)

        pruned_payload = self._url_pruning_tool._prune_content(
            content=content,
            focus_query=focus_query,
            query_hypothesis=query_hypothesis or None,
        )
        pruned_text = str((pruned_payload or {}).get("content") or "").strip()
        if not pruned_text:
            return dict(raw_result)

        selection_stats = (
            (pruned_payload or {}).get("selection_stats")
            if isinstance((pruned_payload or {}).get("selection_stats"), dict)
            else {}
        )
        payload = dict(raw_result)
        payload["content"] = self._url_pruning_tool._format_pruned_content(
            content=pruned_text,
            source_length=len(content),
            selection_stats=selection_stats,
        )
        payload["index_content"] = str(
            (pruned_payload or {}).get("index_content") or pruned_text
        ).strip()
        payload["url"] = str(url or payload.get("url") or "")
        payload["is_summarized"] = True
        payload["is_retrieval_pruned"] = True
        payload["selection_mode"] = selection_stats.get("mode")
        payload["selection_stats"] = selection_stats
        payload["digest_strategy"] = "rag_pruned"
        return payload

    async def _bootstrap_memory_from_rag(
        self,
        *,
        exchange: str,
        criteria: Optional[str],
        console: Optional[Console],
    ) -> None:
        try:
            rag = get_rag_service()
        except Exception as exc:
            self.logger.debug("Pre-research RAG warm-start skipped: %s", exc)
            return

        if not rag.is_ready():
            return

        base_query = " ".join(
            part
            for part in [exchange, criteria or "", "market screening news disclosures analysis"]
            if str(part).strip()
        ).strip()
        if not base_query:
            return

        hypothesis = ""
        rag_cfg = self._config.get("ai", {}).get("rag", {})
        if bool((rag_cfg.get("query_expansion", {}) or {}).get("enabled", True)):
            hypothesis = await self._generate_fetch_query_hypothesis(base_query)

        working_memory_cfg = self._config.get("ai", {}).get("working_memory", {})
        warm_hits = self.memory.refresh_context(
            base_query,
            rag_service=rag,
            top_k=_safe_int(working_memory_cfg.get("refresh_context_top_k", 5), 5),
            recent_days=_safe_int(
                working_memory_cfg.get("refresh_context_recent_days", 30),
                30,
            ),
            context_window=_safe_int(
                working_memory_cfg.get("refresh_context_window", 1),
                1,
            ),
            importance=6,
            tags=["rag_retrieved", "prior_news", "prior_pre_research"],
            preview_chars=700,
            milestone="RAG warm-start completed at pre-research start",
            query_hypothesis=hypothesis or None,
        )
        if warm_hits and console:
            console.print(
                f"[dim]🧠 Pre-research RAG warm-start: {len(warm_hits)} prior findings loaded.[/dim]"
            )

    async def _post_process_fetch_result(
        self,
        *,
        args: Dict[str, Any],
        raw_result: Dict[str, Any],
        console: Optional[Console],
    ) -> Dict[str, Any]:
        url = str(args.get("url") or raw_result.get("url") or "").strip()
        focus_query = self._build_fetch_focus_query(args.get("focus_query"))
        strategy_mode = "complex"
        hypothesis = ""
        if not self._adaptive_digest_enabled:
            payload = await self._fetch_url_with_digest(
                url=url,
                raw_result=raw_result,
                fallback_result_str=raw_result,
                console=console,
            )
            await self._index_fetch_result_to_rag(
                url=url,
                payload=payload,
                strategy=str(payload.get("digest_strategy") or "llm_digest"),
                focus_query=focus_query,
                console=console,
            )
            if console:
                console.print(
                    f"[dim]🧭 Fetch strategy: {payload.get('digest_strategy', 'llm_digest')}[/dim]"
                )
            return payload

        strategy_mode = self._estimate_fetch_complexity(raw_result, focus_query)
        if strategy_mode != "simple":
            hypothesis = await self._generate_fetch_query_hypothesis(focus_query)
        payload = dict(raw_result)

        if strategy_mode in {"moderate", "complex"}:
            payload = self._apply_rag_pruning_to_fetch_result(
                raw_result=raw_result,
                url=url,
                focus_query=focus_query,
                query_hypothesis=hypothesis,
            )

        if self._adaptive_digest_enabled and strategy_mode == "complex":
            selection_stats = (
                payload.get("selection_stats") if isinstance(payload.get("selection_stats"), dict) else {}
            )
            selected_chars = int(
                selection_stats.get("selected_chars", len(str(payload.get("content") or "")))
                or 0
            )
            if selected_chars > self._adaptive_pruned_target_chars:
                payload = await self._fetch_url_with_digest(
                    url=url,
                    raw_result=raw_result,
                    fallback_result_str=payload,
                    console=console,
                )
            else:
                payload["digest_strategy"] = str(
                    payload.get("digest_strategy") or "rag_pruned"
                )
        elif self._adaptive_digest_enabled and strategy_mode == "simple":
            payload["digest_strategy"] = "raw"
            payload["url"] = url
        else:
            payload["digest_strategy"] = str(
                payload.get("digest_strategy") or "rag_pruned"
            )

        await self._index_fetch_result_to_rag(
            url=url,
            payload=payload,
            strategy=str(payload.get("digest_strategy") or "raw"),
            focus_query=focus_query,
            console=console,
        )
        if console:
            console.print(
                f"[dim]🧭 Fetch strategy: {payload.get('digest_strategy', strategy_mode)}[/dim]"
            )
        return payload

    def _build_partial_report(self) -> str:
        partial_report = (
            "# ⚠️ Maximum Process Limit Reached\n\n"
            "Agent reached the step limit and could not complete the research fully. "
            "However, the data collected so far is below:\n\n"
        )

        memory_state = self.memory.to_dict()
        facts = [
            self._format_fact_text(fact)
            for fact in memory_state.get("facts_learned", [])
        ]
        facts = [fact for fact in facts if fact]
        if facts:
            partial_report += "## 💡 Learned Findings\n"
            partial_report += "\n".join(f"- {fact}" for fact in facts)
            partial_report += "\n\n"

        sources = [
            str(source).strip()
            for source in memory_state.get("sources_consulted", [])
            if str(source).strip()
        ]
        if sources:
            partial_report += "## 📚 Consulted Sources\n"
            partial_report += "\n".join(f"- {source}" for source in sources)
            partial_report += "\n\n"

        questions = [
            str(question).strip()
            for question in memory_state.get("unanswered_questions", [])
            if str(question).strip()
        ]
        if questions:
            partial_report += "## ❓ Unanswered Questions\n"
            partial_report += "\n".join(f"- {question}" for question in questions)

        return partial_report

    def _save_session_snapshot(
        self,
        session_id: str,
        *,
        history: List[Dict[str, Any]],
        step: int,
        exchange: str,
        criteria: Optional[str],
        depth_mode: str,
    ) -> str:
        payload: Dict[str, Any] = {
            "exchange": exchange,
            "criteria": criteria,
            "depth_mode": depth_mode,
            "steps_completed": int(step),
            "working_memory": self.memory.to_dict(),
            "evidence_pack": format_working_memory_evidence_pack(
                self.memory.to_dict(),
                max_facts=None,
                max_contradictions=None,
                max_questions=None,
                max_sources=None,
            ),
            "tool_journal": list(self._tool_journal),
            "state": {
                "history": history,
                "history_archives": list(self._history_archives),
                "step": int(step),
                "max_steps": int(self.max_steps),
                "memory": self.memory.to_dict(),
                "exchange": exchange,
                "criteria": criteria,
                "depth_mode": depth_mode,
                "tool_journal": list(self._tool_journal),
            },
        }
        return save_agent_session("pre_research_agent", session_id, payload)

    def _emit_reflection(
        self,
        reasoning: Optional[str],
        tool_plans: list[Dict[str, Any]],
        console: Optional[Console],
        reasoning_streamed: bool = False,
    ) -> None:
        if not self._reflection_enabled:
            return

        # 1. Native Reasoning Tokens — the model's actual thinking process
        if reasoning and reasoning.strip():
            self.logger.info("AI Reasoning: %d characters", len(reasoning))
            should_render_reasoning_panel = not (
                reasoning_streamed and self._suppress_reasoning_panel_after_stream
            )
            if console and should_render_reasoning_panel:
                from rich.panel import Panel
                from rich.markdown import Markdown

                md = Markdown(reasoning.strip())
                panel = Panel(
                    md,
                    title="💭 [bold blue]Thinking Process[/bold blue]",
                    border_style="blue",
                    padding=(0, 1),
                )
                console.print(panel)

        # 2. Tool Details
        if tool_plans:
            tool_info = []
            active_count = 0
            memory_count = 0
            deferred_count = 0
            for plan in tool_plans:
                name = plan.get("name", "unknown")
                args_str = json.dumps(plan.get("args", {}), ensure_ascii=False)
                status = str(plan.get("status") or "planned")
                if status == "execute_now":
                    active_count += 1
                elif status == "memory_update":
                    memory_count += 1
                elif status == "deferred":
                    deferred_count += 1
                tool_info.append(f"{name}({args_str}) [{status}]")

            reflection_msg = (
                f"Reflection: {active_count} tools executing now"
                + (f", {memory_count} memory updates" if memory_count else "")
                + (f", {deferred_count} deferred" if deferred_count else "")
                + f" -> {', '.join(tool_info)}"
            )
            self.logger.info(reflection_msg)

            if console:
                from rich.table import Table

                # Display tool calls as a table
                table = Table(
                    title=(
                        f"🎯 [bold cyan]Plan[/bold cyan]: {active_count} tools executing now"
                        + (f" • {memory_count} memory updates" if memory_count else "")
                        + (f" • {deferred_count} deferred" if deferred_count else "")
                    ),
                    show_header=True,
                    header_style="bold cyan",
                    border_style="cyan",
                    title_style="bold cyan",
                )
                table.add_column("#", style="dim", width=3, justify="right")
                table.add_column("Tool", style="bold green", no_wrap=True)
                table.add_column("Status", style="bold", no_wrap=True)
                table.add_column("Parameters", style="white")

                for i, plan in enumerate(tool_plans, 1):
                    name = plan.get("name", "unknown")
                    try:
                        args = plan.get("args", {})
                        if not isinstance(args, dict):
                            args = {}
                        # Display parameters more cleanly (truncate long ones)
                        args_parts = []
                        for k, v in args.items():
                            v_str = str(v)
                            if len(v_str) > 60:
                                v_str = v_str[:57] + "..."
                            args_parts.append(
                                f"[yellow]{k}[/yellow]=[dim]{v_str}[/dim]"
                            )
                        args_display = "\n".join(args_parts)
                    except (TypeError, ValueError):
                        args_display = "{}"

                    status = str(plan.get("status") or "planned")
                    note = str(plan.get("note") or "").strip()
                    status_label_map = {
                        "execute_now": "[green]running now[/green]",
                        "deferred": "[yellow]deferred[/yellow]",
                        "memory_update": "[magenta]memory[/magenta]",
                        "finish": "[blue]finish[/blue]",
                        "skipped_duplicate": "[yellow]dup-skip[/yellow]",
                        "blocked": "[red]blocked[/red]",
                    }
                    status_display = status_label_map.get(status, status)
                    if note:
                        status_display = f"{status_display}\n[dim]{note}[/dim]"

                    table.add_row(str(i), name, status_display, args_display)

                console.print(table)
                console.print()  # Empty line

    async def _consolidate_working_memory_if_needed(
        self, step: int, console: Optional[Console] = None
    ) -> bool:
        """If facts_learned exceeds _CONSOLIDATION_THRESHOLD, uses LLM (summarizer)
        to compress information into fewer, denser statements.
        Not re-triggered unless at least 5 steps have passed (excessive cost prevention).
        Returns True if consolidation was performed.
        """
        if not self.memory.needs_facts_consolidation():
            return False
        if step - self._last_consolidation_step < 5:
            return False

        fact_records = list(self.memory.to_dict().get("facts_learned", []))
        facts = [self._format_fact_text(fact) for fact in fact_records if self._format_fact_text(fact)]
        original_count = len(facts)
        self.logger.info(
            "Memory consolidation starting: %d facts being compressed...",
            original_count,
        )

        output_lang = self._output_lang

        system_prompt = augment_system_prompt(
            (
            "You are a memory consolidation assistant. "
            "Given a list of research facts, merge semantically related ones into fewer, denser statements. "
            "Rules: "
            "(1) Preserve ALL specific numbers, names, dates, percentages, and financial figures VERBATIM. "
            "(2) Each consolidated fact must be self-contained and information-dense. "
            "(3) Do NOT drop any unique data point. "
            "(4) Return ONLY a valid JSON array of strings — no explanation, no markdown fence. "
            f"(5) Respond in {output_lang}. "
            "Aim to reduce the count by at least 30% while retaining 100% of the information."
            ),
            config=self._config,
            output_language=output_lang,
        )
        facts_json = json.dumps(facts, ensure_ascii=False)
        user_content = f"Consolidate these {original_count} facts into a smaller, denser list:\n\n{facts_json}"

        try:
            response = await self._request_executor.send_async(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                request_type="summarizer",
                timeout_override=self._summarizer_timeout,
            )
            raw = (response.get("content") or "").strip()
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                self.logger.warning(
                    "Memory consolidation: JSON array not found, skipping."
                )
                return False
            consolidated = json.loads(match.group())
            if not isinstance(consolidated, list) or not consolidated:
                self.logger.warning("Memory consolidation: Invalid output, skipping.")
                return False

            try:
                rag = get_rag_service()
                rag.index_pre_research(
                    exchange=self._current_exchange or "UNKNOWN",
                    markdown_content="\n".join(
                        item
                        for item in (format_fact_for_rag(fact) for fact in fact_records)
                        if item
                    ),
                    confidence_score=derive_spill_confidence(fact_records, default=0.5),
                    data_gaps_count=derive_spill_data_gap_count(fact_records),
                )
            except Exception as rag_exc:
                self.logger.debug(
                    "Pre-research spill-over indexing skipped: %s", rag_exc
                )

            self.memory.replace_facts([str(f) for f in consolidated])
            new_count = self.memory.summary_counts()["facts"]
            self._last_consolidation_step = step
            savings_pct = round((1 - new_count / original_count) * 100)
            self.logger.info(
                "Memory consolidation completed: %d → %d facts (%%%d savings)",
                original_count,
                new_count,
                savings_pct,
            )
            if console:
                console.print(
                    f"[magenta]🧠 Memory Consolidation: {original_count} → {new_count} facts "
                    f"([green]+%{savings_pct}[/green] tasarruf)[/magenta]"
                )
            return True
        except Exception as exc:
            self.logger.warning("Memory consolidation failed: %s", exc)
            return False

    async def run(
        self,
        exchange: str,
        criteria: Optional[str] = None,
        console: Optional[Console] = None,
        depth_mode: str = "standard",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._current_exchange = str(exchange or "").strip().upper()
        self.memory.configure_shared_memory(
            scope=f"exchange:{self._current_exchange}",
            agent_name="pre_research_agent",
        )
        self.memory.refresh_from_shared_pool(
            f"{self._current_exchange} market pre research screening signals",
            top_k=_safe_int(
                self._config.get("ai", {})
                .get("working_memory", {})
                .get("shared_pool_search_top_k", 4),
                4,
            ),
            tags=["shared_pool", "prior_agent_memory"],
            milestone="Shared memory warm-start completed at pre-research start",
        )
        active_session_id = str(session_id or "").strip() or generate_agent_session_id(
            "pre-research", self._current_exchange or exchange
        )
        session_path = ""
        resumed_from_snapshot = False
        self._fetch_indices = []
        self._search_indices = []
        self._progress_msg_index = None
        self._history_archives = []
        self._tool_journal = []

        loaded_state: Optional[Dict[str, Any]] = None
        if session_id:
            snapshot = load_agent_session("pre_research_agent", active_session_id)
            raw_state = snapshot.get("state") if isinstance(snapshot, dict) else None
            if isinstance(raw_state, dict):
                loaded_state = raw_state
                resumed_from_snapshot = True

        if loaded_state:
            exchange = str(loaded_state.get("exchange") or exchange)
            criteria = (
                str(loaded_state.get("criteria"))
                if loaded_state.get("criteria") is not None
                else criteria
            )
            depth_mode = str(loaded_state.get("depth_mode") or depth_mode)
            self.memory.reset(keep_facts=False)
            raw_memory = loaded_state.get("memory")
            if isinstance(raw_memory, dict):
                self.memory.from_dict(raw_memory)
            history = (
                loaded_state.get("history")
                if isinstance(loaded_state.get("history"), list)
                else []
            )
            step = _safe_int(loaded_state.get("step", 0), 0)
            self.max_steps = _safe_int(
                loaded_state.get("max_steps", self.max_steps), self.max_steps
            )
            self._history_archives = normalize_history_archives(
                loaded_state.get("history_archives", [])
            )
            self._restore_tool_journal(loaded_state.get("tool_journal", []))
            self._restore_ephemeral_tracking(
                history,
                search_tools=_EPHEMERAL_SEARCH_TOOLS,
                url_tools=frozenset({"fetch_url_content", "summarize_url_content"}),
            )
            self._restore_progress_tracking(history)
            if console:
                console.print(
                    "[bold green]🔄 Resuming pre-research from saved session...[/bold green]"
                )
        else:
            self.memory.reset(keep_facts=True)
            self._history_archives = []
            self._tool_journal = []
            step = 0

        await self._bootstrap_memory_from_rag(
            exchange=self._current_exchange or exchange,
            criteria=criteria,
            console=console,
        )

        if console:
            console.print(
                f"[dim]🧠 Pre-research planning is running with a {self._step_request_timeout:.0f}s AI timeout.[/dim]"
            )

        today = datetime.now().strftime("%d %B %Y")
        try:
            system_prompt = self._prompt_resolver("pre_research_agent")
            # Inject output language from settings
            system_prompt = system_prompt.replace("{output_language}", self._output_lang)
        except Exception:
            self.logger.warning(
                "pre_research_agent prompt could not be resolved, default prompt will be used."
            )
            system_prompt = "You are a financial research agent. Use tools to research and summarize."

        depth_instructions = ""
        if depth_mode == "deep":
            depth_instructions = (
                "\n\n**DEEP RESEARCH MODE:** Use at least 5 different sources, "
                "read at least 2 full contents, verify at least 1 claim and "
                "search for at least 1 conflicting viewpoint."
            )

        criteria_text = criteria.strip() if isinstance(criteria, str) else ""

        # Build stable prefix for cache optimization:
        # - Static system prompt (no dynamic content)
        # - Stable first user message with only the task instruction
        # - Dynamic content (date, exchange, criteria) in a FOLLOW-UP message
        # This ensures cache hits on the prefix across requests
        stable_user_prefix = (
            f"{self._build_tool_reliability_notice()}"
            "Start research, rank all suitable companies you find according to criteria."
        )

        dynamic_context_message = (
            f"Date: {today}\n"
            f"Target Exchange: {exchange}\n"
            f"Additional Criteria: {criteria_text or 'None'}\n"
            f"{depth_instructions}"
        )

        if is_bist_market_context(self._current_exchange):
            dynamic_context_message += (
                "\n\nBIST guidance: use English `search_web` queries first for discovery "
                "and use KAP when official disclosures matter. Do not use `yfinance_search` "
                "for broad screening; reserve it for a specific ticker such as `FROTO.IS` or "
                "an index symbol such as `XU100.IS`. If a web search returns mostly US-centric "
                "results, refine the query with a BIST-specific ticker, company name, or index symbol."
            )

        if not loaded_state:
            history = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": stable_user_prefix},
                {"role": "user", "content": dynamic_context_message},
            ]

        total_non_memory_tools_executed = 0
        total_memory_updates = 0
        if loaded_state:
            (
                total_non_memory_tools_executed,
                total_memory_updates,
            ) = self._restore_execution_counters(history)
        consecutive_no_tool_calls = 0
        # Track (tool_name, args_hash) to detect and skip identical duplicate calls.
        _NON_DEDUP_TOOLS = {"update_working_memory", "finish", "search_memory"}
        executed_tool_signatures: set[str] = set()
        tool_request_kwargs = build_native_tool_request_kwargs(
            "pre_research",
            max_parallel_tools=self._max_parallel_tools,
        )

        while step < self.max_steps:
            step += 1
            self.logger.info("Pre-Research Step %d/%d", step, self.max_steps)

            total_chars = sum(
                len(str(m.get("content", ""))) + len(str(m.get("tool_calls", "")))
                for m in history
            )
            if self._token_budget_enabled:
                token_count = estimate_tokens_for_messages(
                    history, self._token_encoding
                )
                if token_count > self._history_token_limit:
                    history = await self._summarize_history(history)
            self._maybe_inject_self_reflection(history, step)
            # elif total_chars > self._history_char_limit:
            #    history = await self._summarize_history(history)

            if console:
                console.print(
                    f"[dim]🧠 Pre-Research Step {step}/{self.max_steps}: requesting AI plan...[/dim]"
                )

            try:
                response = await self._request_executor.send_async(
                    history,
                    request_type="pre_research",
                    timeout_override=self._step_request_timeout,
                    console=console,
                    **tool_request_kwargs,
                )
            except Exception as e:
                self.logger.error("Pre-research AI request failed: %s", e)
                session_path = self._save_session_snapshot(
                    active_session_id,
                    history=history,
                    step=step,
                    exchange=exchange,
                    criteria=criteria,
                    depth_mode=depth_mode,
                )
                return {
                    "error": str(e),
                    "analysis": "AI Request Failed",
                    "session_id": active_session_id,
                    "session_path": session_path,
                    "resumed_from_snapshot": resumed_from_snapshot,
                }

            raw_content = response.get("content", "")
            if not isinstance(raw_content, str):
                raw_content = ""
            tool_calls, content = parse_tool_calls_from_content(raw_content)
            native_tool_call_payload = (
                response.get("tool_calls")
                if isinstance(response.get("tool_calls"), list)
                else None
            )
            native_tool_calls = normalize_tool_calls(native_tool_call_payload)
            if native_tool_calls:
                tool_calls = native_tool_calls
            if content.strip() in (
                "{'format': {'type': 'text'}}",
                '{"format": {"type": "text"}}',
            ):
                content = ""

            assistant_history_message = {"role": "assistant", "content": raw_content}
            if native_tool_calls:
                assistant_history_message["tool_calls"] = native_tool_call_payload
            history.append(assistant_history_message)

            # Compress temporary (ephemeral) tool outputs marked in previous step.
            # fetch_url_content: last 2 outputs kept in history.
            # search_*: last 1 output kept (snippets saved to WM).
            _COMPRESSED_STUB = json.dumps(
                {
                    "status": "ephemeral_compressed",
                    "info": "Raw content was processed and removed to save tokens. Key findings are in working memory.",
                }
            )
            for _indices_list, _keep in (
                (self._fetch_indices, 2),
                (self._search_indices, 1),
            ):
                if len(_indices_list) > _keep:
                    for _idx in _indices_list[:-_keep]:
                        if (
                            _idx < len(history)
                            and history[_idx].get("role") in {"user", "tool"}
                        ):
                            tool_content = history[_idx].get("content", "")
                            assistant_msg = self._find_previous_assistant(history, _idx)
                            self._capture_ephemeral_result_before_stub(
                                tool_name=extract_tool_name_from_tool_result(
                                    history[_idx],
                                    history=history,
                                    message_index=_idx,
                                ),
                                tool_content=tool_content,
                                assistant_has_memory_update=self._assistant_has_memory_update(
                                    assistant_msg
                                ),
                            )
                            history[_idx]["content"] = _COMPRESSED_STUB
                    del _indices_list[:-_keep]

            if not tool_calls:
                if not content:
                    self._record_tool_journal_step(
                        step,
                        [],
                        assistant_summary="",
                        notes=["empty response; requested a tool call or finish()"],
                    )
                    history.append(
                        {
                            "role": "user",
                            "content": "Error: Empty response. Please call a tool or call finish() with your report.",
                        }
                    )
                    continue

                # Is it a true final report or incomplete/hallucination?
                is_final_report = len(content) > 600 and (
                    content.strip().startswith("#") or "\n##" in content
                )
                if is_final_report and step > 1:
                    self._record_tool_journal_step(
                        step,
                        [],
                        assistant_summary=content,
                        notes=["assistant returned a final report without finish()"],
                    )
                    await self.memory.flush()
                    session_path = self._save_session_snapshot(
                        active_session_id,
                        history=history,
                        step=step,
                        exchange=exchange,
                        criteria=criteria,
                        depth_mode=depth_mode,
                    )
                    return {
                        "analysis": content,
                        "steps": step,
                        "model_used": response.get("model", "unknown"),
                        "confidence": 1.0,
                        "timestamp": datetime.now().timestamp(),
                        "working_memory": self.memory.to_dict(),
                        "step_limit_reached": False,
                        "session_id": active_session_id,
                        "session_path": session_path,
                        "resumed_from_snapshot": resumed_from_snapshot,
                    }

                # Model may have written tool call as plain text or with malformed JSON. Inject corrector.
                has_block = content_has_tool_call_markup(raw_content)
                consecutive_no_tool_calls += 1
                self.logger.warning(
                    "Agent responded without a usable tool call (%s, content=%d chars). Injecting correction (attempt %d).",
                    "textual tool markup" if has_block else "no tool call found",
                    len(content),
                    consecutive_no_tool_calls,
                )
                if consecutive_no_tool_calls >= 2:
                    correction_msg = (
                        "🚨 CRITICAL: Still no usable tool call after two attempts. "
                        "Use a native tool call now. If native tool calling is unavailable, reply with ONLY this exact strict JSON block:\n"
                        "```json\n"
                        '[{"name": "update_working_memory", "args": {"new_facts": ["retrying"]}}]\n'
                        "```"
                    )
                elif has_block:
                    correction_msg = (
                        "🚨 Your last response wrote tool calls as text, but native tool calls are required when available. "
                        "If you must fall back to text, the JSON must be strict and parseable. Format:\n"
                        "```json\n"
                        '[{"name": "tool_name", "args": {"key": "value"}}]\n'
                        "```"
                    )
                else:
                    correction_msg = (
                        "⚠️ Your last response did not produce a usable native tool call. "
                        "Call the tool through the API instead of writing it in prose. "
                        "If native tool calling is unavailable, fall back to a single strict JSON block:\n"
                        "```json\n"
                        '[{"name": "tool_name", "args": {"key": "value"}}]\n'
                        "```\n"
                        "If you are done, use the finish tool with your full report in final_analysis."
                    )
                self._record_tool_journal_step(
                    step,
                    [],
                    assistant_summary=content,
                    notes=["issued correction because no usable tool call was produced"],
                )
                history.append({"role": "user", "content": correction_msg})
                continue

            consecutive_no_tool_calls = 0

            tool_calls = [canonicalize_bist_market_tool_call(call) for call in tool_calls]

            memory_calls = [
                t
                for t in tool_calls
                if t.get("name") == "update_working_memory"
            ]
            memory_was_updated = bool(memory_calls)
            memory_update_payloads = [
                mem_call.get("args", {})
                for mem_call in memory_calls
                if isinstance(mem_call.get("args"), dict)
            ]
            for mem_call in memory_calls:
                mem_args = mem_call.get("args", {})
                if isinstance(mem_args, dict):
                    self._update_working_memory(mem_args, console)
                    total_memory_updates += 1
                    if mem_call.get("id"):
                        history.append(
                            build_tool_result_history_message(
                                tool_name="update_working_memory",
                                result={
                                    "status": "ok",
                                    "memory_updated": True,
                                    "counts": self.memory.summary_counts(),
                                },
                                tool_call=mem_call,
                            )
                        )
                elif mem_call.get("id"):
                    history.append(
                        build_tool_result_history_message(
                            tool_name="update_working_memory",
                            result={
                                "error": "Invalid arguments for update_working_memory",
                                "error_code": "invalid_arguments",
                            },
                            tool_call=mem_call,
                        )
                    )

            finish_call = next(
                (t for t in tool_calls if t.get("name") == "finish"), None
            )

            # If finish is mixed with executable tools in the same step,
            # execute tools first and require finish in the next step.
            non_finish_tool_calls = [
                t
                for t in tool_calls
                if t.get("name") not in ("finish", "update_working_memory")
            ]
            finish_was_deferred = bool(finish_call and non_finish_tool_calls)
            finish_deferred_notice: Optional[str] = None
            if finish_call and non_finish_tool_calls:
                self.logger.warning(
                    "finish() called together with %d additional tool(s); deferring finish until tools are executed.",
                    len(non_finish_tool_calls),
                )
                if finish_call.get("id"):
                    history.append(
                        build_tool_result_history_message(
                            tool_name="finish",
                            result={
                                "status": "deferred",
                                "error": "finish must be the only tool in its step",
                                "error_code": "finish_deferred",
                            },
                            tool_call=finish_call,
                        )
                    )
                finish_deferred_notice = (
                    "⚠️ You called finish() together with other tools in the same step. "
                    "finish() was ignored for now. First execute tool calls and process results, "
                    "then call finish() in a separate step with full final_analysis."
                )
                finish_call = None

            preview_plan_rows = build_tool_plan_preview(
                tool_calls,
                max_parallel_tools=self._max_parallel_tools,
                non_dedup_tools=_NON_DEDUP_TOOLS,
                executed_signatures=executed_tool_signatures,
            )

            self._emit_reflection(
                response.get("reasoning"),
                preview_plan_rows,
                console,
                reasoning_streamed=bool(response.get("reasoning_streamed", False)),
            )

            if finish_call:
                args = finish_call.get("args", {}) if isinstance(finish_call.get("args"), dict) else {}
                final_analysis = str(args.get("final_analysis", "") or "").strip()
                research_summary = str(args.get("research_summary", "") or "").strip()

                # Fallback: model wrote the report in the message body but called finish({}).
                if not final_analysis and content and len(content) > 200:
                    self.logger.warning(
                        "finish() called with empty final_analysis; using message content as fallback "
                        "(%d chars).",
                        len(content),
                    )
                    final_analysis = content.strip()

                if not final_analysis:
                    self._record_tool_journal_step(
                        step,
                        tool_calls,
                        assistant_summary=content,
                        memory_updates=memory_update_payloads,
                        notes=["finish() was rejected because final_analysis was empty"],
                    )
                    if finish_call.get("id"):
                        history.append(
                            build_tool_result_history_message(
                                tool_name="finish",
                                result={
                                    "status": "blocked",
                                    "error": "finish requires a non-empty final_analysis",
                                    "error_code": "missing_final_analysis",
                                },
                                tool_call=finish_call,
                            )
                        )
                    history.append(
                        {
                            "role": "user",
                            "content": (
                                "⚠️ finish() called without a usable `final_analysis`. "
                                "Provide the complete final report in `final_analysis` or in the assistant message body."
                            ),
                        }
                    )
                    continue

                # Hard guard against premature finish.
                min_steps_required = 2
                min_tools_required = 2
                min_memory_updates_required = 1
                if (
                    step < min_steps_required
                    or total_non_memory_tools_executed < min_tools_required
                    or total_memory_updates < min_memory_updates_required
                ):
                    blockers: List[str] = []
                    if step < min_steps_required:
                        blockers.append(
                            f"at least {min_steps_required} steps required (current: {step})"
                        )
                    if total_non_memory_tools_executed < min_tools_required:
                        blockers.append(
                            "not enough research tools executed "
                            f"(required: {min_tools_required}, current: {total_non_memory_tools_executed})"
                        )
                    if total_memory_updates < min_memory_updates_required:
                        blockers.append(
                            "working memory not updated enough "
                            f"(required: {min_memory_updates_required}, current: {total_memory_updates})"
                        )

                    reason = "; ".join(blockers)
                    self.logger.warning(
                        "Premature finish blocked in pre_research: %s", reason
                    )
                    self._record_tool_journal_step(
                        step,
                        tool_calls,
                        assistant_summary=content,
                        memory_updates=memory_update_payloads,
                        notes=[f"premature finish blocked: {reason}"],
                    )
                    if finish_call.get("id"):
                        history.append(
                            build_tool_result_history_message(
                                tool_name="finish",
                                result={
                                    "status": "blocked",
                                    "error": reason,
                                    "error_code": "premature_finish",
                                },
                                tool_call=finish_call,
                            )
                        )
                    history.append(
                        {
                            "role": "user",
                            "content": (
                                "⛔ Premature finish blocked: "
                                f"{reason}. Continue research with tools, update working memory, "
                                "then call finish() with complete final_analysis."
                            ),
                        }
                    )
                    continue

                finish_notes = ["finish() accepted"]
                if research_summary:
                    finish_notes.append("research_summary provided")
                self._record_tool_journal_step(
                    step,
                    tool_calls,
                    assistant_summary=content,
                    memory_updates=memory_update_payloads,
                    notes=finish_notes,
                )
                if finish_call.get("id"):
                    history.append(
                        build_tool_result_history_message(
                            tool_name="finish",
                            result={
                                "status": "ok",
                                "accepted": True,
                            },
                            tool_call=finish_call,
                        )
                    )
                await self.memory.flush()
                session_path = self._save_session_snapshot(
                    active_session_id,
                    history=history,
                    step=step,
                    exchange=exchange,
                    criteria=criteria,
                    depth_mode=depth_mode,
                )
                return {
                    "analysis": final_analysis,
                    "steps": step,
                    "model_used": response.get("model", "unknown"),
                    "confidence": 1.0,
                    "timestamp": datetime.now().timestamp(),
                    "working_memory": self.memory.to_dict(),
                    "research_summary": research_summary,
                    "step_limit_reached": False,
                    "session_id": active_session_id,
                    "session_path": session_path,
                    "resumed_from_snapshot": resumed_from_snapshot,
                }

            tasks = []
            param_map: List[Dict[str, Any]] = []
            skipped_due_to_limit: List[Dict[str, Any]] = []
            executable_count = 0
            for tool_call in tool_calls:
                fn_name = tool_call.get("name")
                fn_args = tool_call.get("args", {})
                tool_meta = {
                    "name": fn_name or "unknown_tool",
                    "tool_call": tool_call,
                }

                if not fn_name:
                    tasks.append(
                        self._mock_tool_error("Tool name missing in tool call.")
                    )
                    param_map.append(tool_meta)
                    continue

                if fn_name == "update_working_memory":
                    continue

                if not isinstance(fn_args, dict):
                    tasks.append(
                        self._mock_tool_error(f"Invalid JSON arguments for {fn_name}")
                    )
                    param_map.append(tool_meta)
                    continue

                fn_args = normalize_tool_args(fn_name, fn_args)
                tool_call["args"] = fn_args

                validation_error = validate_tool_args(fn_name, fn_args)
                if validation_error:
                    tasks.append(self._mock_tool_error(validation_error))
                    param_map.append(tool_meta)
                    continue

                if fn_name == "yfinance_search":
                    bist_block_reason = should_block_bist_yfinance_search(
                        exchange=self._current_exchange,
                        query=fn_args.get("query"),
                        type_filter=fn_args.get("type_filter"),
                    )
                    if bist_block_reason:
                        tasks.append(self._mock_tool_error(bist_block_reason))
                        param_map.append(tool_meta)
                        continue

                # Pre-flight: search_memory requires 'query' or 'hit_ids'.
                if fn_name == "search_memory":
                    has_query = bool(str(fn_args.get("query") or "").strip())
                    has_hit_ids = bool(fn_args.get("hit_ids"))
                    if not has_query and not has_hit_ids:
                        tasks.append(
                            self._mock_tool_error(
                                "search_memory: 'query' (string) or 'hit_ids' (list) is required. "
                                "Provide a natural language query to search memory."
                            )
                        )
                        param_map.append(tool_meta)
                        continue

                if (
                    self._max_parallel_tools is not None
                    and executable_count >= self._max_parallel_tools
                ):
                    skipped_due_to_limit.append(tool_meta)
                    continue

                # Duplicate detection: skip identical tool calls across steps.
                if fn_name not in _NON_DEDUP_TOOLS:
                    sig = make_tool_signature(fn_name, fn_args)
                    if sig and sig in executed_tool_signatures:
                        tasks.append(
                            self._mock_tool_error(
                                f"Duplicate tool call skipped: '{fn_name}' was already called with identical "
                                f"parameters in a previous step. Use different parameters or a different tool."
                            )
                        )
                        param_map.append(tool_meta)
                        executable_count += 1
                        continue
                    if sig:
                        executed_tool_signatures.add(sig)

                tasks.append(self._execute_tool_safe(fn_name, fn_args, console=console))
                param_map.append(tool_meta)
                executable_count += 1

            results = await asyncio.gather(*tasks)
            executed_tools = [item["name"] for item in param_map]
            journal_tool_results = [
                {"name": param_map[i]["name"], "result": result}
                for i, result in enumerate(results)
            ]
            journal_notes: List[str] = []
            if finish_was_deferred:
                journal_notes.append(
                    "finish() was deferred because it was mixed with executable tools"
                )

            if skipped_due_to_limit:
                skipped_names = [item["name"] for item in skipped_due_to_limit]
                skipped_str = ", ".join(skipped_names)
                self.logger.info(
                    "Tool concurrency limit applied (%d). Deferred tools: %s",
                    self._max_parallel_tools,
                    skipped_str,
                )
                journal_notes.append(
                    f"deferred {len(skipped_names)} tool(s) because of the parallel tool limit"
                )
                for skipped_tool in skipped_due_to_limit:
                    if skipped_tool["tool_call"].get("id"):
                        history.append(
                            build_tool_result_history_message(
                                tool_name=skipped_tool["name"],
                                result={
                                    "status": "deferred",
                                    "error": "parallel tool limit reached for this step",
                                    "error_code": "parallel_limit",
                                },
                                tool_call=skipped_tool["tool_call"],
                            )
                        )

            for i, result in enumerate(results):
                tool_name = param_map[i]["name"]
                tool_call = param_map[i]["tool_call"]
                if tool_name not in ("finish", "update_working_memory"):
                    total_non_memory_tools_executed += 1

                history.append(
                    build_tool_result_history_message(
                        tool_name=tool_name,
                        result=result,
                        tool_call=tool_call,
                    )
                )

                # Record tool outputs to ephemeral lists by type.
                if tool_name in ("fetch_url_content", "summarize_url_content"):
                    self._fetch_indices.append(len(history) - 1)
                elif tool_name in _EPHEMERAL_SEARCH_TOOLS:
                    self._search_indices.append(len(history) - 1)

            if finish_deferred_notice:
                history.append(
                    {
                        "role": "user",
                        "content": finish_deferred_notice,
                    }
                )

            if skipped_due_to_limit:
                skipped_names = [item["name"] for item in skipped_due_to_limit]
                skipped_str = ", ".join(skipped_names)
                history.append(
                    {
                        "role": "user",
                        "content": (
                            f"[System notice] Max parallel tool limit is {self._max_parallel_tools}. "
                            f"Deferred tools: {skipped_str}. Continue with remaining tools in the next step."
                        ),
                    }
                )

            self._record_tool_journal_step(
                step,
                tool_calls,
                [item["name"] for item in skipped_due_to_limit],
                assistant_summary=content,
                memory_updates=memory_update_payloads,
                tool_results=journal_tool_results,
                notes=journal_notes,
            )

            # Check memory consolidation after tool outputs are processed.
            await self._consolidate_working_memory_if_needed(step, console)

            if not self._screening_pivot_notified:
                pivot_notice = get_pre_research_pivot_notice(
                    self.memory.to_dict(), executed_tools
                )
                if pivot_notice:
                    history.append({"role": "user", "content": pivot_notice})
                    self._screening_pivot_notified = True

            # Inform AI of remaining step budget.
            # Previous progress message replaced with stub (token savings).
            remaining = self.max_steps - step
            if remaining > 0:
                depth_score = self._current_depth_score()
                if remaining <= 3:
                    step_note = (
                        f" \u26a0\ufe0f CRITICAL: Only {remaining} step(s) left. "
                        "You MUST call finish() NOW with all gathered data. "
                        "Do NOT use any more search tools."
                    )
                else:
                    step_note = " Continue research or call finish() if all quality criteria are satisfied."

                tool_journal_context = self._build_tool_journal_context()
                tool_journal_note = (
                    f"\n\n{tool_journal_context}" if tool_journal_context else ""
                )

                # Model got tool result but didn't update memory — block reminder.
                memory_reminder = ""
                if (
                    executed_tools
                    and not memory_was_updated
                    and remaining > 3
                    and any(tool_name in VERIFIED_MEMORY_TOOLS for tool_name in executed_tools)
                ):
                    tools_str = ", ".join(executed_tools)
                    memory_reminder = (
                        f"\n\n\u2139\ufe0f You received source-backed results from [{tools_str}] without a memory update. "
                        "If those results contain verified figures or durable findings, save them now with "
                        "`update_working_memory`. Store verified facts only; move tentative interpretations to "
                        "questions or milestones instead of `new_facts`."
                    )

                # Convert old progress message to stub.
                if (
                    self._progress_msg_index is not None
                    and self._progress_msg_index < len(history)
                ):
                    history[self._progress_msg_index]["content"] = "[Progress archived]"

                history.append(
                    {
                        "role": "user",
                        "content": (
                            f"[Progress: Step {step}/{self.max_steps} | {remaining} steps remaining | "
                            f"Research depth score: {depth_score}]"
                            f"{step_note}"
                            f"{memory_reminder}"
                            f"{tool_journal_note}"
                        ),
                    }
                )
                self._progress_msg_index = len(history) - 1

            session_path = self._save_session_snapshot(
                active_session_id,
                history=history,
                step=step,
                exchange=exchange,
                criteria=criteria,
                depth_mode=depth_mode,
            )

        # Max steps reached - Construct partial report
        partial_report = self._build_partial_report()

        session_path = self._save_session_snapshot(
            active_session_id,
            history=history,
            step=step,
            exchange=exchange,
            criteria=criteria,
            depth_mode=depth_mode,
        )

        return {
            "analysis": partial_report,
            "steps": step,
            "model_used": "partial_result",
            "working_memory": self.memory.to_dict(),
            "step_limit_reached": True,
            "warning": "Agent max step budget reached; returning partial results.",
            "session_id": active_session_id,
            "session_path": session_path,
            "resumed_from_snapshot": resumed_from_snapshot,
            # Removing error field so frontend shows report instead of just error.
            # "error": "Max steps reached"
        }

    async def _mock_tool_error(self, error_msg: str) -> str:
        return json.dumps({"error": error_msg})

    async def _fetch_url_with_digest(
        self,
        url: str,
        raw_result: Any,
        fallback_result_str: str,
        console: Optional[Console] = None,
    ) -> Dict[str, Any]:
        """
        Filter fetch_url_content (raw content) output through mini-model (summarizer).
        Extract only parts of the page that answer open questions in working memory.
        Significantly preserves the main model's context window.
        """
        questions = self.memory.to_dict().get("unanswered_questions", [])
        questions_text = (
            "\n".join(f"- {q}" for q in questions)
            if questions
            else "- Extract all key financial metrics, events, figures, and named entities."
        )
        digest_system = augment_system_prompt(
            (
                "You are a precision content extraction assistant. "
                "Given fetched web page content and a list of research questions, "
                "extract ONLY the fragments that directly answer those questions. "
                "Preserve all numbers, names, dates, percentages, and financial figures VERBATIM. "
                "Output a concise structured summary \u2014 do NOT reproduce the full page."
            ),
            config=self._config,
        )
        prepared_payload = self._prepare_fetch_digest_payload(raw_result)
        digest_user = (
            f"Research questions:\n{questions_text}\n\n"
            f"Fetched page content:\n{prepared_payload}\n\n"
            "Extracted relevant information:"
        )
        fallback_payload = (
            dict(fallback_result_str)
            if isinstance(fallback_result_str, dict)
            else dict(raw_result if isinstance(raw_result, dict) else {})
        )
        try:
            digest_response = await self._request_executor.send_async(
                [
                    {"role": "system", "content": digest_system},
                    {"role": "user", "content": digest_user},
                ],
                request_type="summarizer",
                timeout_override=self._summarizer_timeout,
            )
            digest_text = (digest_response.get("content") or "").strip()
            if not digest_text:
                fallback_payload["digest_strategy"] = str(
                    fallback_payload.get("digest_strategy") or "raw_fallback"
                )
                return fallback_payload
            self.logger.info(
                "fetch_url_content digest: %d \u2192 %d karakter",
                len(prepared_payload),
                len(digest_text),
            )
            if console:
                console.print(
                    f"[dim]\U0001f52c URL summary: {len(prepared_payload):,} \u2192 {len(digest_text):,} characters[/dim]"
                )
            return {
                "url": url,
                "digest": digest_text,
                "is_summarized": True,
                "is_retrieval_pruned": False,
                "digest_strategy": "llm_digest",
                "source_length": len(prepared_payload),
            }
        except Exception as exc:
            self.logger.warning(
                "fetch_url_content digest failed, using raw content: %s", exc
            )
            fallback_payload["digest_strategy"] = str(
                fallback_payload.get("digest_strategy") or "raw_fallback"
            )
            return fallback_payload

    async def _execute_tool_safe(
        self, tool_name: str, args: Dict[str, Any], console: Optional[Console] = None
    ) -> str:
        try:
            if tool_name == "update_working_memory":
                self._update_working_memory(args, console)
                return json.dumps(
                    {
                        "status": "Working memory updated",
                        "current_depth_score": self._current_depth_score(),
                    }
                )

            if tool_name == "fetch_url_content":
                normalized_url = str(args.get("url") or "").strip()
                if self._is_duplicate_fetch(normalized_url):
                    return json.dumps(
                        self._build_skip_fetch_payload(normalized_url),
                        ensure_ascii=False,
                        default=str,
                    )

            result = await execute_tool(tool_name, args=args, console=console)
            if tool_name == "fetch_url_content" and not (
                isinstance(result, dict) and result.get("error")
            ):
                self._mark_url_fetched(str(args.get("url") or "").strip())

            standardized_result = to_standard_tool_result(tool_name, result)
            if tool_name == "fetch_url_content" and not (
                isinstance(result, dict) and result.get("error")
            ):
                payload = await self._post_process_fetch_result(
                    args=args,
                    raw_result=standardized_result,
                    console=console,
                )
                return json.dumps(payload, ensure_ascii=False, default=str)

            result_str = json.dumps(standardized_result, ensure_ascii=False, default=str)

            return result_str
        except Exception as exc:
            return json.dumps({"error": str(exc)})

