import json
import asyncio
import re
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from rich.console import Console

from config import get_config
from domain.utils import get_ai_company_context
from services.factories import get_rag_service, get_shared_memory_pool
from services.tools import execute_tool, get_tool_quality_metrics, to_standard_tool_result
from services.ai.memory_formatter import (
    format_working_memory_evidence_pack,
    format_working_memory_for_llm,
)
from services.ai.providers.token_utils import estimate_tokens_for_messages
from services.ai.providers.history_utils import select_immediate_context
from services.ai.providers.memory_rag_utils import (
    derive_spill_confidence,
    derive_spill_data_gap_count,
    format_fact_for_rag,
    generate_hypothetical_rag_document,
)
from services.ai.providers.reflection_prompt_utils import (
    build_reflection_prompt,
    should_inject_reflection,
)
from services.ai.providers.context_preservation_utils import (
    archive_history_segment,
    build_ephemeral_evidence_record,
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
from services.ai.providers.tool_journal_utils import (
    format_tool_journal_for_prompt,
    normalize_tool_journal,
    normalize_tool_journal_step,
    summarize_memory_update_args,
    summarize_tool_result,
)
from services.ai.providers.agent_guardrails import (
    build_tool_plan_preview,
    canonicalize_bist_market_tool_call,
    has_actionable_memory_payload,
    looks_like_final_report_payload,
    make_tool_signature,
    normalize_tool_args,
    sanitize_memory_args,
    validate_tool_args,
)
from services.ai.working_memory import WorkingMemory

# Büyük ham çıktı üreten ve geçmişten erken temizlenmesi gereken arama araçları.
# fetch_url_content / summarize_url_content ayrı takip edilir (daha uzun tutulur — son 2 adet).
_EPHEMERAL_URL_TOOLS: frozenset = frozenset(
    {"fetch_url_content", "summarize_url_content"}
)
_EPHEMERAL_SEARCH_TOOLS: frozenset = frozenset(
    {"search_web", "search_news", "search_google_news"}
)
_COMPRESSED_TOOL_RESULT_STUB = json.dumps(
    {
        "status": "ephemeral_compressed",
        "info": "Raw content was processed and removed to save tokens. Key findings are in working memory.",
    }
)

from domain.utils import safe_int_strict as _safe_int


class AutonomousNewsAgent:
    """
    Autonomous agent that manages the news analysis loop.
    Uses Working Memory to track research progress and gaps.
    """

    def __init__(self, logger, prompt_resolver, request_executor):
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
        agent_steps_cfg = config.get("ai", {}).get("agent_steps", {})
        self._default_max_steps = _safe_int(agent_steps_cfg.get("news", 25), 25)
        self.max_steps = self._default_max_steps
        self._continuation_increment = _safe_int(
            agent_steps_cfg.get("continuation_increment", 10), 10
        )

        tool_limit_cfg = config.get("ai", {}).get("agent_tool_limits", {})
        raw_limit = _safe_int(tool_limit_cfg.get("max_parallel_tools", 0), 0)
        self._max_parallel_tools: Optional[int] = raw_limit if raw_limit > 0 else None

        # Cache config slice used in methods to avoid repeated deepcopy.
        self._output_lang: str = config.get("ai", {}).get("output_language", "English")
        self._config = config

        working_memory_cfg = config.get("ai", {}).get("working_memory", {})
        self._current_symbol: str = ""
        shared_pool = (
            get_shared_memory_pool(config)
            if bool(working_memory_cfg.get("shared_pool_enabled", True))
            else None
        )
        self.memory = WorkingMemory(
            consolidation_callback=self._consolidate_facts_with_llm,
            spill_callback=self._spill_facts_to_rag,
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
            shared_memory_agent_name="autonomous_news_agent",
        )
        # fetch_url_content + summarize_url_content çıktıları: son 2 adım geçmişte tutulur.
        self._fetch_indices: List[int] = []
        # search_web / search_news / search_google_news çıktıları: son 1 adım tutulur
        # (arama snippet'leri WM'ye kaydedildikten sonra gereksizdir).
        self._search_indices: List[int] = []
        self._memory_search_confirmed: bool = False
        self._history_archives: List[Dict[str, Any]] = []
        self._tool_journal: List[Dict[str, Any]] = []

    def _reset_run_state(self) -> None:
        self._fetch_indices.clear()
        self._search_indices.clear()
        self._memory_search_confirmed = False
        self._history_archives = []
        self._tool_journal = []

    def _restore_tool_journal(self, value: Any) -> None:
        self._tool_journal = normalize_tool_journal(value)

    def _record_tool_journal_step(
        self,
        step: int,
        tool_calls: List[Dict[str, Any]],
        deferred_tools: Optional[List[str]] = None,
        assistant_summary: Optional[str] = None,
        memory_updates: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        notes: Optional[List[str]] = None,
    ) -> None:
        journal = getattr(self, "_tool_journal", None)
        if not isinstance(journal, list):
            journal = []

        normalized_memory_updates: List[Dict[str, Any]] = []
        for item in memory_updates or []:
            if not isinstance(item, dict):
                continue
            if item.get("summary"):
                normalized_memory_updates.append(
                    {"summary": str(item.get("summary")).strip()}
                )
                continue
            normalized_memory_updates.append(summarize_memory_update_args(item))

        normalized_tool_results: List[Dict[str, Any]] = []
        for item in tool_results or []:
            if not isinstance(item, dict):
                continue
            if item.get("summary") is not None and item.get("status") is not None:
                normalized_tool_results.append(
                    {
                        "name": str(item.get("name") or "tool").strip() or "tool",
                        "status": str(item.get("status") or "ok").strip() or "ok",
                        "summary": str(item.get("summary") or "").strip(),
                    }
                )
                continue
            normalized_tool_results.append(
                summarize_tool_result(
                    str(item.get("name") or "tool").strip() or "tool",
                    item.get("result"),
                )
            )

        journal.append(
            normalize_tool_journal_step(
                step,
                tool_calls,
                deferred_tools=deferred_tools,
                assistant_summary=assistant_summary,
                memory_updates=normalized_memory_updates,
                tool_results=normalized_tool_results,
                notes=notes,
            )
        )
        self._tool_journal = journal

    def _build_tool_journal_context(self) -> str:
        journal = getattr(self, "_tool_journal", [])
        if not isinstance(journal, list) or not journal:
            return ""
        return format_tool_journal_for_prompt(journal)

    def _restore_ephemeral_tracking(self, history: List[Dict[str, Any]]) -> None:
        self._fetch_indices.clear()
        self._search_indices.clear()
        for idx, message in enumerate(history):
            if str(message.get("role") or "").strip().lower() not in {"user", "tool"}:
                continue
            tool_name = self._extract_tool_name_from_result_message(
                message,
                history=history,
                message_index=idx,
            )
            if tool_name in _EPHEMERAL_URL_TOOLS:
                self._fetch_indices.append(idx)
            elif tool_name in _EPHEMERAL_SEARCH_TOOLS:
                self._search_indices.append(idx)

    def _set_memory_depth_score(self, score: int) -> None:
        self.memory.set_research_depth_score(score)

    @staticmethod
    def _format_fact_text(fact: Any) -> str:
        if isinstance(fact, dict):
            return str(fact.get("text", "")).strip()
        return str(fact).strip()

    def _build_partial_report(self) -> str:
        partial_report = (
            "# ⚠️ Step Limit Reached\n\n"
            "The agent reached the configured step limit and could not complete the analysis. "
            "The report below is based on the data collected so far.\n\n"
        )

        memory_state = self.memory.to_dict()
        facts = [
            self._format_fact_text(fact)
            for fact in memory_state.get("facts_learned", [])
        ]
        facts = [fact for fact in facts if fact]
        if facts:
            partial_report += "## 💡 Findings\n"
            partial_report += "\n".join(f"- {fact}" for fact in facts)
            partial_report += "\n\n"

        sources = [
            str(source).strip()
            for source in memory_state.get("sources_consulted", [])
            if str(source).strip()
        ]
        if sources:
            partial_report += "## 📚 Sources Reviewed\n"
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

    def _build_session_state(
        self,
        *,
        history: List[Dict[str, Any]],
        step: int,
        max_steps: int,
        company_name: str,
        company_data: Dict[str, Any],
        depth_mode: str,
    ) -> Dict[str, Any]:
        return {
            "history": history,
            "history_archives": list(self._history_archives),
            "step": int(step),
            "max_steps": int(max_steps),
            "memory": self.memory.to_dict(),
            "memory_search_confirmed": self._memory_search_confirmed,
            "company_name": company_name,
            "company_data": company_data,
            "depth_mode": depth_mode,
            "tool_journal": list(self._tool_journal),
        }

    def _save_session_snapshot(
        self,
        session_id: str,
        *,
        history: List[Dict[str, Any]],
        step: int,
        max_steps: int,
        company_name: str,
        company_data: Dict[str, Any],
        depth_mode: str,
    ) -> str:
        payload: Dict[str, Any] = {
            "company_name": company_name,
            "symbol": self._current_symbol,
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
            "state": self._build_session_state(
                history=history,
                step=step,
                max_steps=max_steps,
                company_name=company_name,
                company_data=company_data,
                depth_mode=depth_mode,
            ),
        }
        return save_agent_session("autonomous_news_agent", session_id, payload)

    @staticmethod
    def _serialize_facts_for_rag(facts: List[Any]) -> str:
        serialized = [format_fact_for_rag(fact) for fact in facts]
        return "\n".join(item for item in serialized if item)

    def _current_depth_score(self) -> int:
        return int(self.memory.summary_counts().get("score", 0))

    def _build_self_reflection_prompt(self, step: int) -> str:
        reflection_cfg = self._config.get("ai", {}).get("agent_reflection", {})
        prompt = build_reflection_prompt(
            self.memory.to_dict(),
            step=step,
            output_language=self._output_lang,
            max_facts=int(reflection_cfg.get("max_facts", 4) or 4),
            max_questions=int(reflection_cfg.get("max_questions", 3) or 3),
            max_contradictions=int(
                reflection_cfg.get("max_contradictions", 2) or 2
            ),
        )
        journal_context = self._build_tool_journal_context()
        return f"{prompt}\n\n{journal_context}" if journal_context else prompt

    def _maybe_inject_self_reflection(
        self, history: List[Dict[str, Any]], step: int
    ) -> None:
        reflection_cfg = self._config.get("ai", {}).get("agent_reflection", {})
        if not bool(reflection_cfg.get("enabled", True)):
            return

        interval_steps = int(reflection_cfg.get("interval_steps", 3) or 3)
        if not should_inject_reflection(
            self.memory.to_dict(),
            step=step,
            interval_steps=interval_steps,
        ):
            return

        prompt = self._build_self_reflection_prompt(step)
        if not prompt:
            return
        history.append({"role": "user", "content": prompt})

    @staticmethod
    def _prepare_fetch_digest_payload(raw_result: Any) -> str:
        payload = raw_result.get("data") if isinstance(raw_result, dict) else raw_result
        if isinstance(payload, dict):
            compact_payload = {
                key: payload.get(key)
                for key in (
                    "url",
                    "title",
                    "source",
                    "source_info",
                    "published_at",
                    "author",
                    "summary",
                    "content",
                )
                if payload.get(key)
            }
            payload = compact_payload or payload

        if isinstance(payload, str):
            return payload

        return json.dumps(payload, default=str, ensure_ascii=False)

    async def _consolidate_facts_with_llm(self, facts: List[str]) -> List[str]:
        """WorkingMemory callback'i için LLM tabanlı fakt konsolidasyonu."""
        if not facts:
            return []

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
            "Aim to reduce the count by at least 30%% while retaining 100%% of the information."
            ),
            config=self._config,
            output_language=output_lang,
        )
        user_content = (
            f"Consolidate these {len(facts)} facts into a smaller, denser list:\n\n"
            f"{json.dumps(facts, ensure_ascii=False)}"
        )

        try:
            response = await self._request_executor.send_async(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                request_type="summarizer",
                timeout_override=30,
            )
            raw = (response.get("content") or "").strip()
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                return []

            consolidated = json.loads(match.group())
            if not isinstance(consolidated, list):
                return []
            return [str(item).strip() for item in consolidated if str(item).strip()]
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            self.logger.warning(
                "Callback memory consolidation failed (JSON/Type error): %s", exc
            )
            return []
        except Exception as exc:
            self.logger.warning(
                "Callback memory consolidation failed (unexpected error): %s", exc
            )
            return []

    def _spill_facts_to_rag(self, facts: List[str]) -> None:
        if not facts:
            return
        try:
            rag = get_rag_service()
            rag.index_analysis(
                symbol=self._current_symbol or "UNKNOWN",
                news_output=self._serialize_facts_for_rag(list(facts)),
                final_output=None,
                confidence_score=derive_spill_confidence(facts, default=0.55),
                data_gaps_count=derive_spill_data_gap_count(facts),
            )
        except Exception as exc:
            self.logger.debug("Working memory spill-over skipped: %s", exc)

    async def _summarize_history(
        self, history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Compresses conversation history when it gets too long, using a cheaper model.
        Preserves System Prompt and immediate context (last few messages).
        """
        if len(history) < 5:
            return history

        self.logger.info("Compressing conversation history (token optimization)...")

        # 1. Slice Strategy
        system_msg = history[0]
        immediate_context = select_immediate_context(
            history,
            token_limit=self._history_token_limit,
            encoding=self._token_encoding,
            min_keep=4,
        )
        cut_index = len(history) - len(immediate_context)
        # The chunk to summarize
        to_summarize = history[1:cut_index]

        if not to_summarize:
            return history

        archive_history_segment(
            self._history_archives,
            segment=list(to_summarize),
            reason="history_summarized",
        )

        # 2. Prepare Summarization Request
        # We start a NEW temporary conversation just for summarization
        try:
            prompt_template = self._prompt_resolver("history_summarizer")

            # Inject output language
            output_lang = self._output_lang
            system_prompt = prompt_template.replace("{output_language}", output_lang)
        except Exception:
            system_prompt = augment_system_prompt(
                "Summarize the following conversation history, preserving key facts and numbers.",
                config=self._config,
            )

        # Convert the middle chunk to a single string for the summarizer
        context_str = json.dumps(to_summarize, default=str, ensure_ascii=False)

        tool_trace: List[str] = []
        for msg in to_summarize:
            if msg.get("role") != "assistant":
                continue
            native_calls = normalize_tool_calls(msg.get("tool_calls"))
            parsed_calls, _ = parse_tool_calls_from_content(msg.get("content", ""))
            for call in (native_calls or parsed_calls):
                fn_name = call.get("name", "unknown_tool")
                fn_args = call.get("args", {})
                tool_trace.append(
                    f"{fn_name}({json.dumps(fn_args, ensure_ascii=False)})"
                )
        tool_trace_str = "\n".join(f"- {line}" for line in tool_trace[:50])
        if not tool_trace_str:
            tool_trace_str = "- No tool call trace in compressed segment."

        memory_snapshot = self._format_memory_snapshot_for_summary()
        evidence_pack = format_working_memory_evidence_pack(
            self.memory.to_dict(),
            max_facts=None,
            max_contradictions=None,
            max_questions=None,
            max_sources=None,
        )

        summary_request_history = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Current Working Memory Snapshot (preserve ALL facts, numbers, dates and figures below):\n"
                    f"{memory_snapshot}\n\n"
                    f"Current Evidence Pack:\n{evidence_pack}\n\n"
                    f"Tool Call Trace (preserve research strategy and sequence):\n"
                    f"{tool_trace_str}\n\n"
                    f"Conversation History to Compress:\n{context_str}"
                ),
            },
        ]

        # 3. Execute with 'summarizer' model config
        try:
            response = await self._request_executor.send_async(
                summary_request_history, request_type="summarizer", timeout_override=60
            )
            summary_text = response.get("content", "Summary failed.")

            anchor_index = self._find_first_non_system_index(history)
            preserve_until = anchor_index + 1 if anchor_index is not None else 1

            # 4. Reconstruct History
            new_history = list(history[:preserve_until])
            new_history.append(
                {
                    "role": "user",
                    "content": (
                        f"PREVIOUS CONTEXT SUMMARY:\n{summary_text}\n\n"
                        f"EVIDENCE PACK:\n{evidence_pack or '- None'}\n\nEnd of summary."
                    ),
                }
            )
            new_history.extend(immediate_context)

            self.logger.info(
                "History compressed. Reduced from %d to %d messages.",
                len(history),
                len(new_history),
            )
            return new_history

        except Exception as e:
            self.logger.error("History summarization failed: %s", e)
            return history  # Fail safe: return original history

    @staticmethod
    def _find_first_non_system_index(history: list[Dict[str, Any]]) -> Optional[int]:
        for index, message in enumerate(history):
            role = str(message.get("role") or "").strip().lower()
            if role not in {"system", "developer"}:
                return index
        return None

    def _format_memory_snapshot_for_summary(self) -> str:
        return format_working_memory_for_llm(self.memory.to_dict(), style="snapshot")

    def _build_tool_reliability_notice(self) -> str:
        try:
            snapshot = get_tool_quality_metrics()
        except Exception as exc:
            self.logger.debug("Tool metrics snapshot unavailable: %s", exc)
            return ""

        by_tool = snapshot.get("by_tool", {}) if isinstance(snapshot, dict) else {}
        if not isinstance(by_tool, dict):
            return ""

        unstable: List[str] = []
        for tool_name, metrics in by_tool.items():
            if not isinstance(metrics, dict):
                continue
            calls = int(metrics.get("calls", 0) or 0)
            errors = int(metrics.get("errors", 0) or 0)
            if calls < 3:
                continue
            rate = (errors / calls) if calls else 0.0
            if rate >= 0.5:
                unstable.append(
                    f"{tool_name} ({errors}/{calls}, error_rate={rate:.0%})"
                )

        if not unstable:
            return ""

        unstable_text = "; ".join(unstable[:5])
        return (
            "\n\n⚠️ Tool Reliability Notice: These tools have elevated failure rates in this runtime: "
            f"{unstable_text}. Prefer alternatives when possible and cross-check critical facts."
        )

    def _emit_reflection(
        self,
        content: Optional[str],
        tool_plans: list[Dict[str, Any]],
        console: Optional[Console],
        reasoning_streamed: bool = False,
    ) -> None:
        if not self._reflection_enabled:
            return

        # 1. Native reasoning / short thought text
        if content and content.strip():
            thought = content.strip()
            # Show only short thought snippets, not long final answers.
            if len(thought) < 500 and not thought.startswith("#"):
                self.logger.info("AI Thought: %s", thought)
                should_render_reasoning_panel = not (
                    reasoning_streamed and self._suppress_reasoning_panel_after_stream
                )
                if console and should_render_reasoning_panel:
                    from rich.panel import Panel
                    from rich.markdown import Markdown

                    md = Markdown(thought)
                    panel = Panel(
                        md,
                        title="💭 [bold blue]Thinking Process[/bold blue]",
                        border_style="blue",
                        padding=(0, 1),
                    )
                    console.print(panel)

        # 2. Tool details
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

                # Tool call'ları tablo olarak göster
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
                        # Parametreleri daha temiz göster (uzun olanları kısalt)
                        args_parts = []
                        for k, v in args.items():
                            v_str = str(v)
                            if len(v_str) > 60:
                                v_str = v_str[:57] + "..."
                            args_parts.append(
                                f"[yellow]{k}[/yellow]=[dim]{v_str}[/dim]"
                            )
                        args_display = "\n".join(args_parts)
                    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
                        args_display = "{}"

                    status = str(plan.get("status") or "planned")
                    note = str(plan.get("note") or "").strip()
                    if note:
                        status = f"{status}\n{note}"

                    table.add_row(str(i), name, status, args_display)

                console.print(table)
                console.print()  # Boş satır

    def _extract_symbol(self, company_data: Dict[str, Any]) -> str:
        symbol = str(company_data.get("symbol") or "").strip().upper()
        if symbol:
            return symbol
        profile = (
            company_data.get("company_profile", {})
            if isinstance(company_data, dict)
            else {}
        )
        if isinstance(profile, dict):
            return str(profile.get("symbol") or "").strip().upper()
        return ""

    async def _bootstrap_memory_from_rag(
        self,
        company_name: str,
        company_data: Dict[str, Any],
        console: Optional[Console],
    ) -> None:
        symbol = self._extract_symbol(company_data)
        query = f"{company_name} {symbol} recent analysis risks opportunities".strip()

        try:
            rag = get_rag_service()
        except Exception as exc:
            self.logger.debug("RAG warm-start skipped: %s", exc)
            return

        rag_cfg = self._config.get("ai", {}).get("rag", {})
        query_expansion_cfg = rag_cfg.get("query_expansion", {})
        query_hypothesis = ""
        if bool(query_expansion_cfg.get("enabled", True)) and bool(
            query_expansion_cfg.get("warm_start_enabled", True)
        ):
            try:
                query_hypothesis = await generate_hypothetical_rag_document(
                    query,
                    request_executor=self._request_executor,
                    language=self._output_lang,
                    timeout_seconds=float(
                        query_expansion_cfg.get("timeout_seconds", 15) or 15
                    ),
                    max_chars=_safe_int(
                        query_expansion_cfg.get("hypothesis_max_chars", 900),
                        900,
                    ),
                )
            except Exception as exc:
                self.logger.debug("RAG HyDE warm-start skipped: %s", exc)

        working_memory_cfg = self._config.get("ai", {}).get("working_memory", {})
        warm_facts = self.memory.refresh_context(
            query,
            rag_service=rag,
            symbol_filter=symbol or None,
            top_k=_safe_int(working_memory_cfg.get("refresh_context_top_k", 5), 5),
            recent_days=_safe_int(
                working_memory_cfg.get("refresh_context_recent_days", 180), 180
            ),
            context_window=_safe_int(
                working_memory_cfg.get("refresh_context_window", 1), 1
            ),
            importance=7,
            tags=["rag_retrieved", "prior_analysis"],
            preview_chars=800,
            milestone="RAG warm-start completed at analysis start",
            query_hypothesis=query_hypothesis or None,
        )

        if not warm_facts:
            return

        score_before = int(self.memory.summary_counts().get("score", 0))
        self._set_memory_depth_score(score_before)
        if console:
            console.print(
                f"[dim]🧠 RAG warm-start: loaded {len(warm_facts)} historical findings into working memory.[/dim]"
            )

    @staticmethod
    def _find_previous_assistant(
        history: List[Dict[str, Any]], from_idx: int
    ) -> Optional[Dict[str, Any]]:
        for idx in range(from_idx - 1, -1, -1):
            msg = history[idx]
            if msg.get("role") == "assistant":
                return msg
            if msg.get("role") == "user":
                continue
        return None

    @staticmethod
    def _assistant_has_memory_update(assistant_msg: Optional[Dict[str, Any]]) -> bool:
        if not assistant_msg:
            return False
        native_calls = normalize_tool_calls(assistant_msg.get("tool_calls"))
        parsed_calls, _ = parse_tool_calls_from_content(assistant_msg.get("content", ""))
        for call in (native_calls or parsed_calls):
            if call.get("name") == "update_working_memory":
                return True
        return False

    @staticmethod
    def _extract_tool_name_from_result_message(
        message: Dict[str, Any],
        *,
        history: Optional[List[Dict[str, Any]]] = None,
        message_index: Optional[int] = None,
    ) -> str:
        role = str(message.get("role") or "").strip().lower()
        if role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "").strip()
            if tool_call_id and history is not None and message_index is not None:
                for idx in range(message_index - 1, -1, -1):
                    previous = history[idx]
                    if str(previous.get("role") or "").strip().lower() != "assistant":
                        continue
                    for tool_call in normalize_tool_calls(previous.get("tool_calls")):
                        if str(tool_call.get("id") or "").strip() == tool_call_id:
                            return str(tool_call.get("name") or "").strip() or "tool"
                    break
            return "tool"

        content = message.get("content", "")
        if not isinstance(content, str):
            return "tool"
        match = re.search(r'<tool_result\s+name="([^"]+)"', content)
        if not match:
            return "tool"
        return match.group(1)

    def _build_fallback_fact(self, tool_name: str, tool_content: Any) -> Optional[str]:
        preview = ""
        if isinstance(tool_content, str):
            try:
                parsed = json.loads(tool_content)
            except Exception:
                parsed = None

            if isinstance(parsed, dict):
                if parsed.get("error"):
                    return None
                preview = json.dumps(
                    parsed,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )[:1200]
            else:
                preview = tool_content[:1200]
        else:
            preview = str(tool_content)[:1200]

        preview = preview.strip()
        if not preview:
            return None

        return f"[Fallback:{tool_name}] Ephemeral output automatically backed up before stub: {preview}"

    def _capture_ephemeral_result_before_stub(
        self,
        *,
        tool_name: str,
        tool_content: Any,
        assistant_has_memory_update: bool,
    ) -> None:
        evidence_record = build_ephemeral_evidence_record(tool_name, tool_content)
        if evidence_record:
            self.memory.add_evidence_records([evidence_record])

        if assistant_has_memory_update:
            return

        fallback_fact = self._build_fallback_fact(
            tool_name,
            str((evidence_record or {}).get("preview") or tool_content),
        )
        if fallback_fact:
            self.memory.update(
                new_facts=[fallback_fact],
                fact_importance=6,
                fact_tags=["fallback", "ephemeral"],
                research_milestones=[
                    "Ephemeral tool output automatically backed up before stub"
                ],
                source_summary="fallback:auto_backup",
            )

    def _progress_context(self) -> str:
        memory_state = self.memory.to_dict()

        raw_facts = memory_state.get("facts_learned", [])
        facts_text: List[str] = []
        for item in raw_facts[-2:]:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
            else:
                text = str(item).strip()
            if text:
                facts_text.append(text)

        open_questions = [
            str(q).strip()
            for q in memory_state.get("unanswered_questions", [])
            if str(q).strip()
        ][:2]
        contradictions = [
            str(c).strip()
            for c in memory_state.get("contradictions_found", [])
            if str(c).strip()
        ][:1]
        rejected = [
            str(r).strip()
            for r in memory_state.get("rejected_hypotheses", [])
            if str(r).strip()
        ][:1]
        milestones = [
            str(m).strip()
            for m in memory_state.get("research_milestones", [])
            if str(m).strip()
        ][:1]

        chunks = []
        if facts_text:
            chunks.append("Recent findings: " + " | ".join(facts_text))
        if open_questions:
            chunks.append("Open questions: " + " | ".join(open_questions))
        if contradictions:
            chunks.append("Contradictions: " + " | ".join(contradictions))
        if rejected:
            chunks.append("Rejected hypothesis: " + " | ".join(rejected))
        if milestones:
            chunks.append("Latest milestone: " + " | ".join(milestones))

        return " ; ".join(chunks)

    async def run(
        self,
        company_name: str,
        company_data: Dict[str, Any],
        initial_news: Optional[List[Dict[str, Any]]] = None,
        console: Optional[Console] = None,
        depth_mode: str = "standard",
        session_id: Optional[str] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        on_max_steps_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the agentic loop with Working Memory tracking.

        Args:
            depth_mode: "standard" or "deep" - deep mode enforces more thorough research
            resume_state: Optional state to resume from previous interrupted run
            on_max_steps_callback: Optional callback when max steps reached, returns True to continue
        """
        self._current_symbol = (
            self._extract_symbol(company_data)
            or company_name.strip().upper()
            or "UNKNOWN"
        )
        self.memory.configure_shared_memory(
            scope=f"symbol:{self._current_symbol}",
            agent_name="autonomous_news_agent",
        )
        self.memory.refresh_from_shared_pool(
            f"{company_name} {self._current_symbol} recent analysis risks opportunities",
            top_k=_safe_int(
                self._config.get("ai", {})
                .get("working_memory", {})
                .get("shared_pool_search_top_k", 4),
                4,
            ),
            tags=["shared_pool", "prior_agent_memory"],
            milestone="Shared memory warm-start completed at analysis start",
        )
        max_steps = self._default_max_steps
        active_session_id = str(session_id or "").strip() or generate_agent_session_id(
            "autonomous-news", self._current_symbol or company_name
        )
        session_path = ""
        resumed_from_snapshot = False

        if not resume_state and session_id:
            snapshot = load_agent_session("autonomous_news_agent", active_session_id)
            raw_state = snapshot.get("state") if isinstance(snapshot, dict) else None
            if isinstance(raw_state, dict):
                resume_state = raw_state
                resumed_from_snapshot = True

        # Eğer önceki durumdan devam ediliyorsa yükle
        if resume_state:
            self.logger.info("Resuming analysis from saved state...")
            self._reset_run_state()
            company_name = str(resume_state.get("company_name") or company_name)
            raw_company_data = resume_state.get("company_data")
            if isinstance(raw_company_data, dict):
                company_data = raw_company_data
            raw_history = resume_state.get("history", [])
            history = raw_history if isinstance(raw_history, list) else []
            step = _safe_int(resume_state.get("step", 0), 0)
            max_steps = _safe_int(resume_state.get("max_steps", max_steps), max_steps)
            self._memory_search_confirmed = bool(
                resume_state.get("memory_search_confirmed", False)
            )
            self._history_archives = normalize_history_archives(
                resume_state.get("history_archives", [])
            )
            self._restore_tool_journal(resume_state.get("tool_journal", []))
            depth_mode = str(resume_state.get("depth_mode") or depth_mode)

            # Restore memory if available
            raw_memory = resume_state.get("memory")
            if isinstance(raw_memory, dict):
                self.memory.from_dict(raw_memory)
            self._restore_ephemeral_tracking(history)

            # Add continuation message
            if console:
                console.print(
                    "[bold green]🔄 Resuming analysis from saved session...[/bold green]"
                )

            self.logger.info("State restored: step %d/%d", step, max_steps)
        else:
            # Reset working memory for new research session
            self.memory.reset(keep_facts=False)
            self._reset_run_state()

            today = datetime.now().strftime("%d %B %Y")
            try:
                # Note: Prompt should be updated to not enforce JSON format strictly since we use tool calling
                system_prompt = self._prompt_resolver("autonomous_news_agent")

                # Inject output language from settings
                system_prompt = system_prompt.replace("{output_language}", self._output_lang)
            except Exception:
                # Fallback if resolver fails or prompt not loaded yet
                self.logger.warning(
                    "Could not resolve autonomous_news_agent prompt, using default string."
                )
                system_prompt = "You are a financial news agent. Use available tools to research and analyze."

            # Filter company data to only include essential AI context
            company_payload = get_ai_company_context(company_data)
            await self._bootstrap_memory_from_rag(company_name, company_data, console)

            # Initial news summary if provided
            news_summary = ""
            if initial_news:
                news_summary = (
                    f"\nAlready fetched {len(initial_news)} articles. Snippets:\n"
                )
                for i, art in enumerate(
                    initial_news[:5]
                ):  # Only first 5 to save tokens
                    url = art.get("link") or art.get("url") or "No URL"
                    news_summary += f"{i+1}. {art.get('title')} (Source: {art.get('source_info', 'Unknown')}) - URL: {url}\n"

            # Depth mode instructions
            depth_instructions = ""
            if depth_mode == "deep":
                depth_instructions = "\n\n**DEEP RESEARCH MODE ACTIVE:** Previous analysis was deemed insufficient. You MUST:\n- Consult at least 5 different sources\n- Verify key claims with independent evidence using available tools\n- Actively search for contradicting viewpoints\n- Update working memory after each discovery\n- Do NOT finish until research_depth_score >= 5"

            # Initial user input with context
            # Cache optimization: separate stable prefix from dynamic content
            company_context_str = json.dumps(
                company_payload, default=str, ensure_ascii=False
            )

            # Stable prefix for cache hits - static instruction that doesn't change
            stable_user_prefix = (
                f"{self._build_tool_reliability_notice()}"
                "\n\nBegin your research using the available tools and the provided context."
            )

            # Dynamic context in separate message - varies per request
            dynamic_context_message = (
                f"Date: {today}\n"
                f"Target: {company_name}\n"
                f"Context: {company_context_str}\n"
                f"{news_summary}"
                f"{depth_instructions}"
            )

            history = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": stable_user_prefix},
                {"role": "user", "content": dynamic_context_message},
            ]

            step = 0

        consecutive_no_tool_calls = 0
        _NON_DEDUP_TOOLS = {"update_working_memory", "finish", "search_memory"}
        executed_tool_signatures: set[str] = set()
        tool_request_kwargs = build_native_tool_request_kwargs(
            "news",
            max_parallel_tools=self._max_parallel_tools,
        )
        while step < max_steps:
            step += 1
            self.logger.info("Agent Step %d/%d", step, max_steps)

            if self._token_budget_enabled:
                token_count = estimate_tokens_for_messages(
                    history, self._token_encoding
                )
                if token_count > self._history_token_limit:
                    history = await self._summarize_history(history)
            self._maybe_inject_self_reflection(history, step)
            # elif total_chars > self._history_char_limit:
            #    history = await self._summarize_history(history)

            # Using async AI executor from async method to avoid event loop issues.
            try:
                response = await self._request_executor.send_async(
                    history,
                    request_type="news",
                    timeout_override=120,
                    console=console,
                    **tool_request_kwargs,
                )
            except Exception as e:
                self.logger.error("AI Request Failed: %s", e)
                session_path = self._save_session_snapshot(
                    active_session_id,
                    history=history,
                    step=step,
                    max_steps=max_steps,
                    company_name=company_name,
                    company_data=company_data,
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

            # 1. Update history with assistant response
            # Note: If there are tool calls, content might be null or thought process.
            # We must include the assistant message exactly as is logic-wise for the API.
            assistant_history_message = {"role": "assistant", "content": raw_content}
            if native_tool_calls:
                assistant_history_message["tool_calls"] = native_tool_call_payload
            history.append(assistant_history_message)

            # Bir önceki adımda işaretlenen geçici (ephemeral) araç çıktılarını sıkıştır.
            # fetch_url_content + summarize_url_content: son 2 çıktı geçmişte tutulur.
            # search_*: son 1 çıktı geçmişte tutulur (snippet'ler WM'ye kaydedildi).
            for _indices_list, _keep in (
                (self._fetch_indices, 2),
                (self._search_indices, 1),
            ):
                if len(_indices_list) > _keep:
                    for _idx in _indices_list[:-_keep]:
                        if _idx < len(history) and history[_idx].get("role") in {"user", "tool"}:
                            assistant_msg = self._find_previous_assistant(history, _idx)
                            tool_content = history[_idx].get("content", "")
                            self._capture_ephemeral_result_before_stub(
                                tool_name=self._extract_tool_name_from_result_message(
                                    history[_idx],
                                    history=history,
                                    message_index=_idx,
                                ),
                                tool_content=tool_content,
                                assistant_has_memory_update=self._assistant_has_memory_update(
                                    assistant_msg
                                ),
                            )
                            history[_idx]["content"] = _COMPRESSED_TOOL_RESULT_STUB
                    del _indices_list[:-_keep]

            if not tool_calls:
                if not content:
                    self.logger.error("Empty response from AI (no content, no tools)")
                    self._record_tool_journal_step(
                        step,
                        [],
                        assistant_summary="",
                        notes=["empty response; requested a tool call or final answer"],
                    )
                    history.append(
                        {
                            "role": "user",
                            "content": "Error: Empty response. Please use a tool or provide an answer.",
                        }
                    )
                    continue

                # Gerçek bir final raporu mu yoksa yarım / halusinasyon mu?
                is_final_report = len(content) > 600 and (
                    content.strip().startswith("#") or "\n##" in content
                )
                if not is_final_report:
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

                self.logger.info(
                    "Agent responded without tool calls (final report fallback)."
                )
                if step > 1 and len(content) > 100:
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
                        max_steps=max_steps,
                        company_name=company_name,
                        company_data=company_data,
                        depth_mode=depth_mode,
                    )
                    return {
                        "analysis": content,
                        "steps": step,
                        "model_used": response.get("model", "unknown"),
                        "confidence": 1.0,
                        "has_news_data": True,
                        "timestamp": datetime.now().timestamp(),
                        "step_limit_reached": False,
                        "session_id": active_session_id,
                        "session_path": session_path,
                        "resumed_from_snapshot": resumed_from_snapshot,
                    }
                else:
                    continue

            # 2. Process Tool Calls
            consecutive_no_tool_calls = 0
            tool_calls = [canonicalize_bist_market_tool_call(call) for call in tool_calls]
            preview_plan_rows = build_tool_plan_preview(
                tool_calls,
                max_parallel_tools=self._max_parallel_tools,
                non_dedup_tools=_NON_DEDUP_TOOLS,
                executed_signatures=executed_tool_signatures,
            )
            self._emit_reflection(
                content,
                preview_plan_rows,
                console,
                reasoning_streamed=bool(response.get("reasoning_streamed", False)),
            )
            # Check for update_working_memory tool first
            memory_calls = [
                t
                for t in tool_calls
                if t.get("name") == "update_working_memory"
            ]
            memory_update_payloads = [
                mem_call.get("args", {})
                for mem_call in memory_calls
                if isinstance(mem_call.get("args"), dict)
            ]
            for mem_call in memory_calls:
                mem_args = mem_call.get("args", {})
                if isinstance(mem_args, dict):
                    self._update_working_memory(mem_args, console)
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

            has_search_memory_call = any(
                t.get("name") == "search_memory"
                for t in (tool_calls or [])
            )
            if has_search_memory_call:
                self._memory_search_confirmed = True

            if not self._memory_search_confirmed and step <= 2:
                self.logger.warning(
                    "Mandatory search_memory not called yet (step %d). Forcing correction.",
                    step,
                )
                self._record_tool_journal_step(
                    step,
                    tool_calls,
                    assistant_summary=content,
                    memory_updates=memory_update_payloads,
                    notes=["mandatory search_memory enforcement triggered"],
                )
                for blocked_call in tool_calls:
                    if blocked_call.get("name") == "update_working_memory":
                        continue
                    if blocked_call.get("id"):
                        history.append(
                            build_tool_result_history_message(
                                tool_name=str(blocked_call.get("name") or "tool"),
                                result={
                                    "status": "blocked",
                                    "error": "search_memory must be called before other external tools or finish in the first two steps",
                                    "error_code": "mandatory_search_memory",
                                },
                                tool_call=blocked_call,
                            )
                        )
                history.append(
                    {
                        "role": "user",
                        "content": (
                            "⚠️ Mandatory requirement not satisfied: call `search_memory` now before other external searches or finish(). "
                            "Use query with company name and symbol, then update_working_memory with key retrieved facts."
                        ),
                    }
                )
                continue

            # Check for finish tool
            finish_call = next(
                (t for t in tool_calls if t.get("name") == "finish"), None
            )
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

            if finish_call:
                args = (
                    finish_call.get("args", {})
                    if isinstance(finish_call.get("args"), dict)
                    else {}
                )
                final_analysis = str(args.get("final_analysis", "") or "").strip()
                research_summary = str(args.get("research_summary", "") or "").strip()

                if not final_analysis and content and len(content.strip()) > 200:
                    self.logger.warning(
                        "finish() called with empty final_analysis; using assistant message content as fallback (%d chars).",
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
                    max_steps=max_steps,
                    company_name=company_name,
                    company_data=company_data,
                    depth_mode=depth_mode,
                )
                return {
                    "analysis": final_analysis,
                    "steps": step,
                    "model_used": response.get("model", "unknown"),
                    "confidence": 1.0,
                    "has_news_data": True,
                    "timestamp": datetime.now().timestamp(),
                    "working_memory": self.memory.to_dict(),
                    "research_summary": research_summary,
                    "step_limit_reached": False,
                    "session_id": active_session_id,
                    "session_path": session_path,
                    "resumed_from_snapshot": resumed_from_snapshot,
                }

            # Execute tools in parallel where possible
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
                    # Provide error feedback to model
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

                if fn_name not in _NON_DEDUP_TOOLS:
                    sig = make_tool_signature(fn_name, fn_args)
                    if sig and sig in executed_tool_signatures:
                        tasks.append(
                            self._mock_tool_error(
                                f"Duplicate tool call skipped: '{fn_name}' was already called with identical parameters in a previous step. Use different parameters or a different tool."
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

            # Wait for all tools
            results = await asyncio.gather(*tasks)
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

            # 3. Add Tool Outputs to History
            for i, result in enumerate(results):
                tool_name = param_map[i]["name"]
                tool_call = param_map[i]["tool_call"]

                history.append(
                    build_tool_result_history_message(
                        tool_name=tool_name,
                        result=result,
                        tool_call=tool_call,
                    )
                )

                # Araç çıktılarını türüne göre ephemeral listelerine kaydet.
                if tool_name in _EPHEMERAL_URL_TOOLS:
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

            # Kalan adım bütçesini AI'ya bildir.
            remaining = max_steps - step
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
                progress_context = self._progress_context()
                progress_context_text = (
                    f" | {progress_context}" if progress_context else ""
                )
                tool_journal_context = self._build_tool_journal_context()
                tool_journal_text = (
                    f"\n\n{tool_journal_context}" if tool_journal_context else ""
                )

                history.append(
                    {
                        "role": "user",
                        "content": (
                            f"[Progress: Step {step}/{max_steps} | {remaining} steps remaining | "
                            f"Research depth score: {depth_score}{progress_context_text}]"
                            f"{step_note}"
                            f"{tool_journal_text}"
                        ),
                    }
                )

            session_path = self._save_session_snapshot(
                active_session_id,
                history=history,
                step=step,
                max_steps=max_steps,
                company_name=company_name,
                company_data=company_data,
                depth_mode=depth_mode,
            )

            # Truncate history if getting too large? (Managed by context window usually, but good to keep in mind)

        # Max steps reached - save state and check if user wants to continue
        current_state = self._build_session_state(
            history=history,
            step=step,
            max_steps=max_steps,
            company_name=company_name,
            company_data=company_data,
            depth_mode=depth_mode,
        )
        session_path = self._save_session_snapshot(
            active_session_id,
            history=history,
            step=step,
            max_steps=max_steps,
            company_name=company_name,
            company_data=company_data,
            depth_mode=depth_mode,
        )

        # If callback provided, ask user if they want to continue
        if on_max_steps_callback:
            try:
                if console:
                    console.print(
                        f"\n[yellow]⚠️  Maximum operation limit reached ({max_steps} steps).[/yellow]"
                    )

                should_continue = on_max_steps_callback(current_state)

                if should_continue:
                    # Increase max_steps and continue
                    additional_steps = max(1, self._continuation_increment)
                    max_steps += additional_steps
                    current_state["max_steps"] = max_steps

                    if console:
                        console.print(
                            f"[bold green]✅ Continuing... (+{additional_steps} steps added)[/bold green]\n"
                        )

                    self.logger.info(
                        f"User chose to continue. New limit: {max_steps}"
                    )

                    # Continue the loop - return to caller so it can recursively call run again
                    return {
                        "needs_continuation": True,
                        "state": current_state,
                        "new_max_steps": max_steps,
                        "step_limit_reached": True,
                        "session_id": active_session_id,
                        "session_path": session_path,
                        "resumed_from_snapshot": resumed_from_snapshot,
                    }
            except Exception as e:
                self.logger.error("Continue callback failed: %s", e)

        # No continuation - build partial report from collected data
        partial_report = self._build_partial_report()

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
        }

    async def _mock_tool_error(self, error_msg: str) -> str:
        return json.dumps({"error": error_msg})

    def _update_working_memory(
        self, args: Dict, console: Optional[Console] = None
    ) -> None:
        """Updates the working memory with new research findings."""
        args = sanitize_memory_args(args)
        if looks_like_final_report_payload(args):
            self.logger.info(
                "Skipping working-memory update that appears to contain final report content; model should call finish() instead."
            )
            if console:
                console.print(
                    "[yellow]⚠️ Skipped working-memory update because it looked like final report content. The agent should call finish() instead.[/yellow]"
                )
            return

        if not has_actionable_memory_payload(args):
            self.logger.info(
                "Skipping working-memory update because no verified or actionable content remained after sanitization."
            )
            if console:
                console.print(
                    "[yellow]⚠️ Skipped working-memory update because no verified or actionable content remained after filtering.[/yellow]"
                )
            return

        self.memory.update_from_args(args)

        if console:
            counts = self.memory.summary_counts()
            score = counts["score"]
            facts = counts["facts"]
            sources = counts["sources"]
            console.print(
                f"[magenta]📝 Working Memory Updated: {facts} findings, {sources} sources, Depth Score: {score}[/magenta]"
            )

        self.logger.info(
            "Working Memory Updated: depth_score=%d",
            self._current_depth_score(),
        )

    async def _execute_tool_safe(
        self, tool_name: str, args: Dict, console: Optional[Console] = None
    ) -> str:
        # Handle internal working memory tool
        if tool_name == "update_working_memory":
            self._update_working_memory(args, console)
            return json.dumps(
                {
                    "status": "Working memory updated",
                    "current_depth_score": self._current_depth_score(),
                }
            )

        self.logger.info("Agent Action: %s", tool_name)
        if console:
            if (
                tool_name in ["search_web", "search_news", "search_google_news"]
                and "query" in args
            ):
                console.print(
                    f"[cyan]🔧 Running tool: {tool_name} -> [yellow]'{args['query']}'[/yellow][/cyan]"
                )
            elif (
                tool_name in ("fetch_url_content", "summarize_url_content")
                and "url" in args
            ):
                console.print(
                    f"[cyan]🔧 Running tool: {tool_name} -> [yellow]'{args['url']}'[/yellow][/cyan]"
                )
            else:
                console.print(f"[cyan]🔧 Running tool: {tool_name}[/cyan]")

        try:
            result = await execute_tool(tool_name, args=args, console=console)

            standardized_result = to_standard_tool_result(tool_name, result)
            result_str = json.dumps(standardized_result, default=str, ensure_ascii=False)

            # fetch_url_content (ham içerik) → agent digest filter ile sıkıştır.
            # summarize_url_content zaten retrieval-pruned içerik döndürüyor — ek digest gerekmez.
            if tool_name == "fetch_url_content" and not (
                isinstance(result, dict) and result.get("error")
            ):
                result_str = await self._fetch_url_with_digest(
                    url=args.get("url", ""),
                    raw_result=standardized_result,
                    fallback_result_str=result_str,
                    console=console,
                )

            if console:
                if isinstance(result, dict) and result.get("error"):
                    console.print(
                        f"[red]\u274c Tool error ({tool_name}): {result['error']}[/red]"
                    )
                else:
                    console.print(f"[green]\u2705 Tool completed: {tool_name}[/green]")

            return result_str
        except Exception as e:
            if console:
                console.print(f"[red]❌ Tool crashed ({tool_name}): {e}[/red]")
            return json.dumps({"error": f"Error executing tool {tool_name}: {str(e)}"})

    async def _fetch_url_with_digest(
        self,
        url: str,
        raw_result: Any,
        fallback_result_str: str,
        console: Optional[Console] = None,
    ) -> str:
        """
        fetch_url_content çıktısını mini-model (summarizer) aracılığıyla filtreler.
        Sayfadaki yalnızca çalışma belleğindeki açık sorulara yanıt veren kısımları çıkarır.
        Ana modelin context window'unu büyük ölçüde korur.
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
        try:
            digest_response = await self._request_executor.send_async(
                [
                    {"role": "system", "content": digest_system},
                    {"role": "user", "content": digest_user},
                ],
                request_type="summarizer",
                timeout_override=30,
            )
            digest_text = (digest_response.get("content") or "").strip()
            if not digest_text:
                return fallback_result_str
            self.logger.info(
                "fetch_url_content digest: %d \u2192 %d characters",
                len(prepared_payload),
                len(digest_text),
            )
            if console:
                console.print(
                    f"[dim]\U0001f52c URL summary: {len(prepared_payload):,} \u2192 {len(digest_text):,} characters[/dim]"
                )
            return json.dumps({"url": url, "digest": digest_text}, ensure_ascii=False)
        except Exception as exc:
            self.logger.warning(
                "fetch_url_content digest failed, using raw content: %s", exc
            )
            return fallback_result_str
