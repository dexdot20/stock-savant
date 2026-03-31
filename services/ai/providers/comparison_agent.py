import asyncio
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console

from config import get_config
from services.factories import get_rag_service, get_shared_memory_pool
from services.tools import execute_tool, get_tool_quality_metrics, to_standard_tool_result
from services.ai.memory_formatter import format_working_memory_evidence_pack
from services.ai.providers.token_utils import estimate_tokens_for_messages
from services.ai.providers.memory_rag_utils import (
    derive_spill_confidence,
    derive_spill_data_gap_count,
    format_fact_for_rag,
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
    make_tool_signature,
    normalize_tool_args,
    validate_tool_args,
)
from services.ai.working_memory import WorkingMemory
from services.ai.providers.research_agent_support import ResearchAgentSupportMixin

# Search tools that produce large raw output and need early cleanup from history.
# fetch_url_content / summarize_url_content are tracked separately (last 2 kept).
_EPHEMERAL_URL_TOOLS: frozenset = frozenset(
    {"fetch_url_content", "summarize_url_content"}
)
# search_* tools: keep last 1 (snippets become unnecessary after written to WM).
_EPHEMERAL_SEARCH_TOOLS: frozenset = frozenset({"search_news", "search_google_news"})

from domain.utils import safe_int_strict as _safe_int


class ComparisonAgent(ResearchAgentSupportMixin):
    """
    Agentic comparison agent for multiple companies.
    Uses tools to verify metrics and news, and produces a comparative report.
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

        agent_steps_cfg = config.get("ai", {}).get("agent_steps", {})
        self.max_steps = _safe_int(agent_steps_cfg.get("comparison", 40), 40)

        tool_limit_cfg = config.get("ai", {}).get("agent_tool_limits", {})
        raw_limit = _safe_int(tool_limit_cfg.get("max_parallel_tools", 0), 0)
        self._max_parallel_tools: Optional[int] = raw_limit if raw_limit > 0 else None

        # Cache config slice used in methods to avoid repeated deepcopy.
        self._output_lang: str = config.get("ai", {}).get("output_language", "English")
        self._config = config

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
            shared_memory_agent_name="comparison_agent",
        )
        self._fetch_indices: List[int] = []
        self._search_indices: List[int] = []
        self._progress_msg_index: Optional[int] = None
        self._last_consolidation_step: int = 0
        self._current_tickers: List[str] = []
        self._history_archives: List[Dict[str, Any]] = []
        self._tool_journal: List[Dict[str, Any]] = []

    def _build_partial_report(self) -> str:
        partial_report = (
            "# Step Limit Reached\n\n"
            "The agent reached the configured step limit and could not complete the comparison. "
            "Collected findings are listed below.\n\n"
        )

        memory_state = self.memory.to_dict()
        facts = [
            self._format_fact_text(fact)
            for fact in memory_state.get("facts_learned", [])
        ]
        facts = [fact for fact in facts if fact]
        if facts:
            partial_report += "## Findings\n"
            partial_report += "\n".join(f"- {fact}" for fact in facts)
            partial_report += "\n\n"

        sources = [
            str(source).strip()
            for source in memory_state.get("sources_consulted", [])
            if str(source).strip()
        ]
        if sources:
            partial_report += "## Sources Reviewed\n"
            partial_report += "\n".join(f"- {source}" for source in sources)
            partial_report += "\n\n"

        questions = [
            str(question).strip()
            for question in memory_state.get("unanswered_questions", [])
            if str(question).strip()
        ]
        if questions:
            partial_report += "## Yanitlanmamis Sorular\n"
            partial_report += "\n".join(f"- {question}" for question in questions)

        return partial_report

    def _save_session_snapshot(
        self,
        session_id: str,
        *,
        history: List[Dict[str, Any]],
        step: int,
        tickers: List[str],
        criteria: Optional[str],
        depth_mode: str,
        initial_data: Optional[List[Dict[str, Any]]],
    ) -> str:
        payload: Dict[str, Any] = {
            "tickers": list(tickers or []),
            "criteria": criteria,
            "depth_mode": depth_mode,
            "initial_data": initial_data,
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
                "tickers": list(tickers or []),
                "criteria": criteria,
                "depth_mode": depth_mode,
                "initial_data": initial_data,
                "tool_journal": list(self._tool_journal),
            },
        }
        return save_agent_session("comparison_agent", session_id, payload)

    def _emit_reflection(
        self,
        content: Optional[str],
        tool_plans: list[Dict[str, Any]],
        console: Optional[Console],
        reasoning_streamed: bool = False,
    ) -> None:
        if not self._reflection_enabled:
            return

        if content and content.strip():
            thought = content.strip()
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
                        title="Thinking",
                        border_style="blue",
                        padding=(0, 1),
                    )
                    console.print(panel)

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

                table = Table(
                    title=(
                        f"Plan: {active_count} tools executing now"
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
                        args_parts = []
                        for k, v in args.items():
                            v_str = str(v)
                            if len(v_str) > 60:
                                v_str = v_str[:57] + "..."
                            args_parts.append(f"{k}={v_str}")
                        args_display = "\n".join(args_parts)
                    except Exception:
                        args_display = "{}"

                    status = str(plan.get("status") or "planned")
                    note = str(plan.get("note") or "").strip()
                    if note:
                        status = f"{status}\n{note}"

                    table.add_row(str(i), name, status, args_display)

                console.print(table)
                console.print()

    async def _consolidate_working_memory_if_needed(
        self, step: int, console: Optional[Console] = None
    ) -> bool:
        """facts_learned eşiği aşıldığında LLM (summarizer) ile sıkıştırır.
        En az 5 adım geçmedikçe tekrar tetiklenmez.
        """
        if not self.memory.needs_facts_consolidation():
            return False
        if step - self._last_consolidation_step < 5:
            return False

        fact_records = list(self.memory.to_dict().get("facts_learned", []))
        facts = [self._format_fact_text(fact) for fact in fact_records if self._format_fact_text(fact)]
        original_count = len(facts)
        self.logger.info(
            "Working memory consolidation started: %d facts", original_count
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
                timeout_override=30,
            )
            raw = (response.get("content") or "").strip()
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                self.logger.warning(
                    "Working memory consolidation skipped: JSON array not found"
                )
                return False
            consolidated = json.loads(match.group())
            if not isinstance(consolidated, list) or not consolidated:
                self.logger.warning(
                    "Working memory consolidation skipped: invalid output"
                )
                return False

            try:
                rag = get_rag_service()
                rag.index_analysis(
                    symbol=",".join(self._current_tickers) or "UNKNOWN",
                    news_output="\n".join(
                        item
                        for item in (format_fact_for_rag(fact) for fact in fact_records)
                        if item
                    ),
                    final_output=None,
                    confidence_score=derive_spill_confidence(fact_records, default=0.5),
                    data_gaps_count=derive_spill_data_gap_count(fact_records),
                )
            except Exception as rag_exc:
                self.logger.debug("Comparison spill-over indexing skipped: %s", rag_exc)

            self.memory.replace_facts([str(f) for f in consolidated])
            new_count = self.memory.summary_counts()["facts"]
            self._last_consolidation_step = step
            savings_pct = round((1 - new_count / original_count) * 100)
            self.logger.info(
                "Working memory consolidation completed: %d -> %d facts (%d%% saved)",
                original_count,
                new_count,
                savings_pct,
            )
            if console:
                console.print(
                    f"[magenta]🧠 Memory Consolidation: {original_count} → {new_count} facts "
                    f"([green]{savings_pct}% saved[/green])[/magenta]"
                )
            return True
        except Exception as exc:
            self.logger.warning("Working memory consolidation failed: %s", exc)
            return False

    async def run(
        self,
        tickers: List[str],
        criteria: Optional[str] = None,
        initial_data: Optional[List[Dict[str, Any]]] = None,
        console: Optional[Console] = None,
        depth_mode: str = "standard",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._current_tickers = [
            str(t).strip().upper() for t in (tickers or []) if str(t).strip()
        ]
        self.memory.configure_shared_memory(
            scope="comparison:" + ",".join(self._current_tickers),
            agent_name="comparison_agent",
        )
        self.memory.refresh_from_shared_pool(
            " ".join(self._current_tickers)
            + " comparison relative performance risks opportunities",
            top_k=_safe_int(
                self._config.get("ai", {})
                .get("working_memory", {})
                .get("shared_pool_search_top_k", 4),
                4,
            ),
            tags=["shared_pool", "prior_agent_memory"],
            milestone="Shared memory warm-start completed at comparison start",
        )
        active_session_id = str(session_id or "").strip() or generate_agent_session_id(
            "comparison", ",".join(self._current_tickers)
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
            snapshot = load_agent_session("comparison_agent", active_session_id)
            raw_state = snapshot.get("state") if isinstance(snapshot, dict) else None
            if isinstance(raw_state, dict):
                loaded_state = raw_state
                resumed_from_snapshot = True

        if loaded_state:
            saved_tickers = loaded_state.get("tickers")
            if isinstance(saved_tickers, list):
                self._current_tickers = [
                    str(item).strip().upper()
                    for item in saved_tickers
                    if str(item).strip()
                ]
            criteria = (
                str(loaded_state.get("criteria"))
                if loaded_state.get("criteria") is not None
                else criteria
            )
            depth_mode = str(loaded_state.get("depth_mode") or depth_mode)
            saved_initial_data = loaded_state.get("initial_data")
            if isinstance(saved_initial_data, list):
                initial_data = saved_initial_data
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
                url_tools=_EPHEMERAL_URL_TOOLS,
            )
            self._restore_progress_tracking(history)
            if console:
                console.print(
                    "[bold green]🔄 Resuming comparison from saved session...[/bold green]"
                )
        else:
            self.memory.reset(keep_facts=True)
            self._history_archives = []
            self._tool_journal = []
            step = 0

        today = datetime.now().strftime("%d %B %Y")
        try:
            system_prompt = self._prompt_resolver("comparison_agent")
            system_prompt = system_prompt.replace("{output_language}", self._output_lang)
        except Exception:
            self.logger.warning("comparison_agent prompt not found, using fallback.")
            system_prompt = augment_system_prompt(
                "You are a financial comparison agent. Use tools to verify and compare.",
                config=self._config,
            )

        depth_instructions = ""
        if depth_mode == "deep":
            depth_instructions = (
                "\n\nDEEP MODE: Use at least 3 sources, read 2 full articles, "
                "verify 1 claim, and check for contradictions."
            )

        criteria_text = criteria.strip() if isinstance(criteria, str) else ""
        tickers_text = ", ".join(tickers) if tickers else "N/A"
        data_blob = (
            json.dumps(initial_data, ensure_ascii=False, default=str)
            if initial_data
            else "None"
        )
        user_message = (
            f"Date: {today}\n"
            f"Tickers to compare: {tickers_text}\n"
            f"Additional criteria: {criteria_text or 'None'}\n"
            f"{depth_instructions}\n\n"
            "Current Financial Summary (JSON):\n"
            f"{data_blob}\n\n"
            "Working Memory Protocol: use update_working_memory only after meaningful, source-backed findings or a small related batch.\n"
            "Put unverified interpretations into questions or research milestones instead of new_facts.\n"
            f"{self._build_tool_reliability_notice()}"
            "Start the comparison and present the differences in a clear report."
        )

        if not loaded_state:
            history = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

        total_non_memory_tools_executed = 0
        total_memory_updates = 0
        if loaded_state:
            (
                total_non_memory_tools_executed,
                total_memory_updates,
            ) = self._restore_execution_counters(history)
        consecutive_no_tool_calls = 0
        _NON_DEDUP_TOOLS = {"update_working_memory", "finish", "search_memory"}
        executed_tool_signatures: set[str] = set()
        tool_request_kwargs = build_native_tool_request_kwargs(
            "comparison",
            max_parallel_tools=self._max_parallel_tools,
        )

        while step < self.max_steps:
            step += 1
            self.logger.info("Comparison Step %d/%d", step, self.max_steps)

            if self._token_budget_enabled:
                token_count = estimate_tokens_for_messages(
                    history, self._token_encoding
                )
                if token_count > self._history_token_limit:
                    history = await self._summarize_history(history)
            self._maybe_inject_self_reflection(history, step)

            try:
                response = await self._request_executor.send_async(
                    history,
                    request_type="comparison",
                    timeout_override=120,
                    console=console,
                    **tool_request_kwargs,
                )
            except Exception as e:
                self.logger.error("Comparison AI request failed: %s", e)
                session_path = self._save_session_snapshot(
                    active_session_id,
                    history=history,
                    step=step,
                    tickers=self._current_tickers,
                    criteria=criteria,
                    depth_mode=depth_mode,
                    initial_data=initial_data,
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

            # Bir önceki adımda işaretlenen geçici (ephemeral) araç çıktılarını sıkıştır.
            # fetch/summarize: son 2, search_*: son 1 çıktı tutulur.
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

                # Gerçek bir final raporu mu yoksa yarım / halusinasyon mu?
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
                        tickers=self._current_tickers,
                        criteria=criteria,
                        depth_mode=depth_mode,
                        initial_data=initial_data,
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
                        "Premature finish blocked in comparison: %s", reason
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
                    tickers=self._current_tickers,
                    criteria=criteria,
                    depth_mode=depth_mode,
                    initial_data=initial_data,
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

            # Araç çıktıları işlendikten sonra bellek konsolidasyonunu kontrol et.
            await self._consolidate_working_memory_if_needed(step, console)

            # Kalan adım bütçesini AI'ya bildir
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

                # Model araç sonucu aldı ama belleği güncellemedi — blokaj hatırlatması.
                memory_reminder = ""
                if (
                    executed_tools
                    and not memory_was_updated
                    and remaining > 3
                    and any(tool_name in VERIFIED_MEMORY_TOOLS for tool_name in executed_tools)
                ):
                    tools_str = ", ".join(executed_tools)
                    memory_reminder = (
                        f"\n\nℹ️ You received source-backed results from [{tools_str}] without a memory update. "
                        "If they contain verified figures or durable findings, save them with `update_working_memory`. "
                        "Store verified facts only; put tentative interpretations into questions or milestones."
                    )

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
                tickers=self._current_tickers,
                criteria=criteria,
                depth_mode=depth_mode,
                initial_data=initial_data,
            )

        partial_report = self._build_partial_report()

        session_path = self._save_session_snapshot(
            active_session_id,
            history=history,
            step=step,
            tickers=self._current_tickers,
            criteria=criteria,
            depth_mode=depth_mode,
            initial_data=initial_data,
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
        }

    async def _mock_tool_error(self, error_msg: str) -> str:
        return json.dumps({"error": error_msg})

    async def _fetch_url_with_digest(
        self,
        url: str,
        raw_result: Any,
        fallback_result_str: str,
        console: Optional[Console] = None,
    ) -> str:
        """
        fetch_url_content (ham içerik) çıktısını mini-model (summarizer) aracılığıyla filtreler.
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
                "fetch_url_content digest: %d \u2192 %d karakter",
                len(prepared_payload),
                len(digest_text),
            )
            if console:
                console.print(
                    f"[dim]\U0001f52c URL özeti: {len(prepared_payload):,} \u2192 {len(digest_text):,} karakter[/dim]"
                )
            return json.dumps({"url": url, "digest": digest_text}, ensure_ascii=False)
        except Exception as exc:
            self.logger.warning(
                "fetch_url_content digest başarısız, ham içerik kullanılıyor: %s", exc
            )
            return fallback_result_str

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

            result = await execute_tool(tool_name, args=args, console=console)

            standardized_result = to_standard_tool_result(tool_name, result)
            result_str = json.dumps(standardized_result, ensure_ascii=False, default=str)

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
            return result_str
        except Exception as exc:
            return json.dumps({"error": str(exc)})

