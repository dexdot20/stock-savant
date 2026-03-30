import json
import re
from typing import Any, Dict, List, Optional

from rich.console import Console

from services.ai.memory_formatter import (
    format_working_memory_evidence_pack,
    format_working_memory_for_llm,
)
from services.ai.providers.agent_guardrails import (
    has_actionable_memory_payload,
    looks_like_final_report_payload,
    sanitize_memory_args,
)
from services.ai.providers.context_preservation_utils import archive_history_segment
from services.ai.providers.context_preservation_utils import build_ephemeral_evidence_record
from services.ai.providers.history_utils import select_immediate_context
from services.ai.providers.reflection_prompt_utils import (
    build_reflection_prompt,
    should_inject_reflection,
)
from services.ai.providers.tool_journal_utils import (
    format_tool_journal_for_prompt,
    normalize_tool_journal,
    normalize_tool_journal_step,
    summarize_memory_update_args,
    summarize_tool_result,
)
from services.ai.providers.tool_call_parser import parse_tool_calls_from_content
from services.ai.providers.system_prompt_utils import augment_system_prompt
from services.tools import get_tool_quality_metrics


class ResearchAgentSupportMixin:
    _history_summary_label = "research"
    _tool_reliability_guidance = (
        "verify critical claims with independent sources."
    )
    _working_memory_console_message = (
        "Working memory updated: {facts} facts, {sources} sources, Depth Score: {score}"
    )

    async def _summarize_history(
        self, history: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        if len(history) < 5:
            return history

        self.logger.info(
            "History summarization started (%s)", self._history_summary_label
        )
        self.logger.debug(
            "History summary input: messages=%d chars=%d",
            len(history),
            sum(len(str(item.get("content", ""))) for item in history),
        )
        immediate_context = select_immediate_context(
            history,
            token_limit=self._history_token_limit,
            encoding=self._token_encoding,
            min_keep=4,
        )
        cut_index = len(history) - len(immediate_context)
        anchor_index = self._find_first_non_system_index(history)
        preserve_until = anchor_index + 1 if anchor_index is not None else 1
        to_summarize = history[preserve_until:cut_index]

        if not to_summarize:
            return history

        archive_history_segment(
            self._history_archives,
            segment=list(to_summarize),
            reason="history_summarized",
        )

        try:
            prompt_template = self._prompt_resolver("history_summarizer")
            output_lang = self._output_lang
            system_prompt = prompt_template.replace("{output_language}", output_lang)
        except Exception:
            system_prompt = augment_system_prompt(
                "Summarize the following conversation history, preserving key facts and numbers.",
                config=self._config,
            )

        context_str = json.dumps(to_summarize, default=str, ensure_ascii=False)
        memory_snapshot = format_working_memory_for_llm(
            self.memory.to_dict(), style="snapshot"
        )
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
                    "Current Working Memory (preserve ALL facts, numbers, dates and figures below):\n"
                    f"{memory_snapshot}\n\n"
                    f"Current Evidence Pack:\n{evidence_pack}\n\n"
                    f"Conversation History to Compress:\n{context_str}"
                ),
            },
        ]

        try:
            response = await self._request_executor.send_async(
                summary_request_history,
                request_type="summarizer",
                timeout_override=60,
            )
            summary_text = response.get("content", "Summary failed.")

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
                "History summarization completed (%s). Messages: %d -> %d",
                self._history_summary_label,
                len(history),
                len(new_history),
            )
            self.logger.debug(
                "History summary output: archived=%d retained=%d evidence_facts=%d",
                len(to_summarize),
                len(new_history),
                len(self.memory.to_dict().get("facts_learned", [])),
            )
            return new_history
        except Exception as exc:
            self.logger.error("History summarization failed: %s", exc)
            return history

    @staticmethod
    def _find_first_non_system_index(history: list[Dict[str, Any]]) -> Optional[int]:
        for index, message in enumerate(history):
            role = str(message.get("role") or "").strip().lower()
            if role not in {"system", "developer"}:
                return index
        return None

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
        self.logger.debug("Tool reliability notice generated: %s", unstable_text)
        return (
            "\n\n⚠️ Tool Reliability Notice: These tools have elevated failure rates in this runtime: "
            f"{unstable_text}. Prefer alternatives when possible and {self._tool_reliability_guidance}"
        )

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
    def _format_fact_text(fact: Any) -> str:
        if isinstance(fact, dict):
            return str(fact.get("text", "")).strip()
        return str(fact).strip()

    def _restore_ephemeral_tracking(
        self,
        history: List[Dict[str, Any]],
        *,
        search_tools: frozenset[str],
        url_tools: Optional[frozenset[str]] = None,
    ) -> None:
        self._fetch_indices = []
        self._search_indices = []
        active_url_tools = url_tools or frozenset()
        for idx, message in enumerate(history):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if not isinstance(content, str) or not content.startswith("<tool_result"):
                continue
            match = re.search(r'<tool_result\s+name="([^"]+)"', content)
            tool_name = match.group(1) if match else ""
            if tool_name in active_url_tools:
                self._fetch_indices.append(idx)
            elif tool_name in search_tools:
                self._search_indices.append(idx)

    def _restore_progress_tracking(self, history: List[Dict[str, Any]]) -> None:
        self._progress_msg_index = None
        for idx in range(len(history) - 1, -1, -1):
            message = history[idx]
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str) and content.startswith("[Progress:"):
                self._progress_msg_index = idx
                return

    def _restore_execution_counters(
        self, history: List[Dict[str, Any]]
    ) -> tuple[int, int]:
        total_non_memory_tools_executed = 0
        total_memory_updates = 0

        for message in history:
            role = message.get("role")
            content = message.get("content", "")
            if (
                role == "user"
                and isinstance(content, str)
                and content.startswith("<tool_result")
            ):
                match = re.search(r'<tool_result\s+name="([^"]+)"', content)
                tool_name = match.group(1) if match else ""
                if tool_name and tool_name not in ("finish", "update_working_memory"):
                    total_non_memory_tools_executed += 1
            elif role == "assistant":
                tool_calls, _ = parse_tool_calls_from_content(content)
                total_memory_updates += sum(
                    1
                    for call in tool_calls
                    if call.get("name") == "update_working_memory"
                )

        return total_non_memory_tools_executed, total_memory_updates

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
        parsed_calls, _ = parse_tool_calls_from_content(assistant_msg.get("content", ""))
        for call in parsed_calls:
            if call.get("name") == "update_working_memory":
                return True
        return False

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

    def _update_working_memory(
        self, args: Dict[str, Any], console: Optional[Console] = None
    ) -> None:
        args = sanitize_memory_args(args)
        self.logger.debug(
            "Working memory update requested: keys=%s facts=%d questions=%d contradictions=%d milestones=%d",
            sorted(list(args.keys())),
            len(args.get("new_facts") or []),
            len(args.get("new_questions") or []),
            len(args.get("contradictions") or []),
            len(args.get("research_milestones") or []),
        )
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

        counts = self.memory.summary_counts()
        self.logger.info(
            "Working memory updated: facts=%d sources=%d questions=%d contradictions=%d milestones=%d score=%d",
            counts["facts"],
            counts["sources"],
            counts["questions"],
            counts["contradictions"],
            counts["milestones"],
            counts["score"],
        )

        if console:
            console.print(
                "[magenta]"
                + self._working_memory_console_message.format(
                    facts=counts["facts"],
                    sources=counts["sources"],
                    score=counts["score"],
                )
                + "[/magenta]"
            )

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

        fallback_preview = str((evidence_record or {}).get("preview") or "").strip()
        if not fallback_preview:
            return

        fallback_fact = (
            f"[Fallback:{tool_name}] Ephemeral çıktı stub öncesi otomatik yedeklendi: "
            f"{fallback_preview}"
        )
        self.memory.update(
            new_facts=[fallback_fact],
            fact_importance=6,
            fact_tags=["fallback", "ephemeral"],
            research_milestones=[
                "Ephemeral araç çıktısı stub öncesi otomatik yedeklendi"
            ],
            source_summary="fallback:auto_backup",
        )