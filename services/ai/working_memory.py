from __future__ import annotations

import asyncio
import difflib
import json
import logging
import re
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Default trimming limits for each field.
DEFAULT_MAX_FACTS: int = 60
DEFAULT_MAX_SOURCES: int = 30
DEFAULT_MAX_QUESTIONS: int = 20
DEFAULT_MAX_CONTRADICTIONS: int = 20
DEFAULT_MAX_EVIDENCE_RECORDS: int = 24

# Default threshold where LLM-based consolidation is triggered.
DEFAULT_CONSOLIDATION_THRESHOLD: int = 45
DEFAULT_MAX_FACTS_ADAPTIVE_LIMIT: int = 90
DEFAULT_ADAPTIVE_USAGE_STEP: float = 0.2
DEFAULT_SIMILARITY_THRESHOLD: float = 0.82
DEFAULT_SIMILARITY_WINDOW: int = 12
DEFAULT_CONSOLIDATION_SIMILARITY_RATIO: float = 0.35
DEFAULT_CONSOLIDATION_CACHE_SIZE: int = 32
DEFAULT_IMPORTANCE_RECALC_INTERVAL_SECONDS: int = 120

# Consolidation callback type alias (async): texts → consolidated texts
ConsolidationCallback = Callable[[List[str]], Awaitable[List[str]]]
SpillCallback = Callable[[List["Fact"]], None]


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _tokenize_text(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_.%:-]+", _normalize_text(value)))


def _jaccard_similarity(left: str, right: str) -> float:
    left_tokens = _tokenize_text(left)
    right_tokens = _tokenize_text(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _fact_similarity_score(left: str, right: str) -> float:
    normalized_left = _normalize_text(left)
    normalized_right = _normalize_text(right)
    token_score = _jaccard_similarity(normalized_left, normalized_right)
    sequence_score = difflib.SequenceMatcher(
        None, normalized_left, normalized_right
    ).ratio()
    return max(token_score, sequence_score)


def _merge_fact_details(
    fact: "Fact",
    *,
    importance: int,
    shared_reference_count: int,
    tags: Sequence[str],
    pinned: bool,
    provenance: Optional[Dict[str, Any]],
) -> None:
    fact.base_importance = max(fact.base_importance, importance)
    fact.importance = max(fact.importance, importance)
    fact.shared_reference_count = max(
        int(fact.shared_reference_count or 0), int(shared_reference_count or 0)
    )
    if tags:
        merged_tags = list(dict.fromkeys([*fact.tags, *[str(tag) for tag in tags]]))
        fact.tags = [tag for tag in merged_tags if str(tag).strip()]
    fact.pinned = fact.pinned or bool(pinned)
    fact.last_accessed = _now_iso()
    fact.access_count = max(0, int(fact.access_count or 0)) + 1
    if provenance:
        fact.provenance = {
            **dict(fact.provenance or {}),
            **_clean_provenance(provenance),
        }


def _find_similar_fact(
    facts: Sequence["Fact"],
    text: str,
    similarity_threshold: float,
    similarity_window: int,
) -> Optional["Fact"]:
    if similarity_threshold <= 0:
        return None

    recent_facts = list(facts[-max(1, int(similarity_window)) :])
    best_match: Optional[Fact] = None
    best_score = 0.0
    for fact in reversed(recent_facts):
        score = _fact_similarity_score(fact.text, text)
        if score >= similarity_threshold and score > best_score:
            best_match = fact
            best_score = score
    return best_match


def _similarity_density(
    facts: Sequence["Fact"],
    *,
    similarity_threshold: float,
    window: int,
) -> float:
    comparable = [fact for fact in facts if fact.text.strip()]
    comparable = comparable[-max(2, int(window)) :]
    if len(comparable) < 2:
        return 0.0

    comparisons = 0
    similar_pairs = 0
    for idx, left in enumerate(comparable[:-1]):
        for right in comparable[idx + 1 :]:
            comparisons += 1
            if _fact_similarity_score(left.text, right.text) >= similarity_threshold:
                similar_pairs += 1
    if comparisons == 0:
        return 0.0
    return similar_pairs / comparisons


def _truncate_preview(text: str, limit: int = 800) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= max(120, int(limit)):
        return normalized
    return normalized[: max(120, int(limit))].rstrip() + "..."


def _extract_numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?%?", str(text or "")))


def _unique_texts(items: Iterable[str], similarity_threshold: float) -> List[str]:
    unique: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        if any(
            _fact_similarity_score(existing, text) >= similarity_threshold
            for existing in unique
        ):
            continue
        unique.append(text)
    return unique


def _ensure_list(value: Any) -> List[str]:
    """Convert any value to a clean list of strings."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


from domain.utils import safe_int_strict as _safe_int


def _clean_provenance(value: Any) -> Dict[str, Any]:
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return {"source_id": ", ".join(parts)} if parts else {}

    if not isinstance(value, dict):
        text = str(value or "").strip()
        return {"source_id": text} if text else {}

    cleaned: Dict[str, Any] = {}
    for key, raw in value.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        if raw is None:
            continue
        if isinstance(raw, (str, int, float, bool)):
            if isinstance(raw, str) and not raw.strip():
                continue
            cleaned[key_text] = raw
    return cleaned


def _clean_evidence_record(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    cleaned: Dict[str, Any] = {}
    for key in (
        "tool_name",
        "source_id",
        "title",
        "url",
        "captured_at",
        "preview",
        "payload_kind",
    ):
        raw = value.get(key)
        text = str(raw or "").strip()
        if text:
            cleaned[key] = text

    tags_raw = value.get("tags", [])
    if isinstance(tags_raw, list):
        tags = [str(tag).strip() for tag in tags_raw if str(tag).strip()]
        if tags:
            cleaned["tags"] = tags

    metadata = value.get("metadata")
    if isinstance(metadata, dict):
        normalized_meta = _clean_provenance(metadata)
        if normalized_meta:
            cleaned["metadata"] = normalized_meta

    return cleaned


def _evidence_key(record: Dict[str, Any]) -> str:
    tool_name = str(record.get("tool_name") or "").strip().lower()
    source_id = str(
        record.get("source_id") or record.get("url") or record.get("title") or ""
    ).strip().lower()
    preview = str(record.get("preview") or "")[:240]
    return _normalize_text(f"{tool_name}|{source_id}|{preview}")


def _normalize_importance(value: Any, default: int = 5) -> int:
    if isinstance(value, str):
        mapped = {
            "low": 3,
            "medium": 5,
            "med": 5,
            "high": 8,
            "critical": 10,
        }.get(value.strip().lower())
        if mapped is not None:
            return mapped
    return _safe_int(value, default)


def _legacy_fact_provenance(item: Dict[str, Any]) -> Dict[str, Any]:
    provenance = _clean_provenance(item.get("provenance", {}))
    if provenance:
        return provenance
    return _clean_provenance(
        {
            "source_id": item.get("source_id") or item.get("source"),
            "timestamp": item.get("timestamp") or item.get("date"),
            "evidence_type": item.get("evidence_type") or item.get("source"),
        }
    )


def _now_iso() -> str:
    return datetime.now().isoformat()


def _parse_datetime(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Fact dataclass — importance score and contextual tag support
# ---------------------------------------------------------------------------


@dataclass
class Fact:
    """Represents a single research fact.

    Attributes:
        text:       The fact's text.
        importance: Importance score between 1-10. Higher value = more valuable.
                    When capacity is full, lowest score fact is deleted.
        tags:       Contextual tags (e.g., ["finance", "news"]).
                    Filterable via get_facts(tag=...).
    """

    text: str
    importance: int = 5
    base_importance: int = 5
    tags: List[str] = field(default_factory=list)
    pinned: bool = False
    provenance: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)
    last_accessed: str = field(default_factory=_now_iso)
    access_count: int = 0
    shared_reference_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "importance": self.importance,
            "base_importance": self.base_importance,
            "tags": list(self.tags),
            "pinned": self.pinned,
            "provenance": dict(self.provenance),
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "shared_reference_count": self.shared_reference_count,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Fact":
        current_importance = int(data.get("importance", 5))
        return Fact(
            text=str(data.get("text", "")),
            importance=current_importance,
            base_importance=int(data.get("base_importance", current_importance)),
            tags=list(data.get("tags", [])),
            pinned=_to_bool(data.get("pinned", False)),
            provenance=_clean_provenance(data.get("provenance", {})),
            created_at=str(data.get("created_at") or _now_iso()),
            last_accessed=str(data.get("last_accessed") or data.get("created_at") or _now_iso()),
            access_count=max(0, _safe_int(data.get("access_count", 0), 0)),
            shared_reference_count=max(
                0,
                _safe_int(data.get("shared_reference_count", 0), 0),
            ),
        )


# ---------------------------------------------------------------------------
# State & Memory classes
# ---------------------------------------------------------------------------


@dataclass
class WorkingMemoryState:
    facts_learned: List[Fact] = field(default_factory=list)
    sources_consulted: List[str] = field(default_factory=list)
    unanswered_questions: List[str] = field(default_factory=list)
    contradictions_found: List[str] = field(default_factory=list)
    research_milestones: List[str] = field(default_factory=list)
    rejected_hypotheses: List[str] = field(default_factory=list)
    evidence_trail: List[Dict[str, Any]] = field(default_factory=list)
    research_depth_score: int = 0


class WorkingMemory:
    """Per-agent working memory.

    New features:
    - **Priority-Based Eviction:** When capacity is full, instead of deleting the
      oldest fact, the fact with the lowest importance score is deleted.
    - **Contextual Tagging:** Each fact carries a tag list; filterable via get_facts(tag=...)
    - **Self-Managing Consolidation:** If consolidation_callback is provided, when the
      threshold is exceeded, the memory consolidates itself in the background (asyncio.Task).
    """

    def __init__(
        self,
        initial_facts: Optional[Iterable[str]] = None,
        consolidation_callback: Optional[ConsolidationCallback] = None,
        spill_callback: Optional[SpillCallback] = None,
        *,
        max_facts: int = DEFAULT_MAX_FACTS,
        max_sources: int = DEFAULT_MAX_SOURCES,
        max_questions: int = DEFAULT_MAX_QUESTIONS,
        max_contradictions: int = DEFAULT_MAX_CONTRADICTIONS,
        max_evidence_records: int = DEFAULT_MAX_EVIDENCE_RECORDS,
        consolidation_threshold: int = DEFAULT_CONSOLIDATION_THRESHOLD,
        adaptive_max_facts: int = DEFAULT_MAX_FACTS_ADAPTIVE_LIMIT,
        adaptive_usage_step: float = DEFAULT_ADAPTIVE_USAGE_STEP,
        fact_similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        fact_similarity_window: int = DEFAULT_SIMILARITY_WINDOW,
        consolidation_similarity_ratio: float = DEFAULT_CONSOLIDATION_SIMILARITY_RATIO,
        consolidation_cache_size: int = DEFAULT_CONSOLIDATION_CACHE_SIZE,
        importance_recalc_interval_seconds: int = DEFAULT_IMPORTANCE_RECALC_INTERVAL_SECONDS,
        shared_memory_pool: Optional[Any] = None,
        shared_memory_scope: Optional[str] = None,
        shared_memory_agent_name: Optional[str] = None,
    ) -> None:
        self._state = WorkingMemoryState()
        self._seen_facts: set[str] = set()
        self._seen_questions: set[str] = set()
        self._seen_contradictions: set[str] = set()
        self._seen_sources: set[str] = set()
        self._seen_milestones: set[str] = set()
        self._seen_rejected_hypotheses: set[str] = set()
        self._seen_evidence: set[str] = set()
        self._question_importance: Dict[str, int] = {}
        self._contradiction_importance: Dict[str, int] = {}
        # Callback: async fn(List[str]) -> List[str] — if not provided, automatic consolidation is disabled
        self._consolidation_callback: Optional[ConsolidationCallback] = (
            consolidation_callback
        )
        self._spill_callback: Optional[SpillCallback] = spill_callback
        # Flag to prevent concurrent callback invocation
        self._consolidating: bool = False
        self._consolidation_task: Optional[asyncio.Task[Any]] = None
        self._base_max_facts = max(5, int(max_facts))
        self._max_facts = self._base_max_facts
        self._max_sources = max(5, int(max_sources))
        self._max_questions = max(3, int(max_questions))
        self._max_contradictions = max(3, int(max_contradictions))
        self._max_evidence_records = max(5, int(max_evidence_records))
        self._consolidation_threshold = max(3, int(consolidation_threshold))
        self._adaptive_max_facts = max(self._base_max_facts, int(adaptive_max_facts))
        self._adaptive_usage_step = min(max(float(adaptive_usage_step), 0.05), 1.0)
        self._fact_similarity_threshold = min(
            max(float(fact_similarity_threshold), 0.0), 1.0
        )
        self._fact_similarity_window = max(2, int(fact_similarity_window))
        self._consolidation_similarity_ratio = min(
            max(float(consolidation_similarity_ratio), 0.0), 1.0
        )
        self._consolidation_cache_size = max(4, int(consolidation_cache_size))
        self._consolidation_cache: OrderedDict[str, List[str]] = OrderedDict()
        self._importance_recalc_interval_seconds = max(
            5, int(importance_recalc_interval_seconds)
        )
        self._last_importance_refresh = 0.0
        self._shared_memory_pool = shared_memory_pool
        self._shared_memory_scope = str(shared_memory_scope or "").strip() or None
        self._shared_memory_agent_name = (
            str(shared_memory_agent_name or "").strip() or None
        )

        if initial_facts:
            self.add_facts(initial_facts)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, keep_facts: bool = False) -> None:
        facts = list(self._state.facts_learned) if keep_facts else []
        self._state = WorkingMemoryState(facts_learned=facts)
        self._seen_facts = {_normalize_text(f.text) for f in facts}
        self._seen_questions.clear()
        self._seen_contradictions.clear()
        self._seen_sources.clear()
        self._seen_milestones.clear()
        self._seen_rejected_hypotheses.clear()
        self._seen_evidence.clear()
        self._question_importance.clear()
        self._contradiction_importance.clear()
        self._consolidating = False
        self._consolidation_task = None
        self._consolidation_cache.clear()
        self._last_importance_refresh = 0.0

    # ------------------------------------------------------------------
    # Adding facts
    # ------------------------------------------------------------------

    def add_facts(
        self,
        facts: Iterable[str],
        importance: int = 5,
        tags: Optional[List[str]] = None,
        pinned: bool = False,
        provenance: Optional[Dict[str, Any]] = None,
        shared_reference_count: int = 0,
    ) -> None:
        """Add new items to the facts list.

        Args:
            facts:      List of texts to add.
            importance: Importance score between 1-10 (default 5).
            tags:       List of contextual tags.
        """
        normalized_facts = [str(item).strip() for item in facts if str(item).strip()]
        if not normalized_facts:
            return
        projected_usage = (
            len(self._state.facts_learned) + len(normalized_facts)
        ) / max(1, self._base_max_facts)
        self.adjust_limits(projected_usage)
        added = _add_facts_capped(
            self._state.facts_learned,
            self._seen_facts,
            normalized_facts,
            self._max_facts,
            importance,
            tags or [],
            pinned,
            provenance,
            self._spill_callback,
            shared_reference_count=shared_reference_count,
            similarity_threshold=self._fact_similarity_threshold,
            similarity_window=self._fact_similarity_window,
        )
        self._state.research_depth_score += added
        self._sync_to_shared_memory_pool()
        self._maybe_consolidate()

    def _touch_fact(self, fact: Fact, access_increment: int = 1) -> None:
        fact.last_accessed = _now_iso()
        fact.access_count = max(0, int(fact.access_count or 0)) + max(0, access_increment)
        self._recalculate_fact_importance(fact)

    def _recalculate_fact_importance(self, fact: Fact) -> int:
        base = max(1, min(10, int(fact.base_importance or fact.importance or 5)))
        access_count = max(0, int(fact.access_count or 0))
        shared_refs = max(0, int(fact.shared_reference_count or 0))
        last_accessed = _parse_datetime(fact.last_accessed)
        days_since_access = 0.0
        if last_accessed is not None:
            days_since_access = max(
                0.0,
                (datetime.now() - last_accessed).total_seconds() / 86400.0,
            )

        usage_bonus = min(2.0, access_count * 0.35)
        shared_bonus = min(1.5, shared_refs * 0.15)
        recent_use_bonus = 0.0
        if access_count > 0:
            if days_since_access <= 1.0:
                recent_use_bonus = 0.75
            elif days_since_access <= 3.0:
                recent_use_bonus = 0.4

        staleness_penalty = 0.0
        if days_since_access >= 14.0:
            staleness_penalty = 1.0
        elif days_since_access >= 7.0:
            staleness_penalty = 0.5

        updated = round(base + usage_bonus + shared_bonus + recent_use_bonus - staleness_penalty)
        fact.importance = max(1, min(10, updated))
        return fact.importance

    def recalculate_importance_scores(self, *, force: bool = False) -> int:
        now = time.monotonic()
        if (
            not force
            and self._last_importance_refresh
            and (now - self._last_importance_refresh)
            < self._importance_recalc_interval_seconds
        ):
            return 0

        changes = 0
        for fact in self._state.facts_learned:
            before = int(fact.importance or 0)
            after = self._recalculate_fact_importance(fact)
            if before != after:
                changes += 1
        self._last_importance_refresh = now
        return changes

    def adjust_limits(self, usage_factor: float) -> Dict[str, int]:
        normalized_usage = max(0.0, float(usage_factor))
        if self._adaptive_max_facts <= self._base_max_facts:
            self._max_facts = self._base_max_facts
        elif normalized_usage <= 1.0:
            self._max_facts = self._base_max_facts
        else:
            overflow = normalized_usage - 1.0
            steps = max(1, int((overflow / self._adaptive_usage_step) + 0.999999))
            expanded = self._base_max_facts + int(
                steps * self._base_max_facts * self._adaptive_usage_step
            )
            self._max_facts = min(self._adaptive_max_facts, expanded)

        return {
            "max_facts": self._max_facts,
            "base_max_facts": self._base_max_facts,
            "adaptive_max_facts": self._adaptive_max_facts,
        }

    def configure_shared_memory(
        self,
        *,
        shared_memory_pool: Optional[Any] = None,
        scope: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> None:
        if shared_memory_pool is not None:
            self._shared_memory_pool = shared_memory_pool
        if scope is not None:
            self._shared_memory_scope = str(scope).strip() or None
        if agent_name is not None:
            self._shared_memory_agent_name = str(agent_name).strip() or None
        self._sync_to_shared_memory_pool()

    def _sync_to_shared_memory_pool(self) -> None:
        if self._shared_memory_pool is None or not self._shared_memory_scope:
            return
        try:
            self._shared_memory_pool.sync_memory_state(
                self._shared_memory_scope,
                self.to_dict(),
                agent_name=self._shared_memory_agent_name or "working-memory",
            )
        except Exception as exc:
            logger.debug("Shared memory sync skipped: %s", exc)

    def refresh_from_shared_pool(
        self,
        query: str,
        *,
        top_k: int = 5,
        importance: int = 7,
        tags: Optional[List[str]] = None,
        milestone: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self._shared_memory_pool is None:
            return []

        normalized_query = str(query or "").strip()
        if not normalized_query:
            return []

        try:
            hits = self._shared_memory_pool.search_facts(
                normalized_query,
                scope=self._shared_memory_scope,
                top_k=max(1, int(top_k)),
            )
        except Exception as exc:
            logger.debug("Shared memory refresh skipped: %s", exc)
            return []

        if not hits:
            return []

        base_tags = list(tags or ["shared_pool", "context_refresh"])
        before_count = len(self._state.facts_learned)
        entries: List[Dict[str, Any]] = []
        for hit in hits:
            timestamp = str(hit.get("last_updated") or hit.get("created_at") or "").strip()
            fact_text = str(hit.get("text") or "").strip()
            if not fact_text:
                continue
            prefix = f"[Shared:{hit.get('scope') or self._shared_memory_scope}@{timestamp}]"
            hit_tags = [
                str(tag).strip()
                for tag in hit.get("tags", [])
                if str(tag).strip()
            ]
            self.add_facts(
                [f"{prefix} {fact_text}"],
                importance=max(
                    importance,
                    int(hit.get("importance", importance) or importance),
                ),
                tags=[*base_tags, *hit_tags],
                provenance={
                    "evidence_type": "shared_memory_pool",
                    "source_id": str(hit.get("id") or "shared-memory").strip(),
                    "scope": str(
                        hit.get("scope") or self._shared_memory_scope or ""
                    ).strip(),
                    "shared_score": hit.get("shared_score"),
                    "timestamp": timestamp or datetime.now().isoformat(),
                },
                shared_reference_count=int(hit.get("reference_count", 0) or 0),
            )
            entries.append(hit)

        added = len(self._state.facts_learned) - before_count
        if entries and added > 0:
            self.update(
                research_milestones=[
                    milestone or f"Shared pool context refresh for query: {normalized_query}"
                ],
                source_summary=f"Shared memory refresh: loaded {added} findings for '{normalized_query}'",
            )

        return entries

    # ------------------------------------------------------------------
    # Batch update
    # ------------------------------------------------------------------

    def update(
        self,
        new_facts: Optional[Iterable[str]] = None,
        new_questions: Optional[Iterable[str]] = None,
        contradictions: Optional[Iterable[str]] = None,
        research_milestones: Optional[Iterable[str]] = None,
        rejected_hypotheses: Optional[Iterable[str]] = None,
        evidence_records: Optional[Iterable[Dict[str, Any]]] = None,
        source_summary: Optional[str] = None,
        fact_importance: int = 5,
        fact_shared_reference_count: int = 0,
        question_importance: int = 5,
        contradiction_importance: int = 7,
        fact_tags: Optional[List[str]] = None,
        fact_pinned: bool = False,
        fact_provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        if new_facts:
            normalized_facts = [str(item).strip() for item in new_facts if str(item).strip()]
            if normalized_facts:
                projected_usage = (
                    len(self._state.facts_learned) + len(normalized_facts)
                ) / max(1, self._base_max_facts)
                self.adjust_limits(projected_usage)
            added = _add_facts_capped(
                self._state.facts_learned,
                self._seen_facts,
                normalized_facts,
                self._max_facts,
                fact_importance,
                fact_tags or [],
                fact_pinned,
                fact_provenance,
                self._spill_callback,
                shared_reference_count=fact_shared_reference_count,
                similarity_threshold=self._fact_similarity_threshold,
                similarity_window=self._fact_similarity_window,
            )
            self._state.research_depth_score += added
            self._maybe_consolidate()

        if new_questions:
            _add_unique_priority_capped(
                self._state.unanswered_questions,
                self._seen_questions,
                self._question_importance,
                new_questions,
                self._max_questions,
                question_importance,
            )

        if contradictions:
            added = _add_unique_priority_capped(
                self._state.contradictions_found,
                self._seen_contradictions,
                self._contradiction_importance,
                contradictions,
                self._max_contradictions,
                contradiction_importance,
            )
            self._state.research_depth_score += added * 2

        if research_milestones:
            added = _add_unique_capped(
                self._state.research_milestones,
                self._seen_milestones,
                research_milestones,
                self._max_facts,
            )
            self._state.research_depth_score += added

        if rejected_hypotheses:
            _add_unique_capped(
                self._state.rejected_hypotheses,
                self._seen_rejected_hypotheses,
                rejected_hypotheses,
                self._max_facts,
            )

        if evidence_records:
            _add_evidence_capped(
                self._state.evidence_trail,
                self._seen_evidence,
                evidence_records,
                self._max_evidence_records,
            )

        if source_summary:
            _add_unique_capped(
                self._state.sources_consulted,
                self._seen_sources,
                [source_summary],
                self._max_sources,
            )
            self._state.research_depth_score += 1

        if (
            new_facts
            or new_questions
            or contradictions
            or research_milestones
            or rejected_hypotheses
            or evidence_records
            or source_summary
        ):
            self._sync_to_shared_memory_pool()

    # ------------------------------------------------------------------
    # Question resolution
    # ------------------------------------------------------------------

    def resolve_questions(self, answered: Iterable[str]) -> int:
        """Remove answered questions from the unanswered list.
        Keeps the list current and focused; encourages AI to generate meaningful questions.
        Returns the number of questions removed.
        """
        removed = 0
        for q in answered:
            key = _normalize_text(str(q))
            if key in self._seen_questions:
                self._state.unanswered_questions = [
                    x
                    for x in self._state.unanswered_questions
                    if _normalize_text(x) != key
                ]
                self._seen_questions.discard(key)
                removed += 1
        return removed

    # ------------------------------------------------------------------
    # Update from LLM tool call arguments
    # ------------------------------------------------------------------

    def update_from_args(self, args: Dict[str, Any]) -> None:
        normalized_args = dict(args or {})
        importance = _normalize_importance(
            normalized_args.get("fact_importance", 5), 5
        )
        question_importance = _safe_int(
            normalized_args.get("question_importance", 5), 5
        )
        contradiction_importance = _safe_int(
            normalized_args.get("contradiction_importance", 7), 7
        )
        tags = _ensure_list(normalized_args.get("fact_tags")) or None
        fact_pinned = _to_bool(normalized_args.get("fact_pinned", False))
        fact_provenance = _clean_provenance(normalized_args.get("fact_provenance", {}))
        derived_sources: List[str] = []

        structured_facts = normalized_args.get("new_facts")
        if isinstance(structured_facts, dict):
            if any(
                key in structured_facts
                for key in ("text", "fact", "source", "date", "provenance")
            ):
                structured_facts = [structured_facts]
            else:
                structured_facts = [
                    {
                        "text": f"{str(key).strip()}: {json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value).strip()}"
                    }
                    for key, value in structured_facts.items()
                    if str(key).strip()
                    and (
                        (isinstance(value, (dict, list)) and str(json.dumps(value, ensure_ascii=False)).strip())
                        or str(value).strip()
                    )
                ]
            normalized_args["new_facts"] = structured_facts
        if isinstance(structured_facts, list) and structured_facts and any(
            isinstance(item, dict) for item in structured_facts
        ):
            for item in structured_facts:
                if isinstance(item, dict):
                    fact_text = str(item.get("text") or item.get("fact", "")).strip()
                    if not fact_text:
                        continue
                    item_importance = _normalize_importance(
                        item.get("importance", item.get("fact_importance", importance))
                        or importance,
                        importance,
                    )
                    item_tags = (
                        _ensure_list(item.get("tags"))
                        or _ensure_list(item.get("fact_tags"))
                        or tags
                        or []
                    )
                    item_provenance = _legacy_fact_provenance(item) or fact_provenance
                    source_label = str(
                        item.get("source") or item_provenance.get("source_id") or ""
                    ).strip()
                    source_date = str(
                        item.get("date") or item_provenance.get("timestamp") or ""
                    ).strip()
                    if source_label:
                        derived_sources.append(
                            f"{source_label} ({source_date})"
                            if source_date
                            else source_label
                        )
                    self.add_facts(
                        [fact_text],
                        importance=item_importance,
                        tags=item_tags,
                        pinned=_to_bool(item.get("pinned", fact_pinned)),
                        provenance=item_provenance,
                    )
                else:
                    fact_text = str(item).strip()
                    if not fact_text:
                        continue
                    self.add_facts(
                        [fact_text],
                        importance=importance,
                        tags=tags,
                        pinned=fact_pinned,
                        provenance=fact_provenance,
                    )

        self.update(
            new_facts=(
                None
                if isinstance(structured_facts, list)
                and structured_facts
                and any(isinstance(item, dict) for item in structured_facts)
                else _ensure_list(normalized_args.get("new_facts"))
            ),
            new_questions=_ensure_list(normalized_args.get("new_questions")),
            contradictions=_ensure_list(normalized_args.get("contradictions")),
            research_milestones=_ensure_list(
                normalized_args.get("research_milestones")
            ),
            rejected_hypotheses=_ensure_list(
                normalized_args.get("rejected_hypotheses")
            ),
            source_summary=(
                str(normalized_args.get("source_summary")).strip()
                if normalized_args.get("source_summary")
                else ("; ".join(dict.fromkeys(derived_sources))[:500] if derived_sources else None)
            ),
            fact_importance=importance,
            question_importance=question_importance,
            contradiction_importance=contradiction_importance,
            fact_tags=tags,
            fact_pinned=fact_pinned,
            fact_provenance=fact_provenance,
        )
        resolved = _ensure_list(normalized_args.get("resolve_questions"))
        if resolved:
            self.resolve_questions(resolved)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_facts(self, tag: Optional[str] = None) -> List[str]:
        """Return a list of fact texts.

        Args:
            tag: If specified, return only facts with this tag.
                 Case insensitive.
        """
        self.recalculate_importance_scores()
        if tag is None:
            selected = list(self._state.facts_learned)
        else:
            tag_lower = tag.strip().lower()
            selected = [
                fact
                for fact in self._state.facts_learned
                if tag_lower in [t.lower() for t in fact.tags]
            ]
        for fact in selected:
            self._touch_fact(fact)
        return [fact.text for fact in selected]

    # Short alias for get_facts() — common in LLM prompts
    def get_fact_texts(self, tag: Optional[str] = None) -> List[str]:
        """Alias for get_facts(); short name for LLM prompt integration."""
        return self.get_facts(tag=tag)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        self.recalculate_importance_scores()
        return {
            "facts_learned": [f.to_dict() for f in self._state.facts_learned],
            "sources_consulted": list(self._state.sources_consulted),
            "unanswered_questions": list(self._state.unanswered_questions),
            "contradictions_found": list(self._state.contradictions_found),
            "research_milestones": list(self._state.research_milestones),
            "rejected_hypotheses": list(self._state.rejected_hypotheses),
            "evidence_trail": [dict(item) for item in self._state.evidence_trail],
            "question_importance": dict(self._question_importance),
            "contradiction_importance": dict(self._contradiction_importance),
            "research_depth_score": self._state.research_depth_score,
            "limits": {
                "base_max_facts": self._base_max_facts,
                "max_facts": self._max_facts,
                "adaptive_max_facts": self._adaptive_max_facts,
                "adaptive_usage_step": self._adaptive_usage_step,
                "fact_similarity_threshold": self._fact_similarity_threshold,
                "fact_similarity_window": self._fact_similarity_window,
                "consolidation_similarity_ratio": self._consolidation_similarity_ratio,
                "consolidation_cache_size": self._consolidation_cache_size,
                "importance_recalc_interval_seconds": self._importance_recalc_interval_seconds,
            },
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        limits = data.get("limits", {})
        if isinstance(limits, dict):
            self._base_max_facts = max(
                5,
                _safe_int(limits.get("base_max_facts", self._base_max_facts), self._base_max_facts),
            )
            self._max_facts = max(
                self._base_max_facts,
                _safe_int(limits.get("max_facts", self._max_facts), self._max_facts),
            )
            self._adaptive_max_facts = max(
                self._base_max_facts,
                _safe_int(
                    limits.get("adaptive_max_facts", self._adaptive_max_facts),
                    self._adaptive_max_facts,
                ),
            )
            self._adaptive_usage_step = min(
                max(
                    float(limits.get("adaptive_usage_step", self._adaptive_usage_step)),
                    0.05,
                ),
                1.0,
            )
            self._fact_similarity_threshold = min(
                max(
                    float(
                        limits.get(
                            "fact_similarity_threshold",
                            self._fact_similarity_threshold,
                        )
                    ),
                    0.0,
                ),
                1.0,
            )
            self._fact_similarity_window = max(
                2,
                _safe_int(
                    limits.get("fact_similarity_window", self._fact_similarity_window),
                    self._fact_similarity_window,
                ),
            )
            self._consolidation_similarity_ratio = min(
                max(
                    float(
                        limits.get(
                            "consolidation_similarity_ratio",
                            self._consolidation_similarity_ratio,
                        )
                    ),
                    0.0,
                ),
                1.0,
            )
            self._consolidation_cache_size = max(
                4,
                _safe_int(
                    limits.get(
                        "consolidation_cache_size", self._consolidation_cache_size
                    ),
                    self._consolidation_cache_size,
                ),
            )
            self._importance_recalc_interval_seconds = max(
                5,
                _safe_int(
                    limits.get(
                        "importance_recalc_interval_seconds",
                        self._importance_recalc_interval_seconds,
                    ),
                    self._importance_recalc_interval_seconds,
                ),
            )

        raw_facts = data.get("facts_learned", [])
        # Backward compatibility: convert old plain string lists to Fact objects
        if raw_facts and isinstance(raw_facts[0], str):
            self._state.facts_learned = [Fact(text=s) for s in raw_facts]
        else:
            self._state.facts_learned = [Fact.from_dict(f) for f in raw_facts]
        self._state.sources_consulted = list(data.get("sources_consulted", []))
        self._state.unanswered_questions = list(data.get("unanswered_questions", []))
        self._state.contradictions_found = list(data.get("contradictions_found", []))
        self._state.research_milestones = list(data.get("research_milestones", []))
        self._state.rejected_hypotheses = list(data.get("rejected_hypotheses", []))
        self._state.evidence_trail = [
            _clean_evidence_record(item) for item in data.get("evidence_trail", [])
        ]
        self._state.evidence_trail = [
            item for item in self._state.evidence_trail if item
        ]
        self._state.research_depth_score = int(data.get("research_depth_score", 0))
        self._seen_facts = {_normalize_text(f.text) for f in self._state.facts_learned}
        self._seen_questions = {
            _normalize_text(q) for q in self._state.unanswered_questions
        }
        self._seen_contradictions = {
            _normalize_text(c) for c in self._state.contradictions_found
        }
        self._seen_sources = {_normalize_text(s) for s in self._state.sources_consulted}
        self._seen_milestones = {
            _normalize_text(m) for m in self._state.research_milestones
        }
        self._seen_rejected_hypotheses = {
            _normalize_text(h) for h in self._state.rejected_hypotheses
        }
        self._seen_evidence = {
            _evidence_key(item) for item in self._state.evidence_trail if item
        }

        raw_q_importance = data.get("question_importance", {})
        if isinstance(raw_q_importance, dict):
            self._question_importance = {}
            for k, v in raw_q_importance.items():
                try:
                    self._question_importance[_normalize_text(str(k))] = max(1, min(10, int(float(v))))
                except (TypeError, ValueError):
                    self._question_importance[_normalize_text(str(k))] = 5
        else:
            self._question_importance = {
                _normalize_text(q): 5 for q in self._state.unanswered_questions
            }

        raw_c_importance = data.get("contradiction_importance", {})
        if isinstance(raw_c_importance, dict):
            self._contradiction_importance = {}
            for k, v in raw_c_importance.items():
                try:
                    self._contradiction_importance[_normalize_text(str(k))] = max(1, min(10, int(float(v))))
                except (TypeError, ValueError):
                    self._contradiction_importance[_normalize_text(str(k))] = 5
        else:
            self._contradiction_importance = {
                _normalize_text(c): 7 for c in self._state.contradictions_found
            }

    def save_snapshot(self, path: Path) -> None:
        """Persist current working memory state to a JSON snapshot file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def load_snapshot(self, path: Path) -> bool:
        """Load working memory state from a JSON snapshot file.

        Returns True on success, False if the file is missing, unreadable, or malformed.
        """
        target = Path(path)
        if not target.exists():
            return False
        try:
            with open(target, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("load_snapshot: could not read %s: %s", target, exc)
            return False
        if not isinstance(raw, dict):
            return False
        working_memory = raw.get("working_memory") if "working_memory" in raw else raw
        if not isinstance(working_memory, dict):
            return False
        try:
            self.from_dict(working_memory)
        except Exception as exc:
            logger.warning("load_snapshot: from_dict failed for %s: %s", target, exc)
            return False
        return True

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def needs_facts_consolidation(self) -> bool:
        """Return whether memory needs consolidation.
        Returns True if facts_learned exceeds the configured consolidation threshold.
        """
        self.recalculate_importance_scores()
        unpinned_facts = [fact for fact in self._state.facts_learned if not fact.pinned]
        if len(unpinned_facts) >= self._consolidation_threshold:
            return True

        if not unpinned_facts:
            return False

        density = _similarity_density(
            unpinned_facts,
            similarity_threshold=self._fact_similarity_threshold,
            window=self._fact_similarity_window,
        )
        return density >= self._consolidation_similarity_ratio

    def replace_facts(
        self,
        consolidated: List[Any],
        importance: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Replace all current facts with LLM-compressed ones.
        The _seen_facts set is also rebuilt; no duplicates are created.
        If importance is None, average importance from existing facts is calculated and preserved.
        """
        preserved_importance = importance
        if preserved_importance is None:
            existing = self._state.facts_learned
            if existing:
                avg = round(sum(f.importance for f in existing) / len(existing))
                preserved_importance = max(1, min(10, avg))
            else:
                preserved_importance = 5

        existing = list(self._state.facts_learned)
        if self._spill_callback:
            spillable = [
                fact for fact in existing if not fact.pinned and fact.text.strip()
            ]
            if spillable:
                try:
                    self._spill_callback(spillable)
                except Exception as exc:
                    logger.warning("Spill callback error (continuing): %s", exc)

        pinned_survivors = [fact for fact in existing if fact.pinned]
        records: List[Fact] = []
        for item in consolidated:
            if isinstance(item, Fact):
                text = str(item.text or "").strip()
                if not text:
                    continue
                records.append(
                    Fact(
                        text=text,
                        importance=max(1, min(10, int(item.importance or preserved_importance))),
                        base_importance=max(1, min(10, int(item.base_importance or item.importance or preserved_importance))),
                        tags=list(item.tags or tags or []),
                        pinned=bool(item.pinned),
                        provenance=_clean_provenance(item.provenance),
                        created_at=str(item.created_at or _now_iso()),
                        last_accessed=str(item.last_accessed or item.created_at or _now_iso()),
                        access_count=max(0, int(item.access_count or 0)),
                        shared_reference_count=max(0, int(item.shared_reference_count or 0)),
                    )
                )
                continue

            if isinstance(item, dict):
                text = str(item.get("text", "") or "").strip()
                if not text:
                    continue
                item_importance = _safe_int(
                    item.get("importance", preserved_importance),
                    preserved_importance,
                )
                item_tags = _ensure_list(item.get("tags")) or list(tags or [])
                records.append(
                    Fact(
                        text=text,
                        importance=max(1, min(10, item_importance)),
                        base_importance=max(
                            1,
                            min(
                                10,
                                _safe_int(
                                    item.get("base_importance", item_importance),
                                    item_importance,
                                ),
                            ),
                        ),
                        tags=item_tags,
                        pinned=_to_bool(item.get("pinned", False)),
                        provenance=_clean_provenance(item.get("provenance", {})),
                        created_at=str(item.get("created_at") or _now_iso()),
                        last_accessed=str(
                            item.get("last_accessed") or item.get("created_at") or _now_iso()
                        ),
                        access_count=max(0, _safe_int(item.get("access_count", 0), 0)),
                        shared_reference_count=max(
                            0,
                            _safe_int(item.get("shared_reference_count", 0), 0),
                        ),
                    )
                )
                continue

            text = str(item).strip()
            if not text:
                continue
            records.append(
                Fact(
                    text=text,
                    importance=preserved_importance,
                    base_importance=preserved_importance,
                    tags=list(tags or []),
                    pinned=False,
                    provenance={},
                )
            )

        self._state.facts_learned = []
        self._seen_facts = set()
        self._consolidating = False

        for fact in records:
            _add_facts_capped(
                self._state.facts_learned,
                self._seen_facts,
                [fact.text],
                self._max_facts,
                fact.importance,
                fact.tags,
                fact.pinned,
                fact.provenance,
                self._spill_callback,
                similarity_threshold=self._fact_similarity_threshold,
                similarity_window=self._fact_similarity_window,
            )

        for fact in pinned_survivors:
            self._state.facts_learned.append(fact)
            self._seen_facts.add(_normalize_text(fact.text))
        self._sync_to_shared_memory_pool()

    def _build_consolidation_cache_key(self, facts: Sequence[Fact]) -> str:
        serialized = [
            {
                "text": _normalize_text(fact.text),
                "importance": int(fact.importance),
                "tags": [str(tag).strip().lower() for tag in fact.tags if str(tag).strip()],
            }
            for fact in facts
            if fact.text.strip()
        ]
        return json.dumps(serialized, ensure_ascii=False, sort_keys=True)

    def _read_consolidation_cache(self, cache_key: str) -> Optional[List[str]]:
        cached = self._consolidation_cache.get(cache_key)
        if cached is None:
            return None
        self._consolidation_cache.move_to_end(cache_key)
        return list(cached)

    def _write_consolidation_cache(self, cache_key: str, values: Sequence[str]) -> None:
        normalized = [str(item).strip() for item in values if str(item).strip()]
        if not normalized:
            return
        self._consolidation_cache[cache_key] = list(normalized)
        self._consolidation_cache.move_to_end(cache_key)
        while len(self._consolidation_cache) > self._consolidation_cache_size:
            self._consolidation_cache.popitem(last=False)

    def _cluster_similar_facts(self, facts: Sequence[Fact]) -> List[List[Fact]]:
        clusters: List[List[Fact]] = []
        threshold = max(0.55, min(0.95, self._fact_similarity_threshold - 0.1))
        for fact in facts:
            placed = False
            for cluster in clusters:
                if any(
                    _fact_similarity_score(member.text, fact.text) >= threshold
                    for member in cluster
                ):
                    cluster.append(fact)
                    placed = True
                    break
            if not placed:
                clusters.append([fact])
        return clusters

    def _extractive_cluster_summary(self, cluster: Sequence[Fact]) -> str:
        ordered_cluster = sorted(
            cluster,
            key=lambda fact: (fact.importance, len(fact.text), fact.text.lower()),
            reverse=True,
        )
        representative = ordered_cluster[0].text.strip()
        if len(ordered_cluster) == 1:
            return representative

        representative_tokens = _tokenize_text(representative)
        representative_numbers = _extract_numeric_tokens(representative)
        additions: List[str] = []
        for fact in ordered_cluster[1:]:
            candidate = fact.text.strip()
            if not candidate:
                continue
            candidate_tokens = _tokenize_text(candidate)
            candidate_numbers = _extract_numeric_tokens(candidate)
            novel_numbers = candidate_numbers - representative_numbers
            novel_tokens = candidate_tokens - representative_tokens
            if novel_numbers or len(novel_tokens) >= 3:
                additions.append(candidate)
                representative_tokens |= candidate_tokens
                representative_numbers |= candidate_numbers

        dense_parts = _unique_texts(
            [representative, *additions],
            similarity_threshold=max(0.9, self._fact_similarity_threshold + 0.08),
        )
        return " | ".join(dense_parts)

    def _fallback_consolidate_group(self, facts: Sequence[Fact]) -> List[str]:
        clusters = self._cluster_similar_facts(facts)
        condensed = [
            self._extractive_cluster_summary(cluster)
            for cluster in clusters
            if cluster
        ]
        return _unique_texts(
            condensed,
            similarity_threshold=max(0.92, self._fact_similarity_threshold + 0.1),
        )

    async def _consolidate_group(self, group_facts: Sequence[Fact]) -> List[str]:
        cache_key = self._build_consolidation_cache_key(group_facts)
        cached = self._read_consolidation_cache(cache_key)
        if cached is not None:
            return cached

        texts = [fact.text for fact in group_facts if fact.text.strip()]
        consolidated: List[str] = []
        if self._consolidation_callback:
            callback_result = await self._consolidation_callback(texts)  # type: ignore[misc]
            consolidated = [
                str(item).strip()
                for item in (callback_result or [])
                if str(item).strip()
            ]

        if not consolidated:
            consolidated = self._fallback_consolidate_group(group_facts)
        if not consolidated:
            consolidated = texts

        self._write_consolidation_cache(cache_key, consolidated)
        return consolidated

    def refresh_context(
        self,
        query: str,
        *,
        rag_service: Any,
        collection: Optional[str] = None,
        symbol_filter: Optional[str] = None,
        top_k: int = 5,
        recent_days: Optional[int] = None,
        context_window: int = 1,
        importance: int = 7,
        tags: Optional[List[str]] = None,
        preview_chars: int = 800,
        milestone: Optional[str] = None,
        query_hypothesis: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        normalized_query = str(query or "").strip()
        if not normalized_query or rag_service is None:
            return []

        try:
            hits = rag_service.search(
                query=normalized_query,
                collection=collection,
                symbol_filter=symbol_filter,
                top_k=max(1, int(top_k)),
                recent_days=recent_days,
                context_window=0,
                query_hypothesis=query_hypothesis,
            )
        except Exception as exc:
            logger.debug("refresh_context skipped: RAG search failed: %s", exc)
            return []

        if not hits:
            return []

        selected_hits = list(hits)
        hit_ids = [
            f"{str(hit.get('collection') or '').strip()}:{str(hit.get('id') or '').strip()}"
            for hit in hits[: max(1, int(top_k))]
            if str(hit.get("collection") or "").strip()
            and str(hit.get("id") or "").strip()
        ]
        if hit_ids and hasattr(rag_service, "fetch_hits"):
            try:
                hydrated = rag_service.fetch_hits(
                    hit_ids, context_window=max(0, int(context_window))
                )
                if hydrated:
                    selected_hits = hydrated
            except Exception as exc:
                logger.debug("refresh_context hydration skipped: %s", exc)

        refreshed_entries: List[Dict[str, Any]] = []
        before_count = len(self._state.facts_learned)
        base_tags = list(tags or ["rag_retrieved", "context_refresh"])

        for hit in selected_hits[: max(1, int(top_k))]:
            metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
            content = _truncate_preview(str(hit.get("content") or ""), preview_chars)
            if not content:
                continue
            doc_type = str(metadata.get("doc_type") or hit.get("collection") or "memory")
            timestamp = str(metadata.get("timestamp") or "").strip()
            quality_score = hit.get("quality_adjusted_score")
            if not isinstance(quality_score, (int, float)):
                quality_score = hit.get("normalized_score")
            retrieval_score = round(float(quality_score or 0.0), 4)
            collection_name = str(hit.get("collection") or "").strip()
            doc_id = str(hit.get("id") or "").strip()
            source_id = (
                f"{collection_name}:{doc_id}"
                if collection_name and doc_id
                else str(metadata.get("source_id") or "rag")
            )
            fact_text = f"[RAG:{doc_type}@{timestamp}] {content}" if timestamp else f"[RAG:{doc_type}] {content}"
            provenance = {
                "evidence_type": "rag",
                "source_id": source_id,
                "doc_group_id": str(metadata.get("doc_group_id") or "").strip(),
                "chunk_index": metadata.get("chunk_index"),
                "retrieval_score": retrieval_score,
                "timestamp": timestamp,
            }
            self.add_facts(
                [fact_text],
                importance=importance,
                tags=base_tags,
                provenance=provenance,
            )
            refreshed_entries.append({"text": fact_text, "provenance": provenance})

        added = len(self._state.facts_learned) - before_count
        if refreshed_entries and added > 0:
            self.update(
                research_milestones=[
                    milestone or f"Context refreshed from RAG for query: {normalized_query}"
                ],
                source_summary=f"RAG context refresh: loaded {added} findings for '{normalized_query}'",
            )

        return refreshed_entries

    @staticmethod
    def _primary_tag(fact: Fact) -> str:
        for tag in fact.tags:
            normalized = str(tag).strip().lower()
            if normalized:
                return normalized
        return "untagged"

    def _maybe_consolidate(self) -> None:
        """Trigger async callback in background when consolidation threshold is exceeded.

        Conditions:
        - consolidation_callback must be defined.
        - Threshold must be exceeded (needs_facts_consolidation() == True).
        - Another consolidation must not already be running (_consolidating flag).
        - A running asyncio event loop must be present.
        """
        if (
            not self._consolidation_callback
            or not self.needs_facts_consolidation()
            or self._consolidating
        ):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # Sync context or no loop — skip
        self._consolidating = True
        self._consolidation_task = loop.create_task(self._run_consolidation())

    async def _run_consolidation(self) -> None:
        """Run consolidation callback and call replace_facts with results."""
        try:
            unpinned_facts = [
                fact
                for fact in self._state.facts_learned
                if not fact.pinned and fact.text.strip()
            ]
            if not unpinned_facts:
                return

            grouped: Dict[str, List[Fact]] = {}
            for fact in unpinned_facts:
                key = self._primary_tag(fact)
                grouped.setdefault(key, []).append(fact)

            consolidated_records: List[Fact] = []
            for group_facts in grouped.values():
                normalized = await self._consolidate_group(group_facts)
                if not normalized:
                    normalized = [fact.text for fact in group_facts if fact.text.strip()]

                avg_importance = round(
                    sum(f.importance for f in group_facts) / len(group_facts)
                )
                avg_importance = max(1, min(10, avg_importance))
                merged_tags = sorted(
                    {
                        str(tag).strip()
                        for fact in group_facts
                        for tag in fact.tags
                        if str(tag).strip()
                    }
                )

                consolidated_records.extend(
                    Fact(
                        text=item,
                        importance=avg_importance,
                        tags=merged_tags,
                        pinned=False,
                    )
                    for item in normalized
                )

            self.replace_facts(
                consolidated_records,
                importance=None,
                tags=None,
            )
        except Exception as exc:
            logger.warning("Consolidation callback error (continuing): %s", exc)
        finally:
            self._consolidating = False

    async def flush(self, timeout_seconds: float = 15.0) -> None:
        """Wait for any in-flight consolidation task to complete before finalizing output."""
        task = self._consolidation_task
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(task), timeout=max(0.1, float(timeout_seconds))
            )
        except asyncio.TimeoutError:
            logger.warning(
                "WorkingMemory flush timed out; continuing with current state."
            )
        except Exception as exc:
            logger.warning(
                "WorkingMemory flush failed; continuing with current state: %s", exc
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def set_research_depth_score(self, score: int) -> None:
        self._state.research_depth_score = max(0, int(score))

    def summary_counts(self) -> Dict[str, int]:
        return {
            "facts": len(self._state.facts_learned),
            "sources": len(self._state.sources_consulted),
            "questions": len(self._state.unanswered_questions),
            "contradictions": len(self._state.contradictions_found),
            "milestones": len(self._state.research_milestones),
            "rejected_hypotheses": len(self._state.rejected_hypotheses),
            "evidence": len(self._state.evidence_trail),
            "score": self._state.research_depth_score,
        }

    def add_evidence_records(self, records: Iterable[Dict[str, Any]]) -> None:
        _add_evidence_capped(
            self._state.evidence_trail,
            self._seen_evidence,
            records,
            self._max_evidence_records,
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _add_facts_capped(
    target: List[Fact],
    seen: set[str],
    items: Iterable[str],
    max_items: int,
    importance: int,
    tags: List[str],
    pinned: bool,
    provenance: Optional[Dict[str, Any]],
    spill_callback: Optional[SpillCallback] = None,
    *,
    shared_reference_count: int = 0,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    similarity_window: int = DEFAULT_SIMILARITY_WINDOW,
) -> int:
    """Add facts as Fact objects; prevent duplicates.

    When capacity (max_items) is exceeded, the fact with the lowest importance score
    is deleted instead of FIFO. If there's a tie, the oldest is preferred.
    Returns the number of items added.
    """
    added = 0
    clamped_importance = max(1, min(10, importance))
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        key = _normalize_text(text)
        if key in seen:
            for fact in reversed(target):
                if _normalize_text(fact.text) == key:
                    _merge_fact_details(
                        fact,
                        importance=clamped_importance,
                        shared_reference_count=shared_reference_count,
                        tags=tags,
                        pinned=pinned,
                        provenance=provenance,
                    )
                    break
            continue
        similar_fact = _find_similar_fact(
            target,
            text,
            similarity_threshold=similarity_threshold,
            similarity_window=similarity_window,
        )
        if similar_fact is not None:
            _merge_fact_details(
                similar_fact,
                importance=clamped_importance,
                shared_reference_count=shared_reference_count,
                tags=tags,
                pinned=pinned,
                provenance=provenance,
            )
            continue
        target.append(
            Fact(
                text=text,
                importance=clamped_importance,
                base_importance=clamped_importance,
                tags=list(tags),
                pinned=bool(pinned),
                provenance=_clean_provenance(provenance),
                created_at=_now_iso(),
                last_accessed=_now_iso(),
                access_count=0,
                shared_reference_count=max(0, int(shared_reference_count or 0)),
            )
        )
        seen.add(key)
        added += 1
        if len(target) > max_items:
            # Pinned facts are preserved as much as possible.
            unpinned_indices = [i for i, fact in enumerate(target) if not fact.pinned]
            candidate_indices = (
                unpinned_indices if unpinned_indices else list(range(len(target)))
            )
            min_idx = min(candidate_indices, key=lambda i: target[i].importance)
            evicted = target.pop(min_idx)
            seen.discard(_normalize_text(evicted.text))
            if spill_callback and evicted.text.strip():
                try:
                    spill_callback([evicted])
                except Exception as exc:
                    logger.warning("Spill callback error during eviction (continuing): %s", exc)
    return added


def _add_unique_capped(
    target: List[str],
    seen: set[str],
    items: Iterable[str],
    max_items: int,
) -> int:
    """Simple FIFO add for sources, questions, and contradictions.
    Add without duplicates; delete oldest entry if capacity is exceeded.
    Returns the number of items added.
    """
    added = 0
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        key = _normalize_text(text)
        if key in seen:
            continue
        target.append(text)
        seen.add(key)
        added += 1
        if len(target) > max_items:
            evicted = target.pop(0)
            seen.discard(_normalize_text(evicted))
    return added


def _add_unique_priority_capped(
    target: List[str],
    seen: set[str],
    importance_map: Dict[str, int],
    items: Iterable[str],
    max_items: int,
    importance: int,
) -> int:
    """Add unique texts without duplicates; delete lowest-priority record when capacity is full.

    When priority is equal, the oldest in the list is deleted.
    """
    added = 0
    clamped_importance = max(1, min(10, int(importance)))

    for item in items:
        text = str(item).strip()
        if not text:
            continue

        key = _normalize_text(text)
        if key in seen:
            importance_map[key] = max(
                importance_map.get(key, clamped_importance), clamped_importance
            )
            continue

        target.append(text)
        seen.add(key)
        importance_map[key] = clamped_importance
        added += 1

        if len(target) > max_items:
            min_idx = min(
                range(len(target)),
                key=lambda i: importance_map.get(_normalize_text(target[i]), 5),
            )
            evicted = target.pop(min_idx)
            evicted_key = _normalize_text(evicted)
            seen.discard(evicted_key)
            importance_map.pop(evicted_key, None)

    return added


def _add_evidence_capped(
    target: List[Dict[str, Any]],
    seen: set[str],
    items: Iterable[Dict[str, Any]],
    max_items: int,
) -> int:
    added = 0
    for item in items:
        record = _clean_evidence_record(item)
        if not record:
            continue
        key = _evidence_key(record)
        if not key or key in seen:
            continue
        target.append(record)
        seen.add(key)
        added += 1
        if len(target) > max_items:
            evicted = target.pop(0)
            seen.discard(_evidence_key(evicted))
    return added
