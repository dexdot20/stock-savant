from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from core.paths import ensure_parent_dir, get_instance_dir

logger = logging.getLogger(__name__)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _tokenize_text(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9_.%:-]+", _normalize_text(value)))


def _similarity_score(left: Any, right: Any) -> float:
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


def _safe_scope(scope: Any) -> str:
    text = str(scope or "global").strip().lower()
    return text or "global"


def _fact_id(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()[:24]


class SharedMemoryPool:
    """File-backed fact pool for cross-agent sharing."""

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        *,
        similarity_threshold: float = 0.6,
    ) -> None:
        self._storage_path = ensure_parent_dir(
            storage_path or (get_instance_dir() / "shared_memory" / "pool.json")
        )
        self._similarity_threshold = min(max(float(similarity_threshold), 0.0), 1.0)
        self._lock = threading.RLock()
        self._state: Dict[str, Any] = {"scopes": {}}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if self._storage_path.exists():
                try:
                    with open(self._storage_path, "r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    if isinstance(payload, dict):
                        self._state = payload
                except Exception as exc:
                    logger.warning("Shared memory pool load failed: %s", exc)
                    self._state = {"scopes": {}}
            self._state.setdefault("scopes", {})
            self._loaded = True

    def _persist(self) -> None:
        payload = dict(self._state)
        payload["last_updated"] = datetime.now().isoformat()
        with open(self._storage_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)

    def _ensure_scope(self, scope: str) -> Dict[str, Any]:
        scopes = self._state.setdefault("scopes", {})
        scope_key = _safe_scope(scope)
        if scope_key not in scopes or not isinstance(scopes.get(scope_key), dict):
            scopes[scope_key] = {
                "facts": [],
                "sources": [],
                "contradictions": [],
                "last_updated": datetime.now().isoformat(),
            }
        bucket = scopes[scope_key]
        bucket.setdefault("facts", [])
        bucket.setdefault("sources", [])
        bucket.setdefault("contradictions", [])
        return bucket

    @staticmethod
    def _clean_fact(item: Any) -> Optional[Dict[str, Any]]:
        if isinstance(item, dict):
            text = str(item.get("text", "") or "").strip()
            importance = int(item.get("importance", 5) or 5)
            tags_raw = item.get("tags", [])
            tags = (
                [str(tag).strip() for tag in tags_raw if str(tag).strip()]
                if isinstance(tags_raw, list)
                else []
            )
            provenance = (
                item.get("provenance", {})
                if isinstance(item.get("provenance"), dict)
                else {}
            )
            pinned = bool(item.get("pinned", False))
        else:
            text = str(getattr(item, "text", item) or "").strip()
            importance = int(getattr(item, "importance", 5) or 5)
            tags = [
                str(tag).strip()
                for tag in getattr(item, "tags", [])
                if str(tag).strip()
            ]
            provenance = (
                getattr(item, "provenance", {})
                if isinstance(getattr(item, "provenance", {}), dict)
                else {}
            )
            pinned = bool(getattr(item, "pinned", False))

        if not text:
            return None

        return {
            "id": _fact_id(text),
            "text": text,
            "importance": max(1, min(10, importance)),
            "tags": tags,
            "pinned": pinned,
            "provenance": provenance,
        }

    @staticmethod
    def _merge_agent_names(existing: list[str], agent_name: str) -> list[str]:
        ordered = list(existing or [])
        if agent_name and agent_name not in ordered:
            ordered.append(agent_name)
        return ordered

    def sync_memory_state(
        self,
        scope: str,
        memory_state: Dict[str, Any],
        *,
        agent_name: str,
    ) -> int:
        self._load()
        payload = memory_state if isinstance(memory_state, dict) else {}
        facts = (
            payload.get("facts_learned", [])
            if isinstance(payload.get("facts_learned", []), list)
            else []
        )
        sources = (
            payload.get("sources_consulted", [])
            if isinstance(payload.get("sources_consulted", []), list)
            else []
        )
        contradictions = (
            payload.get("contradictions_found", [])
            if isinstance(payload.get("contradictions_found", []), list)
            else []
        )

        changed = 0
        with self._lock:
            bucket = self._ensure_scope(scope)
            now_iso = datetime.now().isoformat()

            for raw_fact in facts:
                fact = self._clean_fact(raw_fact)
                if fact is None:
                    continue
                existing = next(
                    (
                        item
                        for item in bucket["facts"]
                        if _normalize_text(item.get("text"))
                        == _normalize_text(fact["text"])
                    ),
                    None,
                )
                if existing is None:
                    fact["created_at"] = now_iso
                    fact["last_updated"] = now_iso
                    fact["agent_names"] = [agent_name] if agent_name else []
                    fact["reference_count"] = 1
                    bucket["facts"].append(fact)
                    changed += 1
                    continue

                existing["importance"] = max(
                    int(existing.get("importance", 5) or 5), fact["importance"]
                )
                existing["tags"] = list(
                    dict.fromkeys([*(existing.get("tags") or []), *fact["tags"]])
                )
                existing["pinned"] = bool(
                    existing.get("pinned", False) or fact["pinned"]
                )
                existing["provenance"] = {
                    **(existing.get("provenance") or {}),
                    **fact["provenance"],
                }
                existing["agent_names"] = self._merge_agent_names(
                    existing.get("agent_names") or [], agent_name
                )
                existing["reference_count"] = (
                    int(existing.get("reference_count", 0) or 0) + 1
                )
                existing["last_updated"] = now_iso
                changed += 1

            for collection_name, items in (
                ("sources", sources),
                ("contradictions", contradictions),
            ):
                for raw_item in items:
                    text = str(raw_item or "").strip()
                    if not text:
                        continue
                    existing = next(
                        (
                            item
                            for item in bucket[collection_name]
                            if _normalize_text(item.get("text")) == _normalize_text(text)
                        ),
                        None,
                    )
                    if existing is None:
                        bucket[collection_name].append(
                            {
                                "text": text,
                                "agent_names": [agent_name] if agent_name else [],
                                "reference_count": 1,
                                "created_at": now_iso,
                                "last_updated": now_iso,
                            }
                        )
                        changed += 1
                        continue

                    existing["agent_names"] = self._merge_agent_names(
                        existing.get("agent_names") or [], agent_name
                    )
                    existing["reference_count"] = (
                        int(existing.get("reference_count", 0) or 0) + 1
                    )
                    existing["last_updated"] = now_iso
                    changed += 1

            if changed:
                bucket["last_updated"] = now_iso
                self._persist()

        return changed

    def search_facts(
        self,
        query: str,
        *,
        scope: Optional[str] = None,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> list[Dict[str, Any]]:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return []

        self._load()
        scope_filter = _safe_scope(scope) if scope else None
        effective_min_score = (
            max(0.05, self._similarity_threshold * 0.25)
            if min_score is None
            else max(0.0, float(min_score))
        )
        results: list[Dict[str, Any]] = []

        with self._lock:
            scopes = (
                self._state.get("scopes", {})
                if isinstance(self._state.get("scopes"), dict)
                else {}
            )
            for scope_name, bucket in scopes.items():
                if scope_filter and scope_name != scope_filter:
                    continue
                facts = bucket.get("facts", []) if isinstance(bucket, dict) else []
                for fact in facts:
                    text = str(fact.get("text") or "").strip()
                    if not text:
                        continue
                    similarity = _similarity_score(normalized_query, text)
                    if similarity < effective_min_score:
                        continue
                    importance = max(
                        1, min(10, int(fact.get("importance", 5) or 5))
                    )
                    reference_bonus = min(
                        0.2, int(fact.get("reference_count", 0) or 0) * 0.02
                    )
                    score = similarity + (importance / 20.0) + reference_bonus
                    results.append(
                        {
                            **fact,
                            "scope": scope_name,
                            "shared_score": round(score, 4),
                            "similarity": round(similarity, 4),
                        }
                    )

        results.sort(
            key=lambda item: (
                -float(item.get("shared_score", 0.0) or 0.0),
                -int(item.get("importance", 0) or 0),
                str(item.get("text") or "").lower(),
            )
        )
        return results[: max(1, int(top_k))]

    def get_scope_snapshot(self, scope: str) -> Dict[str, Any]:
        self._load()
        with self._lock:
            bucket = self._ensure_scope(scope)
            return json.loads(json.dumps(bucket, ensure_ascii=False))


__all__ = ["SharedMemoryPool"]