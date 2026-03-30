"""Unified in-memory cache manager.

Centralizes TTL cache instances and exposes lightweight health statistics.
"""

from __future__ import annotations

import copy
import threading
from datetime import datetime, timezone
from typing import Any, Dict

from cachetools import TTLCache

from core import get_standard_logger

logger = get_standard_logger(__name__)


class UnifiedCacheManager:
    """Singleton-style registry for process-local TTL caches."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._caches: Dict[str, TTLCache] = {}
        self._stats: Dict[str, Dict[str, Any]] = {}

    def get_ttl_cache(self, namespace: str, maxsize: int, ttl_seconds: int) -> TTLCache:
        safe_namespace = str(namespace).strip().lower()
        if not safe_namespace:
            raise ValueError("namespace is required")

        safe_maxsize = max(1, int(maxsize))
        safe_ttl = max(1, int(ttl_seconds))

        with self._lock:
            existing = self._caches.get(safe_namespace)
            if existing is not None:
                if int(getattr(existing, "maxsize", 0)) == safe_maxsize and int(
                    getattr(existing, "ttl", 0)
                ) == safe_ttl:
                    return existing

            cache = TTLCache(maxsize=safe_maxsize, ttl=safe_ttl)
            self._caches[safe_namespace] = cache
            self._stats.setdefault(
                safe_namespace,
                {
                    "hits": 0,
                    "misses": 0,
                    "sets": 0,
                    "invalidations": 0,
                    "last_access_utc": None,
                    "last_set_utc": None,
                    "maxsize": safe_maxsize,
                    "ttl_seconds": safe_ttl,
                },
            )
            self._stats[safe_namespace]["maxsize"] = safe_maxsize
            self._stats[safe_namespace]["ttl_seconds"] = safe_ttl
            logger.debug(
                "Unified cache initialized: namespace=%s maxsize=%d ttl=%ds",
                safe_namespace,
                safe_maxsize,
                safe_ttl,
            )
            return cache

    def get(self, namespace: str, key: Any) -> Any:
        safe_ns = str(namespace).strip().lower()
        with self._lock:
            cache = self._caches.get(safe_ns)
            if cache is None:
                return None
            value = cache.get(key)
            stats = self._stats.setdefault(safe_ns, {})
            stats["last_access_utc"] = datetime.now(timezone.utc).isoformat()
            if value is None:
                stats["misses"] = int(stats.get("misses", 0)) + 1
            else:
                stats["hits"] = int(stats.get("hits", 0)) + 1
        return value

    def get_many(self, namespace: str, keys: list[Any]) -> Dict[Any, Any]:
        safe_ns = str(namespace).strip().lower()
        if not keys:
            return {}

        found: Dict[Any, Any] = {}
        with self._lock:
            cache = self._caches.get(safe_ns)
            if cache is None:
                return {}

            stats = self._stats.setdefault(safe_ns, {})
            stats["last_access_utc"] = datetime.now(timezone.utc).isoformat()

            for key in keys:
                value = cache.get(key)
                if value is None:
                    stats["misses"] = int(stats.get("misses", 0)) + 1
                    continue
                stats["hits"] = int(stats.get("hits", 0)) + 1
                found[key] = value

        return found

    def set(self, namespace: str, key: Any, value: Any) -> None:
        safe_namespace = str(namespace).strip().lower()
        with self._lock:
            cache = self._caches.get(safe_namespace)
            if cache is None:
                raise KeyError(f"Cache namespace not initialized: {safe_namespace}")
            cache[key] = value
            stats = self._stats.setdefault(safe_namespace, {})
            stats["sets"] = int(stats.get("sets", 0)) + 1
            stats["last_set_utc"] = datetime.now(timezone.utc).isoformat()

    def set_many(self, namespace: str, items: Dict[Any, Any]) -> None:
        safe_namespace = str(namespace).strip().lower()
        if not items:
            return

        with self._lock:
            cache = self._caches.get(safe_namespace)
            if cache is None:
                raise KeyError(f"Cache namespace not initialized: {safe_namespace}")

            for key, value in items.items():
                cache[key] = value

            stats = self._stats.setdefault(safe_namespace, {})
            stats["sets"] = int(stats.get("sets", 0)) + len(items)
            stats["last_set_utc"] = datetime.now(timezone.utc).isoformat()

    def invalidate_namespace(self, namespace: str) -> None:
        safe_namespace = str(namespace).strip().lower()
        with self._lock:
            cache = self._caches.get(safe_namespace)
            if cache is not None:
                cache.clear()
            stats = self._stats.setdefault(safe_namespace, {})
            stats["invalidations"] = int(stats.get("invalidations", 0)) + 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            by_namespace = copy.deepcopy(self._stats)

        for namespace, cache in self._caches.items():
            by_namespace.setdefault(namespace, {})["items"] = len(cache)

        total_hits = sum(int(v.get("hits", 0)) for v in by_namespace.values())
        total_misses = sum(int(v.get("misses", 0)) for v in by_namespace.values())
        total_requests = total_hits + total_misses

        return {
            "summary": {
                "namespaces": len(by_namespace),
                "total_hits": total_hits,
                "total_misses": total_misses,
                "hit_rate": round(total_hits / total_requests, 4) if total_requests else 0.0,
            },
            "by_namespace": by_namespace,
        }


_cache_manager = UnifiedCacheManager()


def get_unified_cache() -> UnifiedCacheManager:
    return _cache_manager
