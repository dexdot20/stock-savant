import asyncio
import copy
import hashlib
import httpx
import math
import json
import os
import random
import re
import sys
import tempfile
import threading
import textwrap
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Type
from uuid import uuid4

from config import get_config
from core import get_standard_logger
from core.cache_manager import get_unified_cache
from core.paths import (
    get_ai_reports_path,
    get_python_exec_dir,
    get_tool_health_path,
)
from news_scraper.search_engine import GoogleNewsSearchEngine
from news_scraper.text_preprocessor import split_text_semantically
from news_scraper.content_extractor import ContentExtractor
from services.factories import get_financial_service, get_rag_service
from services.kap import (
    KapLookupError,
    batch_get_disclosure_details,
    get_disclosure_detail,
    search_disclosures,
)
from services.tool_schemas import (
    FetchUrlContentResult,
    GoogleNewsResult,
    KapBatchDetailsResult,
    KapDisclosureDetailResult,
    KapSearchResult,
    PythonExecResult,
    ReportToolResult,
    ToolExecutionResult,
)
from services.network.client import ProxyManager, PROXY_STEP_NEWS_SEARCH
from rich.console import Console
from pydantic import BaseModel

logger = get_standard_logger(__name__)

from domain.utils import safe_int_strict as _safe_int


# Monkeypatch DDGS http_client to fix unsupported browser impersonate options.
# ddgs and primp versions can drift, causing warnings like:
# "Impersonate 'safari_15.3' does not exist, using 'random'".
# We force DDGS to always use "random", which primp supports consistently.
def _patch_ddgs_http_client():
    try:
        from ddgs import http_client

        if hasattr(http_client, "HttpClient"):
            http_client.HttpClient._impersonates = ("random",)
    except (ImportError, AttributeError) as exc:
        logger.debug("DDGS http_client patch skipped: %s", exc)


def _ddgs_first_value(item: Any, *keys: str) -> str:
    if not isinstance(item, dict):
        return ""

    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return str(value)

    return ""


_patch_ddgs_http_client()

_ddgs_lock = asyncio.Lock()
_ddgs_last_call: float = 0.0
_search_cache_lock = asyncio.Lock()
_summary_cache_lock = asyncio.Lock()
_tool_metrics_lock = threading.Lock()
_tool_init_lock = threading.Lock()
_tool_health_lock = threading.Lock()
_report_lock = threading.Lock()
_recent_report_fingerprints: List[str] = []

_cache_cfg = get_config().get("cache", {})
_cache_manager = get_unified_cache()

_analysis_cache_cfg = _cache_cfg.get("tools_web_search", {})
_cache_manager.get_ttl_cache(
    namespace="tools_web_search",
    maxsize=_safe_int(_analysis_cache_cfg.get("max_entries", 256), 256),
    ttl_seconds=_safe_int(_analysis_cache_cfg.get("ttl_seconds", 1800), 1800),
)

_news_cache_cfg = _cache_cfg.get("tools_news_search", {})
_cache_manager.get_ttl_cache(
    namespace="tools_news_search",
    maxsize=_safe_int(_news_cache_cfg.get("max_entries", 256), 256),
    ttl_seconds=_safe_int(_news_cache_cfg.get("ttl_seconds", 900), 900),
)

_summary_cache_cfg = _cache_cfg.get("tools_summary", {})
_cache_manager.get_ttl_cache(
    namespace="tools_summary",
    maxsize=_safe_int(_summary_cache_cfg.get("max_entries", 128), 128),
    ttl_seconds=_safe_int(_summary_cache_cfg.get("ttl_seconds", 3600), 3600),
)

# --- Module-level cached config slices (avoid deep-copy on every call) ---
_full_cfg = get_config()
_health_check_cfg: dict = _full_cfg.get("network", {}).get("health_check", {})
_ai_tool_output_cfg: dict = _full_cfg.get("ai", {}).get("tool_output", {})
del _full_cfg  # don't keep a full copy in memory

# Tool quality metrics – declared here so functions defined below can reference it
_tool_quality_metrics: Dict[str, Dict[str, Any]] = {}


def _safe_json_size(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError, OverflowError):
        return len(str(value))


def _infer_result_count(result: Any) -> Optional[int]:
    if isinstance(result, list):
        return len(result)
    if not isinstance(result, dict):
        return None

    for key in ("results", "quotes", "news"):
        value = result.get(key)
        if isinstance(value, list):
            return len(value)

    for key in ("web", "news"):
        value = result.get(key)
        if isinstance(value, dict) and isinstance(value.get("results"), list):
            return len(value.get("results"))

    return None


def _is_error_result(result: Any) -> bool:
    return isinstance(result, dict) and bool(result.get("error"))


def _is_empty_result(result: Any) -> bool:
    count = _infer_result_count(result)
    if count is not None:
        return count == 0
    if isinstance(result, dict):
        return len(result) == 0
    if isinstance(result, list):
        return len(result) == 0
    return False


def _record_tool_metrics(tool_name: str, result: Any, duration_ms: int) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    result_size = _safe_json_size(result)
    result_count = _infer_result_count(result)
    error = _is_error_result(result)
    empty = _is_empty_result(result)
    truncated = isinstance(result, dict) and bool(result.get("truncated"))

    with _tool_metrics_lock:
        metrics = _tool_quality_metrics.setdefault(
            tool_name,
            {
                "calls": 0,
                "success": 0,
                "errors": 0,
                "empty_results": 0,
                "truncated_results": 0,
                "total_duration_ms": 0,
                "avg_duration_ms": 0,
                "last_duration_ms": 0,
                "total_result_bytes": 0,
                "avg_result_bytes": 0,
                "last_result_bytes": 0,
                "max_result_bytes": 0,
                "last_result_count": None,
                "last_error": None,
                "last_called_at": None,
            },
        )

        metrics["calls"] += 1
        metrics["success"] += 0 if error else 1
        metrics["errors"] += 1 if error else 0
        metrics["empty_results"] += 1 if empty else 0
        metrics["truncated_results"] += 1 if truncated else 0
        metrics["total_duration_ms"] += duration_ms
        metrics["avg_duration_ms"] = int(
            metrics["total_duration_ms"] / max(1, metrics["calls"])
        )
        metrics["last_duration_ms"] = duration_ms
        metrics["total_result_bytes"] += result_size
        metrics["avg_result_bytes"] = int(
            metrics["total_result_bytes"] / max(1, metrics["calls"])
        )
        metrics["last_result_bytes"] = result_size
        metrics["max_result_bytes"] = max(metrics["max_result_bytes"], result_size)
        metrics["last_result_count"] = result_count
        metrics["last_error"] = str(result.get("error"))[:300] if error else None
        metrics["last_called_at"] = now_iso


def get_tool_quality_metrics() -> Dict[str, Any]:
    with _tool_metrics_lock:
        by_tool = copy.deepcopy(_tool_quality_metrics)

    total_calls = sum(v.get("calls", 0) for v in by_tool.values())
    total_errors = sum(v.get("errors", 0) for v in by_tool.values())
    total_success = sum(v.get("success", 0) for v in by_tool.values())

    return {
        "summary": {
            "tools_tracked": len(by_tool),
            "total_calls": total_calls,
            "total_success": total_success,
            "total_errors": total_errors,
            "error_rate": (
                round((total_errors / total_calls), 4) if total_calls else 0.0
            ),
        },
        "by_tool": by_tool,
    }


def reset_tool_quality_metrics(tool_name: Optional[str] = None) -> Dict[str, Any]:
    with _tool_metrics_lock:
        if tool_name:
            _tool_quality_metrics.pop(str(tool_name), None)
        else:
            _tool_quality_metrics.clear()
    return {"status": "ok", "tool_name": tool_name or "all"}


_tool_health_metrics: Dict[str, Dict[str, Any]] = {}


def _persist_tool_health_metrics() -> None:
    try:
        health_path = get_tool_health_path()
        with open(health_path, "w", encoding="utf-8") as handle:
            json.dump(_tool_health_metrics, handle, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.debug("Tool health persistence skipped: %s", exc)


def _restore_tool_health_metrics() -> None:
    global _tool_health_metrics
    try:
        health_path = get_tool_health_path()
        if not health_path.exists():
            return
        with open(health_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            _tool_health_metrics = payload
    except Exception as exc:
        logger.debug("Tool health restore skipped: %s", exc)


_restore_tool_health_metrics()


_TOOL_TOKEN_COST_HINTS: Dict[str, str] = {
    "search_web": "low",
    "search_news": "low",
    "search_google_news": "low",
    "get_tool_quality_metrics": "low",
    "get_tool_health_metrics": "low",
    "report": "low",
    "search_memory": "low",
    "python_exec": "medium",
    "summarize_url_content": "medium",
    "fetch_url_content": "high",
    "yfinance_search": "low",
    "yfinance_overview": "medium",
    "yfinance_index_data": "medium",
    "yfinance_price_history": "medium",
    "yfinance_dividends": "medium",
    "yfinance_analyst": "medium",
    "yfinance_earnings": "medium",
    "yfinance_ownership": "medium",
    "yfinance_sustainability": "medium",
    "yfinance_sector_analysis": "medium",
    "yfinance_company_data": "high",
    "kap_search_disclosures": "low",
    "kap_get_disclosure_detail": "medium",
    "kap_batch_disclosure_details": "medium",
}


def to_standard_tool_result(tool_name: str, result: Any) -> Dict[str, Any]:
    payload = result if isinstance(result, dict) else {"result": result}
    error = payload.get("error") if isinstance(payload, dict) else None
    error_code = payload.get("error_code") if isinstance(payload, dict) else None
    token_cost_hint = _TOOL_TOKEN_COST_HINTS.get(tool_name, "medium")
    is_ephemeral = tool_name in {"fetch_url_content", "summarize_url_content"}
    data = None if error else payload

    envelope = ToolExecutionResult(
        tool_name=str(tool_name),
        success=not bool(error),
        data=data,
        error=(str(error) if error else None),
        error_code=(str(error_code) if error_code else None),
        data_format="json",
        is_ephemeral=is_ephemeral,
        token_cost_hint=token_cost_hint,
    )
    return envelope.model_dump(exclude_none=True)


def _record_tool_health(
    tool_name: str,
    *,
    success: bool,
    duration_ms: int,
    error_code: Optional[str] = None,
) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    now_ts = time.time()
    cfg = _health_check_cfg
    degraded_after = max(1, _safe_int(cfg.get("degraded_after_failures", 3), 3))
    unhealthy_after = max(
        degraded_after + 1,
        _safe_int(cfg.get("unhealthy_after_failures", 6), 6),
    )
    cooldown_seconds = max(0, _safe_int(cfg.get("tool_cooldown_seconds", 300), 300))

    with _tool_health_lock:
        item = _tool_health_metrics.setdefault(
            tool_name,
            {
                "status": "healthy",
                "calls": 0,
                "success": 0,
                "failures": 0,
                "consecutive_failures": 0,
                "last_error_code": None,
                "last_duration_ms": 0,
                "last_called_at": None,
                "cooldown_until_ts": None,
            },
        )
        item["calls"] += 1
        item["last_duration_ms"] = duration_ms
        item["last_called_at"] = now_iso

        if success:
            item["success"] += 1
            item["consecutive_failures"] = 0
            item["last_error_code"] = None
            item["cooldown_until_ts"] = None
        else:
            item["failures"] += 1
            item["consecutive_failures"] += 1
            item["last_error_code"] = str(error_code or "unknown_error")

        if item["consecutive_failures"] >= unhealthy_after:
            item["status"] = "unhealthy"
            if cooldown_seconds > 0:
                item["cooldown_until_ts"] = now_ts + cooldown_seconds
        elif item["consecutive_failures"] >= degraded_after:
            item["status"] = "degraded"
        else:
            item["status"] = "healthy"

        _persist_tool_health_metrics()


def get_tool_health_metrics() -> Dict[str, Any]:
    with _tool_health_lock:
        by_tool = copy.deepcopy(_tool_health_metrics)

    unhealthy = sum(1 for v in by_tool.values() if v.get("status") == "unhealthy")
    degraded = sum(1 for v in by_tool.values() if v.get("status") == "degraded")
    cooling_down = sum(
        1
        for v in by_tool.values()
        if isinstance(v.get("cooldown_until_ts"), (int, float))
        and float(v.get("cooldown_until_ts")) > time.time()
    )

    return {
        "summary": {
            "tools_tracked": len(by_tool),
            "unhealthy_tools": unhealthy,
            "degraded_tools": degraded,
            "cooling_down_tools": cooling_down,
        },
        "by_tool": by_tool,
    }


def _precheck_tool_health(tool_name: str) -> Optional[Dict[str, Any]]:
    with _tool_health_lock:
        item = _tool_health_metrics.get(tool_name)
        if not isinstance(item, dict):
            return None

        cooldown_until_ts = item.get("cooldown_until_ts")
        status = str(item.get("status") or "healthy")
        now_ts = time.time()

        if status != "unhealthy":
            return None

        if isinstance(cooldown_until_ts, (int, float)) and float(cooldown_until_ts) > now_ts:
            retry_after = int(math.ceil(float(cooldown_until_ts) - now_ts))
            return {
                "error": f"Tool '{tool_name}' is temporarily disabled due to repeated failures.",
                "error_code": "tool_temporarily_disabled",
                "tool": tool_name,
                "status": "cooldown",
                "retry_after": retry_after,
            }

        item["status"] = "degraded"
        item["cooldown_until_ts"] = None
        _persist_tool_health_metrics()
        return None


def _is_retryable_tool_error(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if not result.get("error"):
        return False

    error_code = str(result.get("error_code") or "").lower()
    non_retryable_codes = {
        "validation_error",
        "tool_not_found",
        "serialization_error",
        "empty_result",
    }
    if error_code in non_retryable_codes:
        return False

    message = str(result.get("error") or "").lower()
    non_retryable_tokens = (
        "must provide either",
        "invalid argument",
        "validation failed",
        "missing required",
        "unexpected keyword argument",
        "required positional argument",
        "got multiple values for argument",
        " is required",
    )
    if any(t in message for t in non_retryable_tokens):
        return False

    retryable_codes = {
        "execution_error",
        "tool_error",
        "timeout",
        "rate_limit",
        "network_error",
        "connection_error",
    }
    if error_code in retryable_codes:
        return True

    tokens = ("timeout", "timed out", "temporar", "rate limit", "connection", "reset")
    return any(t in message for t in tokens)


def _resolve_tool_output_limit(
    tool_name: str, fallback: Optional[int]
) -> Optional[int]:
    tool_output_cfg = _ai_tool_output_cfg

    if isinstance(tool_output_cfg, dict):
        if not bool(tool_output_cfg.get("enabled", True)):
            return None
        per_tool = tool_output_cfg.get("per_tool", {}) or {}
        raw_limit = per_tool.get(
            tool_name, tool_output_cfg.get("default_max_chars", fallback)
        )
    else:
        raw_limit = fallback or 6000

    if raw_limit in (None, False, "off", "OFF", "none", "NONE"):
        return None

    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        limit = int(fallback or 6000)

    if limit <= 0:
        return None
    return limit


async def _throttle_ddgs_requests(min_interval_seconds: float) -> None:
    global _ddgs_last_call
    async with _ddgs_lock:
        now = time.time()
        wait_time = max(0.0, (_ddgs_last_call + min_interval_seconds) - now)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        _ddgs_last_call = time.time()


_REPORT_SECRET_PATTERNS = [
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(token\s*[:=]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(secret\s*[:=]\s*)([^\s,;]+)"),
    re.compile(r"\bsk-[A-Za-z0-9]{12,}\b"),
]


def _truncate_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def _redact_sensitive_text(text: Any) -> str:
    redacted = str(text or "")
    for pattern in _REPORT_SECRET_PATTERNS:
        redacted = pattern.sub(lambda match: f"{match.group(1)}[REDACTED]" if match.lastindex and match.lastindex > 1 else "[REDACTED]", redacted)
    return redacted


def _sanitize_report_value(value: Any, *, max_chars: int, max_items: int, redact: bool) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        sanitized: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                sanitized["_truncated"] = True
                break
            sanitized[str(key)] = _sanitize_report_value(
                item,
                max_chars=max_chars,
                max_items=max_items,
                redact=redact,
            )
        return sanitized
    if isinstance(value, list):
        items = [
            _sanitize_report_value(
                item,
                max_chars=max_chars,
                max_items=max_items,
                redact=redact,
            )
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            items.append({"_truncated": True})
        return items
    if isinstance(value, (str, int, float, bool)):
        text = str(value)
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    if redact:
        text = _redact_sensitive_text(text)
    return _truncate_text(text, max_chars)


def _build_report_fingerprint(title: str, category: str, summary: str) -> str:
    basis = "|".join(
        [
            str(title or "").strip().lower(),
            str(category or "").strip().lower(),
            str(summary or "").strip().lower(),
        ]
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]


def _build_python_runner_script() -> str:
    return textwrap.dedent(
        r'''
        import builtins
        import contextlib
        import io
        import json
        import os
        import sys
        import time
        import traceback

        def _to_int(v, d=0):
            try:
                return int(v)
            except (TypeError, ValueError):
                return int(d)

        payload_path = sys.argv[1]
        with open(payload_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        code = str(payload.get("code") or "")
        input_data = payload.get("input_data")
        max_output_chars = _to_int(payload.get("max_output_chars", 12000), 12000)
        blocked_modules = {str(name).strip() for name in payload.get("blocked_modules", []) if str(name).strip()}
        sandbox_root = os.path.realpath(os.getcwd())
        file_size_limit = _to_int(payload.get("max_file_size_bytes", 2097152), 2097152)

        def emit(obj):
            sys.__stdout__.write(json.dumps(obj, ensure_ascii=False, default=str))
            sys.__stdout__.flush()

        def trim(text):
            text = str(text or "")
            if max_output_chars <= 0 or len(text) <= max_output_chars:
                return text
            return text[:max_output_chars] + "…"

        try:
            import resource

            max_memory_mb = _to_int(payload.get("max_memory_mb", 256), 256)
            max_file_bytes = max(1024, file_size_limit)
            resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_bytes, max_file_bytes))
            if max_memory_mb > 0:
                max_mem = max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (max_mem, max_mem))
        except (ImportError, OSError, ValueError):
            pass

        original_import = builtins.__import__
        original_open = builtins.open

        def blocked_callable(*args, **kwargs):
            raise PermissionError("This capability is disabled in python_exec sandbox")

        def safe_path(path_value):
            target = os.path.realpath(os.path.join(sandbox_root, str(path_value)))
            if not target.startswith(sandbox_root + os.sep) and target != sandbox_root:
                raise PermissionError("File access outside sandbox is not allowed")
            return target

        def safe_open(file, mode="r", *args, **kwargs):
            target = safe_path(file)
            if any(flag in str(mode) for flag in ("w", "a", "+", "x")):
                parent = os.path.dirname(target)
                if parent:
                    os.makedirs(parent, exist_ok=True)
            return original_open(target, mode, *args, **kwargs)

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            root = str(name or "").split(".", 1)[0]
            if root in blocked_modules:
                raise ImportError(f"Module '{root}' is blocked in python_exec")
            module = original_import(name, globals, locals, fromlist, level)
            if root == "os":
                for attr in ("system", "popen", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe"):
                    if hasattr(module, attr):
                        setattr(module, attr, blocked_callable)
            return module

        builtins.__import__ = guarded_import
        builtins.open = safe_open
        builtins.input = blocked_callable
        builtins.breakpoint = blocked_callable

        exec_globals = {
            "__name__": "__main__",
            "input_data": input_data,
            "sandbox_dir": sandbox_root,
            "__builtins__": builtins,
        }

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        started = time.perf_counter()

        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                exec(compile(code, "<python_exec>", "exec"), exec_globals, exec_globals)
            result_value = exec_globals.get("result")
            files = []
            for root, _, filenames in os.walk(sandbox_root):
                for filename in filenames:
                    if filename in {"runner.py", "payload.json"}:
                        continue
                    full_path = os.path.join(root, filename)
                    files.append(os.path.relpath(full_path, sandbox_root))
            emit(
                {
                    "ok": True,
                    "stdout": trim(stdout_buffer.getvalue()),
                    "stderr": trim(stderr_buffer.getvalue()),
                    "result": result_value,
                    "result_repr": trim(repr(result_value)),
                    "execution_ms": int((time.perf_counter() - started) * 1000),
                    "files": sorted(files),
                    "sandbox_ephemeral": True,
                }
            )
        except Exception as exc:
            emit(
                {
                    "ok": False,
                    "error": str(exc),
                    "error_code": "execution_error",
                    "stdout": trim(stdout_buffer.getvalue()),
                    "stderr": trim(stderr_buffer.getvalue() + traceback.format_exc()),
                    "execution_ms": int((time.perf_counter() - started) * 1000),
                    "files": [],
                    "sandbox_ephemeral": True,
                }
            )
        '''
    ).strip()


class ToolBase:
    name: str = "base"
    description: str = "base tool"
    console: Optional[Console] = None
    output_model: Optional[Type[BaseModel]] = None
    output_max_chars: Optional[int] = None

    async def execute(self, **kwargs) -> Any:
        raise NotImplementedError

    def _normalize_error(
        self, message: Any, error_code: str = "tool_error", details: Any = None
    ) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {
            "error": str(message),
            "error_code": str(error_code),
            "tool": self.name,
        }
        if details is not None:
            normalized["details"] = details
        return normalized

    def sanitize_output(self, result: Any, max_chars: Optional[int] = None) -> Any:
        if result is None:
            return self._normalize_error(
                "Tool returned no data", error_code="empty_result"
            )

        if isinstance(result, dict) and result.get("error"):
            error_code = str(result.get("error_code") or "tool_error")
            details = result.get("details") if "details" in result else None
            normalized = self._normalize_error(
                result.get("error"), error_code=error_code, details=details
            )
            for passthrough in ("status", "retry_after"):
                if passthrough in result:
                    normalized[passthrough] = result[passthrough]
            return normalized

        if self.output_model:
            try:
                if isinstance(result, self.output_model):
                    result = result.model_dump(exclude_none=True)
                else:
                    result = self.output_model.model_validate(result).model_dump(
                        exclude_none=True
                    )
            except Exception as exc:
                return self._normalize_error(
                    "Tool output validation failed",
                    error_code="validation_error",
                    details=str(exc),
                )

        if isinstance(result, (str, int, float, bool)):
            result = {"result": result}

        try:
            serialized = json.dumps(result, ensure_ascii=False, default=str)
        except Exception:
            return self._normalize_error(
                "Tool output is not serializable", error_code="serialization_error"
            )

        resolved_max_chars = _resolve_tool_output_limit(
            self.name, max_chars or self.output_max_chars or 6000
        )
        if resolved_max_chars is None or len(serialized) <= resolved_max_chars:
            return result

        return {
            "warning": "Tool output truncated",
            "preview": serialized[:resolved_max_chars],
            "truncated": True,
        }


class DDGSSearchTool(ToolBase):
    name = "search_web"
    description = "Search the web for general information using DuckDuckGo."

    def __init__(self):
        self.config = get_config()
        self._proxy_manager = ProxyManager(
            config=self.config.get("proxy", {}),
            step=PROXY_STEP_NEWS_SEARCH,
        )

    async def execute(
        self,
        query: str,
        count: int = 5,
        num_results: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict:
        if num_results is not None:
            try:
                count = int(num_results)
            except (TypeError, ValueError):
                count = int(count)
        min_interval = self.config.get("web_search", {}).get("search_delay_seconds", 2)
        region = self.config.get("web_search", {}).get("ddgs_region", "wt-wt")
        safesearch = self.config.get("web_search", {}).get("ddgs_safesearch", "off")
        cache_key = (query, int(count), str(region), str(safesearch))

        async with _search_cache_lock:
            cached = _cache_manager.get("tools_web_search", cache_key)
        if cached is not None:
            return copy.deepcopy(cached)

        proxy = self._proxy_manager.get_proxy()
        backend = str(kwargs.get("backend") or "auto").strip() or "auto"

        await _throttle_ddgs_requests(float(min_interval))

        def _run_ddgs_text(use_proxy: Optional[str]) -> list:
            from ddgs import DDGS

            with DDGS(proxy=use_proxy) as ddgs:
                return list(
                    ddgs.text(
                        query,
                        region=region,
                        safesearch=safesearch,
                        max_results=count,
                        page=1,
                        backend=backend,
                    )
                )

        def _parse_web_raw(raw: list) -> Dict:
            results = [
                {
                    "title": _ddgs_first_value(r, "title"),
                    "url": _ddgs_first_value(r, "href", "url", "link"),
                    "description": _ddgs_first_value(
                        r, "body", "description", "snippet", "content"
                    ),
                }
                for r in raw
            ]
            return {"web": {"results": results, "query": query}}

        try:
            raw = await asyncio.to_thread(_run_ddgs_text, proxy)
            if proxy:
                self._proxy_manager.mark_proxy_success(proxy)
            response = _parse_web_raw(raw)
            async with _search_cache_lock:
                _cache_manager.set(
                    "tools_web_search", cache_key, copy.deepcopy(response)
                )
            return response
        except Exception as e:
            err_str = str(e).lower()
            is_decode_error = "decode" in err_str or "decodeerror" in err_str
            is_no_results = "no results" in err_str

            if proxy:
                if is_no_results:
                    self._proxy_manager.mark_proxy_success(proxy)
                else:
                    self._proxy_manager.mark_proxy_failed(proxy)

            if is_no_results:
                logger.debug("DDGS Search returned no results for query: %r", query)
                return {"web": {"results": [], "query": query}}

            # DecodeError is typically a proxy encoding issue — retry without proxy.
            if is_decode_error and proxy:
                logger.warning(
                    "DDGS Search DecodeError with proxy, retrying without proxy: %s", e
                )
                await _throttle_ddgs_requests(float(min_interval))
                try:
                    raw = await asyncio.to_thread(_run_ddgs_text, None)
                    response = _parse_web_raw(raw)
                    async with _search_cache_lock:
                        _cache_manager.set(
                            "tools_web_search", cache_key, copy.deepcopy(response)
                        )
                    return response
                except Exception as retry_exc:
                    logger.warning(
                        "DDGS Search retry without proxy also failed: %s", retry_exc
                    )
                    return {"web": {"results": [], "query": query}}

            logger.error("DDGS Search Exception: %s", e)
            return {"web": {"results": [], "query": query}}


class DDGSNewsTool(ToolBase):
    name = "search_news"
    description = "Search for recent news articles using DuckDuckGo News."

    def __init__(self):
        self.config = get_config()
        self._proxy_manager = ProxyManager(
            config=self.config.get("proxy", {}),
            step=PROXY_STEP_NEWS_SEARCH,
        )
        self._google_news_engine = GoogleNewsSearchEngine(self.config)

    @staticmethod
    def _normalize_timelimit(value: Any) -> str:
        raw = str(value or "d").strip().lower()
        if raw in {"d", "w", "m"}:
            return raw
        if raw.endswith("d"):
            return "d"
        if raw.endswith("w"):
            return "w"
        if raw.endswith("m"):
            return "m"
        return "d"

    @staticmethod
    def _simplify_query(query: str) -> str:
        tokens = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9]+", str(query or ""))
        if not tokens:
            return str(query or "").strip()

        stopwords = {
            "mart",
            "nisan",
            "mayıs",
            "haziran",
            "temmuz",
            "ağustos",
            "eylül",
            "ekim",
            "kasım",
            "aralık",
            "ocak",
            "şubat",
            "top",
            "largest",
            "companies",
            "company",
            "seviyesi",
            "list",
        }

        filtered: List[str] = []
        for token in tokens:
            lower = token.lower()
            if lower in stopwords:
                continue
            if token.isdigit() and len(token) == 4:
                continue
            filtered.append(token)

        simplified = " ".join(filtered[:6]).strip()
        return simplified or " ".join(tokens[:6]).strip()

    async def _google_news_fallback(
        self,
        query: str,
        count: int,
        timelimit: str,
    ) -> Dict:
        try:
            results = await self._google_news_engine.fetch_google_news_async(
                query=query,
                max_results=max(1, int(count)),
                lang="tr",
                time_period=timelimit,
            )
        except Exception as exc:
            logger.warning("Google News fallback failed: %s", exc)
            return {"news": {"results": [], "query": query}}

        normalized_results = []
        for row in results:
            if not isinstance(row, dict):
                continue
            normalized_results.append(
                {
                    "title": str(row.get("title") or ""),
                    "url": str(row.get("link") or ""),
                    "description": str(row.get("snippet") or ""),
                    "date": str(row.get("time") or ""),
                    "source": str(row.get("source") or ""),
                }
            )

        return {"news": {"results": normalized_results, "query": query}}

    async def execute(
        self,
        query: str,
        count: int = 10,
        timelimit: str = "d",
        max_results: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict:
        if max_results is not None:
            count = max_results
        timelimit = self._normalize_timelimit(timelimit)
        min_interval = self.config.get("web_search", {}).get("search_delay_seconds", 2)
        region = self.config.get("web_search", {}).get("ddgs_region", "wt-wt")
        safesearch = self.config.get("web_search", {}).get("ddgs_safesearch", "off")
        cache_key = (query, int(count), str(timelimit), str(region), str(safesearch))

        async with _search_cache_lock:
            cached = _cache_manager.get("tools_news_search", cache_key)
        if cached is not None:
            return copy.deepcopy(cached)

        proxy = self._proxy_manager.get_proxy()
        backend = str(kwargs.get("backend") or "auto").strip() or "auto"

        await _throttle_ddgs_requests(float(min_interval))

        def _run_ddgs_news(use_proxy: Optional[str], target_query: str) -> list:
            from ddgs import DDGS

            with DDGS(proxy=use_proxy) as ddgs:
                return list(
                    ddgs.news(
                        target_query,
                        region=region,
                        safesearch=safesearch,
                        timelimit=timelimit,
                        max_results=count,
                        page=1,
                        backend=backend,
                    )
                )

        def _parse_raw(raw: list) -> Dict:
            results = [
                {
                    "title": _ddgs_first_value(r, "title"),
                    "url": _ddgs_first_value(r, "url", "href", "link"),
                    "description": _ddgs_first_value(
                        r, "body", "content", "description", "snippet"
                    ),
                    "date": _ddgs_first_value(r, "date", "published"),
                    "source": _ddgs_first_value(r, "source", "publisher"),
                }
                for r in raw
            ]
            return {"news": {"results": results, "query": query}}

        try:
            raw = await asyncio.to_thread(_run_ddgs_news, proxy, query)
            if proxy:
                self._proxy_manager.mark_proxy_success(proxy)
            response = _parse_raw(raw)
            async with _search_cache_lock:
                _cache_manager.set(
                    "tools_news_search", cache_key, copy.deepcopy(response)
                )
            return response
        except Exception as e:
            err_str = str(e).lower()
            is_decode_error = "decode" in err_str or "decodeerror" in err_str
            is_no_results = "no results" in err_str

            if proxy:
                if is_no_results:
                    self._proxy_manager.mark_proxy_success(proxy)
                else:
                    self._proxy_manager.mark_proxy_failed(proxy)

            if is_no_results:
                logger.debug("DDGS News returned no results for query: %r", query)
                return {"news": {"results": [], "query": query}}

            # DecodeError is typically a proxy encoding issue — retry without proxy.
            if is_decode_error and proxy:
                logger.warning(
                    "DDGS News DecodeError with proxy, retrying without proxy: %s", e
                )
                await _throttle_ddgs_requests(float(min_interval))
                try:
                    raw = await asyncio.to_thread(_run_ddgs_news, None, query)
                    response = _parse_raw(raw)
                    async with _search_cache_lock:
                        _cache_manager.set(
                            "tools_news_search", cache_key, copy.deepcopy(response)
                        )
                    return response
                except Exception as retry_exc:
                    logger.warning(
                        f"DDGS News retry without proxy also failed: {retry_exc}"
                    )

                    simplified_query = self._simplify_query(query)
                    if simplified_query and simplified_query != query:
                        logger.warning(
                            "DDGS News retrying with simplified query: %r -> %r",
                            query,
                            simplified_query,
                        )
                        simplified_cache_key = (
                            simplified_query,
                            int(count),
                            str(timelimit),
                            str(region),
                            str(safesearch),
                        )
                        await _throttle_ddgs_requests(float(min_interval))
                        try:
                            raw = await asyncio.to_thread(
                                _run_ddgs_news,
                                None,
                                simplified_query,
                            )
                            response = _parse_raw(raw)
                            if isinstance(response, dict):
                                response["news"]["query"] = query
                                response["news"]["effective_query"] = simplified_query
                            async with _search_cache_lock:
                                _cache_manager.set(
                                    "tools_news_search",
                                    simplified_cache_key,
                                    copy.deepcopy(response),
                                )
                                _cache_manager.set(
                                    "tools_news_search",
                                    cache_key,
                                    copy.deepcopy(response),
                                )
                            return response
                        except Exception as simplified_exc:
                            logger.warning(
                                "DDGS News simplified query retry also failed: %s",
                                simplified_exc,
                            )

                    response = await self._google_news_fallback(
                        query=query,
                        count=int(count),
                        timelimit=timelimit,
                    )
                    async with _search_cache_lock:
                        _cache_manager.set(
                            "tools_news_search", cache_key, copy.deepcopy(response)
                        )
                    return response

            logger.error("DDGS News Exception: %s", e)
            response = await self._google_news_fallback(
                query=query,
                count=int(count),
                timelimit=timelimit,
            )
            async with _search_cache_lock:
                _cache_manager.set(
                    "tools_news_search", cache_key, copy.deepcopy(response)
                )
            return response


class GoogleNewsTool(ToolBase):
    name = "search_google_news"
    description = "Search specifically on Google News."
    output_model = GoogleNewsResult

    def __init__(self):
        self.engine = GoogleNewsSearchEngine(get_config())

    async def execute(
        self,
        query: str,
        days: int = 1,
        language: str = "en",
        count: Optional[int] = None,
        timelimit: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict]:
        if timelimit:
            tl = str(timelimit).strip().lower()
            if tl == "d":
                days = 1
            elif tl == "w":
                days = 7
            elif tl == "m":
                days = 30
            elif tl == "y":
                days = 365

        time_period = self._map_days_to_time_period(days)
        results = await self.engine.fetch_google_news_async(
            query,
            lang=language,
            time_period=time_period,
        )
        if isinstance(count, int) and count > 0:
            return results[:count]
        return results

    @staticmethod
    def _map_days_to_time_period(days: int) -> Optional[str]:
        if days is None or days <= 0:
            return None
        if days <= 1:
            return "d"
        if days <= 7:
            return "w"
        if days <= 31:
            return "m"
        return "y"


class FetchUrlContentTool(ToolBase):
    """Fetches raw content from URL; NO mini AI summarization.
    Content is sent directly to main model — high token cost, full authenticity."""

    name = "fetch_url_content"
    output_model = FetchUrlContentResult

    def __init__(self):
        config = get_config()
        self.extractor = ContentExtractor(config)

    async def execute(self, url: str, focus_query: Optional[str] = None) -> Dict:
        try:
            res = await self.extractor.extract_article_content_async(url)
            if not res or not res.get("content"):
                return {"error": "Failed to fetch content or content empty"}

            res["is_summarized"] = False
            return res
        except Exception as e:
            return {"error": str(e)}


class SummarizeUrlContentTool(ToolBase):
    """Fetches content from URL and returns retrieval-pruned passages.
    Preferred option that protects token budget while preserving source text."""

    name = "summarize_url_content"
    output_model = FetchUrlContentResult
    _FINANCIAL_MARKERS = (
        "revenue",
        "earnings",
        "ebitda",
        "guidance",
        "margin",
        "forecast",
        "outlook",
        "analyst",
        "target",
        "dividend",
        "capex",
        "debt",
        "cash flow",
        "profit",
        "loss",
        "growth",
        "shares",
        "valuation",
        "disclosure",
        "filing",
        "risk",
        "strategy",
    )

    def __init__(self):
        config = get_config()
        ai_cfg = config.get("ai", {}) if isinstance(config, dict) else {}
        rag_cfg = ai_cfg.get("rag", {}) if isinstance(ai_cfg, dict) else {}
        pruning_cfg = ai_cfg.get("url_content_pruning", {}) if isinstance(ai_cfg, dict) else {}
        self.extractor = ContentExtractor(config)
        self._chunk_size_chars = int(
            pruning_cfg.get("chunk_size_chars", rag_cfg.get("news_chunk_size", 1800))
        )
        self._max_selected_chunks = max(
            1, int(pruning_cfg.get("max_selected_chunks", 3))
        )
        self._max_expanded_chunks = max(
            self._max_selected_chunks,
            int(pruning_cfg.get("max_expanded_chunks", 5)),
        )
        self._lead_chunk_count = max(1, int(pruning_cfg.get("lead_chunk_count", 2)))
        self._min_selected_chars = max(
            200, int(pruning_cfg.get("min_selected_chars", 1200))
        )
        self._min_focus_score = float(pruning_cfg.get("min_focus_score", 2.2))
        self._min_salience_score = float(pruning_cfg.get("min_salience_score", 0.45))
        self._fallback_fulltext_chars = max(
            self._min_selected_chars,
            int(pruning_cfg.get("fallback_fulltext_chars", 12000)),
        )
        self._coverage_ratio_floor = float(
            pruning_cfg.get("coverage_ratio_floor", 0.18)
        )

    @staticmethod
    def _normalize_query_terms(value: Optional[str]) -> List[str]:
        terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9._%/-]{1,}", str(value or "").lower())
        seen = set()
        ordered: List[str] = []
        for term in terms:
            if len(term) < 2 or term in seen:
                continue
            seen.add(term)
            ordered.append(term)
        return ordered

    @staticmethod
    def _numeric_signal_count(text: str) -> int:
        return len(re.findall(r"\b\d[\d,.:/%$EURTRYTLMBKmbk-]*\b", text or ""))

    def _score_chunk(
        self,
        chunk: str,
        index: int,
        total_chunks: int,
        focus_query: Optional[str],
        focus_terms: List[str],
    ) -> float:
        normalized = str(chunk or "").lower()
        score = 0.0

        if index < self._lead_chunk_count:
            score += max(0.0, (self._lead_chunk_count - index) * 0.35)

        numeric_hits = self._numeric_signal_count(chunk)
        score += min(2.0, numeric_hits * 0.08)

        keyword_hits = sum(1 for term in self._FINANCIAL_MARKERS if term in normalized)
        score += min(2.0, keyword_hits * 0.22)

        if re.search(r"(^|\n)(#{1,6}\s|[A-Z][^\n]{0,80}:)", str(chunk or "")):
            score += 0.35

        if total_chunks > 1 and index == total_chunks - 1:
            score += 0.1

        if focus_terms:
            overlap = sum(1 for term in focus_terms if term in normalized)
            density = overlap / max(1, len(focus_terms))
            score += overlap * 1.1
            score += density * 2.0
            if focus_query and str(focus_query).strip().lower() in normalized:
                score += 1.4
            if overlap == 0:
                score -= 0.4

        return score

    def _expand_chunk_indices(
        self,
        primary_indices: List[int],
        total_chunks: int,
        *,
        max_chunks: Optional[int] = None,
    ) -> List[int]:
        if total_chunks <= 0:
            return []

        limit = max_chunks if max_chunks is not None else self._max_expanded_chunks
        limit = max(1, min(limit, total_chunks))
        ordered: List[int] = []
        seen = set()

        for index in sorted(primary_indices):
            if 0 <= index < total_chunks and index not in seen:
                ordered.append(index)
                seen.add(index)
            if len(ordered) >= limit:
                return sorted(ordered)

        distance = 1
        while len(ordered) < limit:
            added = False
            for index in sorted(primary_indices):
                for candidate in (index - distance, index + distance):
                    if candidate < 0 or candidate >= total_chunks or candidate in seen:
                        continue
                    ordered.append(candidate)
                    seen.add(candidate)
                    added = True
                    if len(ordered) >= limit:
                        return sorted(ordered)
            if not added:
                break
            distance += 1

        return sorted(ordered)

    def _format_pruned_content(
        self,
        *,
        content: str,
        source_length: int,
        selection_stats: Dict[str, Any],
    ) -> str:
        coverage_ratio = float(selection_stats.get("coverage_ratio", 0.0)) * 100.0
        return (
            "RAG_PRUNED_CONTENT\n"
            f"Source length: {source_length} chars\n"
            f"Selection mode: {selection_stats.get('mode', 'unknown')}\n"
            f"Selected chunks: {selection_stats.get('selected_chunk_count', 0)}/{selection_stats.get('total_chunks', 0)}\n"
            f"Coverage: {coverage_ratio:.1f}%\n\n{content}"
        )

    def _prune_content(
        self,
        *,
        content: str,
        focus_query: Optional[str],
    ) -> Dict[str, Any]:
        normalized_content = str(content or "").strip()
        if not normalized_content:
            return {
                "content": "",
                "index_content": "",
                "selection_stats": {
                    "mode": "empty",
                    "total_chunks": 0,
                    "selected_chunk_count": 0,
                    "selected_chars": 0,
                    "coverage_ratio": 0.0,
                    "focus_query": str(focus_query or "").strip(),
                },
            }

        if len(normalized_content) <= self._fallback_fulltext_chars:
            return {
                "content": normalized_content,
                "index_content": normalized_content,
                "selection_stats": {
                    "mode": "full_short_content",
                    "total_chunks": 1,
                    "selected_chunk_count": 1,
                    "selected_chars": len(normalized_content),
                    "coverage_ratio": 1.0,
                    "focus_query": str(focus_query or "").strip(),
                },
            }

        chunks = split_text_semantically(
            normalized_content,
            max_chunk_size=self._chunk_size_chars,
        )
        chunks = [chunk.strip() for chunk in (chunks or [normalized_content]) if chunk and chunk.strip()]
        if not chunks:
            chunks = [normalized_content]

        stripped_focus_query = str(focus_query or "").strip()
        focus_terms = self._normalize_query_terms(stripped_focus_query)

        scored_chunks: List[Dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            scored_chunks.append(
                {
                    "index": index,
                    "score": self._score_chunk(
                        chunk,
                        index,
                        len(chunks),
                        stripped_focus_query,
                        focus_terms,
                    ),
                }
            )

        scored_chunks.sort(key=lambda item: (-float(item["score"]), int(item["index"])))
        min_score = self._min_focus_score if focus_terms else self._min_salience_score
        primary = [
            item for item in scored_chunks if float(item.get("score", 0.0)) >= min_score
        ][: self._max_selected_chunks]
        if not primary:
            primary = scored_chunks[: self._max_selected_chunks]

        primary_indices = [int(item["index"]) for item in primary]
        selected_indices = self._expand_chunk_indices(primary_indices, len(chunks))
        selected_text = "\n\n".join(chunks[index] for index in selected_indices).strip()
        top_score = float(primary[0].get("score", 0.0)) if primary else 0.0
        coverage_ratio = len(selected_text) / max(1, len(normalized_content))

        needs_fallback = (
            not selected_text
            or len(selected_text) < self._min_selected_chars
            or coverage_ratio < self._coverage_ratio_floor
            or (bool(focus_terms) and top_score < self._min_focus_score)
        )

        mode = "query_pruned" if focus_terms else "salience_pruned"
        if needs_fallback:
            fallback_seed = [
                int(item["index"])
                for item in scored_chunks[: self._max_expanded_chunks]
            ]
            fallback_indices = self._expand_chunk_indices(
                fallback_seed,
                len(chunks),
                max_chunks=min(len(chunks), self._max_expanded_chunks),
            )
            fallback_text = "\n\n".join(chunks[index] for index in fallback_indices).strip()
            if fallback_text and len(fallback_text) > len(selected_text):
                selected_indices = fallback_indices
                selected_text = fallback_text
                coverage_ratio = len(selected_text) / max(1, len(normalized_content))
                mode = "expanded_fallback"

            if (
                not selected_text
                or len(selected_text) < self._min_selected_chars
                or coverage_ratio < self._coverage_ratio_floor
            ):
                selected_indices = list(range(len(chunks)))
                selected_text = normalized_content
                coverage_ratio = 1.0
                mode = "full_fallback"

        return {
            "content": selected_text,
            "index_content": selected_text,
            "selection_stats": {
                "mode": mode,
                "total_chunks": len(chunks),
                "selected_chunk_count": len(selected_indices),
                "selected_chars": len(selected_text),
                "coverage_ratio": round(coverage_ratio, 4),
                "focus_query": stripped_focus_query,
                "top_score": round(top_score, 4),
            },
        }

    async def execute(self, url: str, focus_query: Optional[str] = None) -> Dict:
        try:
            res = await self.extractor.extract_article_content_async(url)
            if not res or not res.get("content"):
                return {"error": "Failed to fetch content or content empty"}

            content = res["content"]
            cache_key = ("rag_pruned_v1", url, (focus_query or "").strip())

            async with _summary_cache_lock:
                cached_summary = _cache_manager.get("tools_summary", cache_key)

            if cached_summary:
                pruned_payload = cached_summary
            else:
                pruned_payload = self._prune_content(
                    content=content,
                    focus_query=focus_query,
                )

            pruned_text = str((pruned_payload or {}).get("content") or "").strip()
            index_content = str((pruned_payload or {}).get("index_content") or pruned_text).strip()
            selection_stats = (
                (pruned_payload or {}).get("selection_stats")
                if isinstance((pruned_payload or {}).get("selection_stats"), dict)
                else {}
            )

            if pruned_text and not cached_summary:
                async with _summary_cache_lock:
                    _cache_manager.set("tools_summary", cache_key, pruned_payload)

            if not pruned_text:
                return {"error": "Content pruning failed", "url": url}

            formatted_content = self._format_pruned_content(
                content=pruned_text,
                source_length=len(content),
                selection_stats=selection_stats,
            )

            res["content"] = formatted_content
            res["is_summarized"] = True
            res["selection_mode"] = selection_stats.get("mode")
            res["selection_stats"] = selection_stats
            res["is_retrieval_pruned"] = True

            try:
                rag = get_rag_service()
                rag.index_news_article(
                    url=url,
                    content=index_content,
                    confidence_score=0.6,
                )
            except Exception as exc:
                logger.debug("RAG news indexing skipped: %s", exc)

            return res
        except Exception as e:
            return {"error": str(e)}


class SearchMemoryTool(ToolBase):
    name = "search_memory"
    description = "Searches persistent RAG memory for prior analyses, reports, and summarized news."

    async def execute(
        self,
        query: Optional[str] = None,
        collection: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 5,
        recent_days: Optional[int] = None,
        from_timestamp: Optional[str] = None,
        to_timestamp: Optional[str] = None,
        detail_level: str = "preview",
        hit_ids: Optional[List[str]] = None,
        context_window: int = 0,
    ) -> Dict[str, Any]:
        normalized_detail_level = str(detail_level or "preview").strip().lower()
        if normalized_detail_level not in {"preview", "full"}:
            normalized_detail_level = "preview"

        if hit_ids:
            try:
                rag = get_rag_service()
                hydrated = rag.fetch_hits(
                    [str(item) for item in hit_ids if str(item).strip()],
                    context_window=max(0, int(context_window or 0)),
                )
            except Exception as exc:
                logger.error("search_memory hydration failed: %s", exc)
                return {"error": str(exc)}

            results: List[Dict[str, Any]] = []
            for item in hydrated:
                collection_name = str(item.get("collection") or "")
                doc_id = str(item.get("id") or "").strip()
                if not collection_name or not doc_id:
                    continue
                results.append(
                    {
                        "hit_id": f"{collection_name}:{doc_id}",
                        "collection": collection_name,
                        "metadata": item.get("metadata") or {},
                        "content": str(item.get("content") or ""),
                    }
                )

            return {
                "mode": "hydrate",
                "detail_level": "full",
                "count": len(results),
                "results": results,
            }

        if not query or not str(query).strip():
            return {"error": "query is required"}

        try:
            rag = get_rag_service()
            hits = rag.search(
                query=str(query).strip(),
                collection=collection,
                symbol_filter=symbol,
                top_k=max(1, int(limit or 5)),
                recent_days=recent_days,
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp,
                context_window=(
                    max(0, int(context_window or 0))
                    if normalized_detail_level == "full"
                    else 0
                ),
            )
        except Exception as exc:
            logger.error("search_memory execution failed: %s", exc)
            return {"error": str(exc)}

        formatted_hits: List[Dict[str, Any]] = []
        ai_cfg = get_config().get("ai", {})
        preview_cfg = ai_cfg.get("search_memory_preview_chars", {})
        if isinstance(preview_cfg, dict):
            default_preview_size = int(preview_cfg.get("default", 1200))
            per_collection_preview = {
                str(k): int(v) for k, v in preview_cfg.items() if k != "default"
            }
        else:
            default_preview_size = int(preview_cfg or 1200)
            per_collection_preview = {}

        for item in hits:
            content = str(item.get("content") or "")
            collection_name = str(item.get("collection") or "")
            preview_size = per_collection_preview.get(
                collection_name, default_preview_size
            )
            doc_id = str(item.get("id") or "").strip()
            payload: Dict[str, Any] = {
                "hit_id": f"{collection_name}:{doc_id}" if collection_name and doc_id else None,
                "collection": item.get("collection"),
                "distance": item.get("distance"),
                "normalized_score": item.get("normalized_score"),
                "quality_adjusted_score": item.get("quality_adjusted_score"),
                "retrieval_score": item.get("retrieval_score"),
                "rerank_score": item.get("rerank_score"),
                "final_score": item.get("final_score"),
                "matched_queries": item.get("matched_queries"),
                "metadata": item.get("metadata") or {},
            }
            if normalized_detail_level == "full":
                payload["content"] = content
            else:
                payload["content_preview"] = content[:preview_size]
                payload["preview_warning"] = (
                    "Preview snippets are lead-generation only. Do not quote exact numbers, "
                    "company classifications, or rankings from preview alone; hydrate hit_ids "
                    "or verify with direct tools first."
                )
            formatted_hits.append(payload)

        return {
            "mode": "search",
            "query": query,
            "detail_level": normalized_detail_level,
            "count": len(formatted_hits),
            "next_step_hint": (
                "Use hit_ids with detail_level='full' to hydrate selected results with full content. "
                "Treat preview hits as hypotheses, not verified evidence."
            ),
            "results": formatted_hits,
        }


class ToolQualityMetricsTool(ToolBase):
    name = "get_tool_quality_metrics"
    description = (
        "Returns aggregated tool execution quality metrics for adaptive planning."
    )

    async def execute(
        self, tool_name: Optional[str] = None, reset: bool = False
    ) -> Dict[str, Any]:
        if reset:
            return reset_tool_quality_metrics(tool_name)
        snapshot = get_tool_quality_metrics()
        if tool_name:
            return {
                "summary": snapshot.get("summary", {}),
                "tool_name": tool_name,
                "metrics": snapshot.get("by_tool", {}).get(tool_name, {}),
            }
        return snapshot


class ToolHealthMetricsTool(ToolBase):
    name = "get_tool_health_metrics"
    description = "Returns tool health status based on recent success/failure streaks."

    async def execute(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        snapshot = get_tool_health_metrics()
        if tool_name:
            return {
                "summary": snapshot.get("summary", {}),
                "tool_name": tool_name,
                "health": snapshot.get("by_tool", {}).get(tool_name, {}),
            }
        return snapshot


class ReportTool(ToolBase):
    name = "report"
    description = "Writes a redacted developer-facing JSONL report for technical issues or actionable improvements."
    output_model = ReportToolResult

    async def execute(
        self,
        title: str,
        category: str,
        severity: str,
        summary: str,
        details: Optional[str] = None,
        suggested_fix: Optional[str] = None,
        context: Optional[Any] = None,
        fingerprint: Optional[str] = None,
    ) -> Dict[str, Any]:
        cfg = get_config().get("ai", {}).get("report_tool", {})
        if not bool(cfg.get("enabled", True)):
            return self._normalize_error(
                "report tool is disabled",
                error_code="tool_disabled",
            )

        if not str(title or "").strip():
            return self._normalize_error(
                "title must be a non-empty string",
                error_code="invalid_arguments",
            )
        if not str(category or "").strip():
            return self._normalize_error(
                "category must be a non-empty string",
                error_code="invalid_arguments",
            )
        if not str(summary or "").strip():
            return self._normalize_error(
                "summary must be a non-empty string",
                error_code="invalid_arguments",
            )

        max_field_chars = max(200, _safe_int(cfg.get("max_field_chars", 4000), 4000))
        max_context_items = max(1, _safe_int(cfg.get("max_context_items", 20), 20))
        max_reports_per_process = max(1, _safe_int(cfg.get("max_reports_per_process", 200), 200))
        dedupe_window = max(0, _safe_int(cfg.get("dedupe_window", 50), 50))
        redact = bool(cfg.get("redact_sensitive_values", True))

        safe_title = _sanitize_report_value(
            title,
            max_chars=min(240, max_field_chars),
            max_items=max_context_items,
            redact=redact,
        )
        safe_category = _sanitize_report_value(
            category,
            max_chars=80,
            max_items=max_context_items,
            redact=False,
        )
        safe_summary = _sanitize_report_value(
            summary,
            max_chars=max_field_chars,
            max_items=max_context_items,
            redact=redact,
        )
        safe_details = _sanitize_report_value(
            details,
            max_chars=max_field_chars,
            max_items=max_context_items,
            redact=redact,
        )
        safe_fix = _sanitize_report_value(
            suggested_fix,
            max_chars=max_field_chars,
            max_items=max_context_items,
            redact=redact,
        )
        safe_context = _sanitize_report_value(
            context,
            max_chars=max_field_chars,
            max_items=max_context_items,
            redact=redact,
        )

        severity_normalized = str(severity or "medium").strip().lower()
        if severity_normalized not in {"low", "medium", "high", "critical"}:
            severity_normalized = "medium"

        resolved_fingerprint = str(fingerprint or "").strip() or _build_report_fingerprint(
            str(safe_title),
            str(safe_category),
            str(safe_summary),
        )

        with _report_lock:
            recent_window = _recent_report_fingerprints[-dedupe_window:] if dedupe_window else []
            if dedupe_window and resolved_fingerprint in recent_window:
                return {
                    "status": "duplicate_suppressed",
                    "fingerprint": resolved_fingerprint,
                    "duplicate_suppressed": True,
                    "path": str(get_ai_reports_path()),
                }

            report_id = str(uuid4())
            record = {
                "report_id": report_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "title": safe_title,
                "category": safe_category,
                "severity": severity_normalized,
                "summary": safe_summary,
                "details": safe_details,
                "suggested_fix": safe_fix,
                "context": safe_context,
                "fingerprint": resolved_fingerprint,
                "source": "ai_tool",
            }

            reports_path = get_ai_reports_path()
            with open(reports_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

            _recent_report_fingerprints.append(resolved_fingerprint)
            if len(_recent_report_fingerprints) > max_reports_per_process:
                del _recent_report_fingerprints[:-max_reports_per_process]

        return {
            "status": "reported",
            "report_id": report_id,
            "fingerprint": resolved_fingerprint,
            "duplicate_suppressed": False,
            "path": str(get_ai_reports_path()),
        }


class PythonExecTool(ToolBase):
    name = "python_exec"
    description = "Executes bounded local Python inside an isolated sandbox for exact calculations and short scripts."
    output_model = PythonExecResult

    async def execute(
        self,
        code: str,
        input_data: Optional[Any] = None,
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        cfg = get_config().get("ai", {}).get("python_exec", {})
        if not bool(cfg.get("enabled", True)):
            return self._normalize_error(
                "python_exec tool is disabled",
                error_code="tool_disabled",
            )

        source = str(code or "")
        if not source.strip():
            return self._normalize_error(
                "code must be a non-empty string",
                error_code="invalid_arguments",
            )

        default_timeout = max(1, int(cfg.get("default_timeout_seconds", 8)))
        max_timeout = max(default_timeout, _safe_int(cfg.get("max_timeout_seconds", 20), 20))
        safe_timeout = default_timeout
        if timeout_seconds is not None:
            try:
                safe_timeout = max(1, min(int(timeout_seconds), max_timeout))
            except (TypeError, ValueError):
                safe_timeout = default_timeout

        max_output_chars = max(500, _safe_int(cfg.get("max_output_chars", 12000), 12000))
        max_memory_mb = max(64, _safe_int(cfg.get("max_memory_mb", 256), 256))
        max_file_size_bytes = max(65536, _safe_int(cfg.get("max_file_size_mb", 2), 2) * 1024 * 1024)
        blocked_modules = [
            str(name).strip()
            for name in cfg.get("blocked_modules", [])
            if str(name).strip()
        ]

        sandbox_root = get_python_exec_dir()
        runner_script = _build_python_runner_script()

        try:
            with tempfile.TemporaryDirectory(dir=sandbox_root, prefix="run_") as sandbox_dir:
                payload_path = os.path.join(sandbox_dir, "payload.json")
                runner_path = os.path.join(sandbox_dir, "runner.py")
                payload = {
                    "code": source,
                    "input_data": input_data,
                    "max_output_chars": max_output_chars,
                    "max_memory_mb": max_memory_mb,
                    "max_file_size_bytes": max_file_size_bytes,
                    "blocked_modules": blocked_modules,
                }

                with open(payload_path, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, default=str)
                with open(runner_path, "w", encoding="utf-8") as handle:
                    handle.write(runner_script)

                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-I",
                    runner_path,
                    payload_path,
                    cwd=sandbox_dir,
                    env={
                        "PYTHONIOENCODING": "utf-8",
                        "PYTHONNOUSERSITE": "1",
                    },
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        process.communicate(),
                        timeout=safe_timeout + 2,
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await process.communicate()
                    return {
                        "error": f"python_exec timed out after {safe_timeout} seconds",
                        "error_code": "timeout",
                        "details": {
                            "timed_out": True,
                            "timeout_seconds": safe_timeout,
                        },
                    }

                stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
                stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()

                if not stdout_text:
                    return {
                        "error": "python_exec returned no payload",
                        "error_code": "execution_error",
                        "details": _truncate_text(stderr_text, max_output_chars),
                    }

                try:
                    payload_out = json.loads(stdout_text)
                except json.JSONDecodeError:
                    return {
                        "error": "python_exec produced invalid JSON payload",
                        "error_code": "execution_error",
                        "details": _truncate_text(stdout_text or stderr_text, max_output_chars),
                    }

                if not isinstance(payload_out, dict):
                    return {
                        "error": "python_exec returned unexpected payload type",
                        "error_code": "execution_error",
                    }

                if not payload_out.get("ok"):
                    return {
                        "error": str(payload_out.get("error") or "python_exec failed"),
                        "error_code": str(payload_out.get("error_code") or "execution_error"),
                        "details": {
                            "stdout": _truncate_text(payload_out.get("stdout"), max_output_chars),
                            "stderr": _truncate_text(payload_out.get("stderr") or stderr_text, max_output_chars),
                            "execution_ms": payload_out.get("execution_ms"),
                        },
                    }

                return {
                    "stdout": _truncate_text(payload_out.get("stdout"), max_output_chars),
                    "stderr": _truncate_text(payload_out.get("stderr") or stderr_text, max_output_chars),
                    "result": payload_out.get("result"),
                    "result_repr": _truncate_text(payload_out.get("result_repr"), max_output_chars),
                    "execution_ms": int(payload_out.get("execution_ms") or 0),
                    "timed_out": False,
                    "files": payload_out.get("files") if isinstance(payload_out.get("files"), list) else [],
                    "sandbox_ephemeral": bool(payload_out.get("sandbox_ephemeral", True)),
                }
        except Exception as exc:
            return self._normalize_error(
                f"python_exec failed: {exc}",
                error_code="execution_error",
            )


class YFinanceSearchTool(ToolBase):
    name = "yfinance_search"
    description = (
        "Searches for stock tickers, indices or ETFs on Yahoo Finance. "
        "Use type_filter to narrow results to a specific asset class (stock/etf/index/crypto). "
        "Use news_results to include recent Yahoo Finance news alongside quotes."
    )

    # Essential quote fields only — keeps token usage low
    _QUOTE_FIELDS = frozenset(
        {"symbol", "shortname", "longname", "exchange", "quoteType", "score"}
    )

    def __init__(self):
        self.finance_service = get_financial_service()

    def _filter_quotes(self, quotes: Any) -> Any:
        if not isinstance(quotes, list):
            return quotes
        return [
            {k: v for k, v in q.items() if k in self._QUOTE_FIELDS}
            for q in quotes
            if isinstance(q, dict)
        ]

    async def execute(
        self,
        query: str,
        max_results: int = 10,
        news_results: int = 0,
        type_filter: Optional[str] = None,
    ) -> Dict:
        try:
            if type_filter:
                # Precise type-filtered lookup via yf.Lookup
                raw = await asyncio.to_thread(
                    self.finance_service.lookup_ticker,
                    query,
                    count=max_results,
                    asset_type=type_filter.lower(),
                )
                return raw

            # General search via yf.Search — quotes + optional news + research
            raw = await asyncio.to_thread(
                self.finance_service.search_ticker,
                query,
                max_results=max_results,
                news_count=news_results,
            )
            if isinstance(raw, dict):
                if "quotes" in raw:
                    raw["quotes"] = self._filter_quotes(raw["quotes"])
                if not news_results:
                    raw.pop("news", None)
                # Provide explicit guidance when no results returned.
                if not raw.get("quotes") and not raw.get("news"):
                    raw["note"] = (
                        "No results found for this query. "
                        "Try a more specific ticker symbol (e.g. 'FROTO.IS'), "
                        "a direct lookup with type_filter='stock', or search in Turkish."
                    )
            return raw
        except Exception as e:
            logger.error("yFinance Search Error: %s", e)
            return {"error": str(e)}


def _resolve_market_identifier(*candidates: Any) -> Optional[str]:
    for candidate in candidates:
        value = str(candidate or "").strip()
        if value:
            return value
    return None


def _normalize_index_symbol_candidates(symbol: str) -> List[str]:
    cleaned = str(symbol or "").strip().upper()
    if not cleaned:
        return []

    candidates: List[str] = []

    if cleaned.startswith("^") and len(cleaned) > 1:
        without_caret = cleaned[1:]
        if without_caret.startswith("XU") and not without_caret.endswith(".IS"):
            candidates.append(f"{without_caret}.IS")
        candidates.append(cleaned)
        candidates.append(without_caret)
    elif cleaned.startswith("XU") and not cleaned.endswith(".IS"):
        candidates.append(f"{cleaned}.IS")
        candidates.append(cleaned)
    else:
        candidates.append(cleaned)

    deduped: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _extract_stock_codes(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        candidates = re.findall(r"[A-ZÇĞİÖŞÜ]{2,10}(?:\.IS)?", value.upper())
    elif isinstance(value, (list, tuple, set)):
        candidates = []
        for item in value:
            candidates.extend(_extract_stock_codes(item))
    else:
        return []

    normalized_codes: List[str] = []
    blocked_tokens = {"BIST", "XU100", "XU030", "XU050", "INDEX", "ENDX"}
    for code in candidates:
        normalized = str(code).strip().upper()
        if "." in normalized:
            normalized = normalized.split(".", 1)[0]
        if not normalized or normalized in blocked_tokens or normalized.isdigit():
            continue
        if normalized not in normalized_codes:
            normalized_codes.append(normalized)
    return normalized_codes


def _resolve_market_screen_scope(
    *,
    exchange: Optional[str] = None,
    region: Optional[str] = None,
    country: Optional[str] = None,
    symbol: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    exchange_text = str(exchange or "").strip().upper()
    region_text = str(region or country or "").strip().lower()
    symbol_text = str(symbol or "").strip().upper()

    if exchange_text in {"BIST", "BORSA ISTANBUL", "BORSA İSTANBUL", "IST"}:
        return {
            "region": "tr",
            "exchange": "IST",
            "index_symbol": "XU100.IS",
            "market_label": "BIST",
        }

    if symbol_text in {"XU100.IS", "^XU100", "XU100"}:
        return {
            "region": "tr",
            "exchange": "IST",
            "index_symbol": "XU100.IS",
            "market_label": "BIST",
        }

    if region_text in {"tr", "turkey", "turkiye", "türkiye"}:
        return {
            "region": "tr",
            "exchange": exchange_text or "IST",
            "index_symbol": "XU100.IS",
            "market_label": "BIST",
        }

    return None


class YFinanceSectorTool(ToolBase):
    name = "yfinance_sector_analysis"
    description = "Gets detailed information about a financial sector or industry."

    def __init__(self):
        self.finance_service = get_financial_service()

    async def execute(
        self,
        sector_key: Optional[str] = None,
        industry_key: Optional[str] = None,
        sector: Optional[str] = None,
        symbol: Optional[str] = None,
        ticker: Optional[str] = None,
        country: Optional[str] = None,
        region: Optional[str] = None,
        exchange: Optional[str] = None,
        max_tickers: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict:
        try:
            sector_key = str(sector_key or "").strip() or None
            industry_key = str(industry_key or "").strip() or None
            sector = str(
                sector or kwargs.get("sector_filter") or kwargs.get("sector_name") or ""
            ).strip() or None
            exchange = str(exchange or kwargs.get("exchange") or "").strip() or None
            fallback_symbol = _resolve_market_identifier(
                symbol,
                ticker,
                kwargs.get("symbol"),
                kwargs.get("ticker"),
            )

            if not fallback_symbol and exchange:
                exchange_upper = exchange.upper()
                exchange_symbol_map = {
                    "BIST": "XU100.IS",
                    "BORSA ISTANBUL": "XU100.IS",
                    "BORSA İSTANBUL": "XU100.IS",
                    "XU100": "XU100.IS",
                }
                fallback_symbol = exchange_symbol_map.get(exchange_upper)

            if not sector_key and sector and sector.strip().lower() != "all":
                normalized = re.sub(r"\s+", "-", sector.strip().lower())
                normalized = re.sub(r"[^a-z0-9\-]", "", normalized)
                sector_key = normalized or None

            market_scope = _resolve_market_screen_scope(
                exchange=exchange,
                region=region,
                country=country,
                symbol=fallback_symbol,
            )

            if sector_key:
                return self.finance_service.get_sector_info(sector_key)
            elif industry_key:
                return self.finance_service.get_industry_info(industry_key)
            elif market_scope:
                screen_limit = max(1, min(int(max_tickers or 25), 100))
                screen_result = self.finance_service.screen_equities_by_market(
                    region=market_scope.get("region"),
                    exchange=market_scope.get("exchange"),
                    sector=sector,
                    limit=screen_limit,
                )
                quotes = (
                    screen_result.get("quotes") if isinstance(screen_result, dict) else []
                )
                if quotes:
                    payload: Dict[str, Any] = {
                        "mode": "market_screen",
                        "requested_sector": sector,
                        "requested_symbol": fallback_symbol,
                        "requested_exchange": exchange,
                        "requested_region": str(region or country or "").strip() or None,
                        "market_scope": market_scope,
                        **screen_result,
                    }
                    index_symbol = market_scope.get("index_symbol") or fallback_symbol
                    if index_symbol:
                        index_context = self.finance_service.get_index_data(index_symbol)
                        if index_context:
                            payload["index_context"] = index_context
                    return payload
            elif fallback_symbol:
                index_data = self.finance_service.get_index_data(fallback_symbol)
                if index_data:
                    return {
                        "mode": "index_fallback",
                        "requested_symbol": fallback_symbol,
                        "requested_exchange": exchange,
                        "requested_region": str(region or country or "").strip() or None,
                        "note": (
                            "Symbol-based request was routed to index data because "
                            "sector_key/industry_key was not provided."
                        ),
                        "data": index_data,
                    }
            return {
                "error": (
                    "Must provide either sector_key or industry_key. "
                    f"Received sector={sector!r}, symbol={fallback_symbol!r}, "
                    f"country={country!r}, region={region!r}, exchange={exchange!r}, "
                    f"max_tickers={max_tickers!r}"
                ),
                "error_code": "validation_error",
            }
        except Exception as e:
            logger.error("yFinance Sector Error: %s", e)
            return {"error": str(e)}


class YFinanceCompanyTool(ToolBase):
    name = "yfinance_company_data"
    description = "Fetch comprehensive financial data for a specific ticker symbol (e.g., 'AAPL', 'THYAO.IS')."

    _CORE_FIELDS = frozenset(
        {
            "symbol",
            "shortName",
            "longName",
            "sector",
            "industry",
            "country",
            "currency",
            "exchange",
            "quoteType",
            "currentPrice",
            "previousClose",
            "open",
            "dayLow",
            "dayHigh",
            "fiftyTwoWeekLow",
            "fiftyTwoWeekHigh",
            "marketCap",
            "enterpriseValue",
            "volume",
            "averageVolume",
            "beta",
            "trailingPE",
            "forwardPE",
            "priceToBook",
            "priceToSalesTrailing12Months",
            "pegRatio",
            "profitMargins",
            "operatingMargins",
            "grossMargins",
            "returnOnEquity",
            "returnOnAssets",
            "debtToEquity",
            "currentRatio",
            "quickRatio",
            "totalRevenue",
            "ebitda",
            "operatingCashflow",
            "freeCashflow",
            "totalCash",
            "totalDebt",
            "revenueGrowth",
            "earningsGrowth",
            "data_quality",
            "data_quality_report",
            "data_sources",
            "data_source",
            "macro_context",
        }
    )

    _NESTED_SUMMARY_FIELDS = frozenset(
        {
            "company_profile",
            "financial_metrics",
            "valuation_metrics",
            "technical_indicators",
            "balance_sheet_summary",
            "income_statement_summary",
            "cashflow_summary",
        }
    )

    def __init__(self):
        self.finance_service = get_financial_service()
        ai_cfg = get_config().get("ai", {})
        tool_output_cfg = (
            ai_cfg.get("tool_output", {}) if isinstance(ai_cfg, dict) else {}
        )
        if isinstance(tool_output_cfg, dict):
            self._compact_output_enabled = bool(
                tool_output_cfg.get("compact_yfinance_company_data", True)
            )
        else:
            self._compact_output_enabled = True

    def _build_compact_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        compact: Dict[str, Any] = {}
        for key in self._CORE_FIELDS:
            value = data.get(key)
            if value is not None:
                compact[key] = value

        for key in self._NESTED_SUMMARY_FIELDS:
            value = data.get(key)
            if isinstance(value, dict) and value:
                compact[key] = value

        shares_full = data.get("sharesFull")
        if isinstance(shares_full, dict):
            trimmed_shares = {k: v for k, v in shares_full.items() if k != "history"}
            if trimmed_shares:
                compact["sharesFull"] = trimmed_shares

        omitted = [k for k in data.keys() if k not in compact]
        if omitted:
            compact["_meta"] = {
                "compact": True,
                "omitted_fields_count": len(omitted),
                "omitted_fields_preview": omitted[:40],
            }
        return compact

    async def execute(
        self,
        ticker: Optional[str] = None,
        include_financials: Optional[bool] = None,
        symbol: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict:
        try:
            resolved_ticker = _resolve_market_identifier(
                ticker,
                symbol,
                kwargs.get("ticker"),
                kwargs.get("symbol"),
            )
            if not resolved_ticker:
                return {
                    "error": "ticker or symbol is required",
                    "error_code": "validation_error",
                }

            data = self.finance_service.get_company_data(resolved_ticker)
            if not data:
                return {"error": f"No data found for {resolved_ticker}", "error_code": "empty_result"}
            if not isinstance(data, dict):
                return {"error": "Unexpected company data format", "error_code": "empty_result"}
            if not self._compact_output_enabled:
                return data
            return self._build_compact_payload(data)
        except Exception as e:
            logger.error("yFinance Company Data Error: %s", e)
            return {"error": str(e)}


class YFinanceOverviewTool(ToolBase):
    name = "yfinance_overview"
    description = "Fetch modular overview data for a ticker symbol."

    def __init__(self):
        self.finance_service = get_financial_service()

    async def execute(
        self,
        ticker: Optional[str] = None,
        symbol: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            resolved_ticker = _resolve_market_identifier(
                ticker,
                symbol,
                kwargs.get("ticker"),
                kwargs.get("symbol"),
            )
            if not resolved_ticker:
                return {
                    "error": "ticker or symbol is required",
                    "error_code": "validation_error",
                }

            data = self.finance_service.get_overview(resolved_ticker)
            if not data:
                return {"error": f"No overview data found for {resolved_ticker}", "error_code": "empty_result"}
            return data
        except Exception as e:
            logger.error("yFinance Overview Error: %s", e)
            return {"error": str(e)}


class YFinanceIndexDataTool(ToolBase):
    name = "yfinance_index_data"
    description = "Fetch index/ETF data snapshot for a symbol."

    def __init__(self):
        self.finance_service = get_financial_service()

    async def execute(
        self,
        symbol: Optional[str] = None,
        ticker: Optional[str] = None,
        index_symbol: Optional[str] = None,
        include_components: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            resolved_symbol = _resolve_market_identifier(
                symbol,
                ticker,
                index_symbol,
                kwargs.get("symbol"),
                kwargs.get("ticker"),
                kwargs.get("index_symbol"),
            )
            if not resolved_symbol:
                return {
                    "error": "symbol or ticker is required",
                    "error_code": "validation_error",
                }

            attempted_symbols = _normalize_index_symbol_candidates(resolved_symbol)
            for candidate_symbol in attempted_symbols:
                data = self.finance_service.get_index_data(candidate_symbol)
                if data:
                    if isinstance(data, dict):
                        normalized_data = dict(data)
                        current_price = normalized_data.get("regularMarketPrice")
                        week_low = normalized_data.get("fiftyTwoWeekLow")
                        week_high = normalized_data.get("fiftyTwoWeekHigh")
                        if (
                            isinstance(current_price, (int, float))
                            and isinstance(week_low, (int, float))
                            and isinstance(week_high, (int, float))
                            and current_price > 0
                            and week_high > 0
                            and (
                                week_low <= 0
                                or week_low > week_high
                                or week_low < float(current_price) * 0.2
                                or float(current_price) > float(week_high) * 2.0
                            )
                        ):
                            logger.info(
                                "Ignoring incoherent index snapshot for %s (requested=%s): price=%s low=%s high=%s",
                                candidate_symbol,
                                resolved_symbol,
                                current_price,
                                week_low,
                                week_high,
                            )
                            continue
                        if candidate_symbol != resolved_symbol:
                            normalized_data["requested_symbol"] = resolved_symbol
                            normalized_data["resolved_symbol"] = candidate_symbol
                        if include_components is not None:
                            normalized_data["include_components_requested"] = bool(include_components)
                        data = normalized_data
                    return data

            return {
                "error": f"No index/ETF data found for {resolved_symbol}",
                "error_code": "empty_result",
                "attempted_symbols": attempted_symbols,
            }
        except Exception as e:
            logger.error("yFinance Index Data Error: %s", e)
            return {"error": str(e)}


class YFinancePriceHistoryTool(ToolBase):
    name = "yfinance_price_history"
    description = "Fetch OHLCV history and technical indicators for a ticker."

    def __init__(self):
        self.finance_service = get_financial_service()

    async def execute(
        self,
        ticker: Optional[str] = None,
        period: str = "1y",
        interval: str = "1d",
        symbol: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            resolved_ticker = _resolve_market_identifier(
                ticker,
                symbol,
                kwargs.get("ticker"),
                kwargs.get("symbol"),
            )
            if not resolved_ticker:
                return {
                    "error": "ticker or symbol is required",
                    "error_code": "validation_error",
                }

            # Accept 'range' as an alias for 'period' (models often use yfinance terminology)
            effective_period = period if period != "1y" else kwargs.get("range", period)

            data = self.finance_service.get_price_history(
                ticker=resolved_ticker,
                period=effective_period,
                interval=interval,
            )
            if not data:
                return {"error": f"No price history found for {resolved_ticker}", "error_code": "empty_result"}
            return data
        except Exception as e:
            logger.error("yFinance Price History Error: %s", e)
            return {"error": str(e)}


class YFinanceDividendsTool(ToolBase):
    name = "yfinance_dividends"
    description = "Fetch dividend and split history for a ticker."

    def __init__(self):
        self.finance_service = get_financial_service()

    async def execute(
        self,
        ticker: Optional[str] = None,
        symbol: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            resolved_ticker = _resolve_market_identifier(
                ticker,
                symbol,
                kwargs.get("ticker"),
                kwargs.get("symbol"),
            )
            if not resolved_ticker:
                return {
                    "error": "ticker or symbol is required",
                    "error_code": "validation_error",
                }

            data = self.finance_service.get_dividends(resolved_ticker)
            if not data:
                return {"error": f"No dividends data found for {resolved_ticker}", "error_code": "empty_result"}
            return data
        except Exception as e:
            logger.error("yFinance Dividends Error: %s", e)
            return {"error": str(e)}


class YFinanceAnalystTool(ToolBase):
    name = "yfinance_analyst"
    description = "Fetch analyst recommendation and estimates for a ticker."

    def __init__(self):
        self.finance_service = get_financial_service()

    async def execute(
        self,
        ticker: Optional[str] = None,
        symbol: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            resolved_ticker = _resolve_market_identifier(
                ticker,
                symbol,
                kwargs.get("ticker"),
                kwargs.get("symbol"),
            )
            if not resolved_ticker:
                return {
                    "error": "ticker or symbol is required",
                    "error_code": "validation_error",
                }

            data = self.finance_service.get_analyst(resolved_ticker)
            if not data:
                return {"error": f"No analyst data found for {resolved_ticker}", "error_code": "empty_result"}
            return data
        except Exception as e:
            logger.error("yFinance Analyst Error: %s", e)
            return {"error": str(e)}


class YFinanceEarningsTool(ToolBase):
    name = "yfinance_earnings"
    description = "Fetch earnings trend and EPS history for a ticker."

    def __init__(self):
        self.finance_service = get_financial_service()

    async def execute(
        self,
        ticker: Optional[str] = None,
        symbol: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            resolved_ticker = _resolve_market_identifier(
                ticker,
                symbol,
                kwargs.get("ticker"),
                kwargs.get("symbol"),
            )
            if not resolved_ticker:
                return {
                    "error": "ticker or symbol is required",
                    "error_code": "validation_error",
                }

            data = self.finance_service.get_earnings(resolved_ticker)
            if not data:
                return {"error": f"No earnings data found for {resolved_ticker}", "error_code": "empty_result"}
            return data
        except Exception as e:
            logger.error("yFinance Earnings Error: %s", e)
            return {"error": str(e)}


class YFinanceOwnershipTool(ToolBase):
    name = "yfinance_ownership"
    description = "Fetch ownership and insider snapshots for a ticker."

    def __init__(self):
        self.finance_service = get_financial_service()

    async def execute(
        self,
        ticker: Optional[str] = None,
        symbol: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            resolved_ticker = _resolve_market_identifier(
                ticker,
                symbol,
                kwargs.get("ticker"),
                kwargs.get("symbol"),
            )
            if not resolved_ticker:
                return {
                    "error": "ticker or symbol is required",
                    "error_code": "validation_error",
                }

            data = self.finance_service.get_ownership(resolved_ticker)
            if not data:
                return {"error": f"No ownership data found for {resolved_ticker}", "error_code": "empty_result"}
            return data
        except Exception as e:
            logger.error("yFinance Ownership Error: %s", e)
            return {"error": str(e)}


class YFinanceSustainabilityTool(ToolBase):
    name = "yfinance_sustainability"
    description = "Fetch sustainability/ESG snapshot for a ticker."

    def __init__(self):
        self.finance_service = get_financial_service()

    async def execute(
        self,
        ticker: Optional[str] = None,
        symbol: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            resolved_ticker = _resolve_market_identifier(
                ticker,
                symbol,
                kwargs.get("ticker"),
                kwargs.get("symbol"),
            )
            if not resolved_ticker:
                return {
                    "error": "ticker or symbol is required",
                    "error_code": "validation_error",
                }

            data = self.finance_service.get_sustainability(resolved_ticker)
            if not data:
                return {"error": f"No sustainability data found for {resolved_ticker}", "error_code": "empty_result"}
            return data
        except Exception as e:
            logger.error("yFinance Sustainability Error: %s", e)
            return {"error": str(e)}


class KapSearchDisclosuresTool(ToolBase):
    name = "kap_search_disclosures"
    description = "Search KAP disclosures by stock code and filters."
    output_model = KapSearchResult

    async def execute(
        self,
        stock_codes: Optional[List[str]] = None,
        stock_code: Optional[str] = None,
        symbol: Optional[str] = None,
        ticker: Optional[str] = None,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        category: Optional[str] = None,
        days: Optional[int] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        if not stock_codes:
            derived_codes = _extract_stock_codes(stock_code)
            if not derived_codes:
                derived_codes = _extract_stock_codes(symbol)
            if not derived_codes:
                derived_codes = _extract_stock_codes(ticker)
            if not derived_codes:
                derived_codes = _extract_stock_codes(query)
            if derived_codes:
                stock_codes = derived_codes
        if not stock_codes:
            return {
                "error": "stock_codes must contain at least one symbol",
                "error_code": "validation_error",
            }

        normalized_codes = []
        for code in stock_codes:
            if not isinstance(code, str):
                continue
            normalized = code.strip().upper()
            if "." in normalized:
                normalized = normalized.split(".", 1)[0]
            if normalized:
                normalized_codes.append(normalized)

        if not normalized_codes:
            return {
                "error": "No valid stock code found in stock_codes",
                "error_code": "validation_error",
            }
        try:
            items = await asyncio.to_thread(
                search_disclosures,
                normalized_codes,
                category=category,
                days=days,
                from_date=from_date,
                to_date=to_date,
                limit=limit,
            )
            return [
                item.model_dump(by_alias=True, exclude_none=True, mode="json")
                for item in items
            ]
        except KapLookupError as exc:
            return {
                "error": str(exc),
                "details": exc.details,
                "error_code": "validation_error",
            }
        except httpx.HTTPStatusError as exc:
            status_code = getattr(exc.response, "status_code", "unknown")
            return {
                "error": f"KAP list request failed ({status_code})",
                "error_code": "network_error",
            }
        except httpx.RequestError as exc:
            logger.error("KAP disclosures list error: %s", exc)
            return {"error": str(exc), "error_code": "network_error"}
        except Exception as exc:
            logger.error("KAP disclosures list error: %s", exc)
            return {"error": str(exc)}


class KapGetDisclosureDetailTool(ToolBase):
    name = "kap_get_disclosure_detail"
    description = "Get full KAP disclosure details by disclosure index."
    output_model = KapDisclosureDetailResult

    async def execute(self, disclosure_index: int) -> Any:
        try:
            normalized_index = int(disclosure_index)
        except (TypeError, ValueError):
            return {
                "error": "disclosure_index must be an integer",
                "error_code": "validation_error",
            }

        try:
            detail = await asyncio.to_thread(get_disclosure_detail, normalized_index)
            if detail is None:
                return {
                    "error": "Disclosure detail could not be parsed",
                    "error_code": "empty_result",
                }
            return detail.model_dump(by_alias=True, exclude_none=True, mode="json")
        except httpx.HTTPStatusError as exc:
            status_code = getattr(exc.response, "status_code", "unknown")
            return {
                "error": f"KAP detail request failed ({status_code})",
                "error_code": "network_error",
            }
        except httpx.RequestError as exc:
            logger.error("KAP disclosure detail error: %s", exc)
            return {"error": str(exc), "error_code": "network_error"}
        except Exception as exc:
            logger.error("KAP disclosure detail error: %s", exc)
            return {"error": str(exc)}


class KapBatchDisclosureDetailsTool(ToolBase):
    name = "kap_batch_disclosure_details"
    description = "Fetch multiple KAP disclosure details in parallel from embedded KAP integration."
    output_model = KapBatchDetailsResult

    async def execute(self, disclosure_indexes: List[int], max_workers: int = 5) -> Any:
        if not disclosure_indexes:
            return {
                "error": "disclosure_indexes must contain at least one index",
                "error_code": "validation_error",
            }

        safe_max_workers = max(1, int(max_workers))

        normalized_indexes: List[int] = []
        for disclosure_index in disclosure_indexes:
            try:
                normalized_indexes.append(int(disclosure_index))
            except (TypeError, ValueError):
                continue

        if not normalized_indexes:
            return {"error": "No valid disclosure index found in disclosure_indexes", "error_code": "validation_error"}

        try:
            results, errors = await asyncio.to_thread(
                batch_get_disclosure_details,
                normalized_indexes,
                safe_max_workers,
            )
            return {
                "results": [
                    (
                        item.model_dump(by_alias=True, exclude_none=True, mode="json")
                        if item is not None
                        else None
                    )
                    for item in results
                ],
                "errors": {str(key): value for key, value in errors.items()},
            }
        except httpx.HTTPStatusError as exc:
            status_code = getattr(exc.response, "status_code", "unknown")
            return {
                "error": f"KAP batch detail request failed ({status_code})",
                "error_code": "network_error",
            }
        except httpx.RequestError as exc:
            logger.error("KAP batch detail error: %s", exc)
            return {"error": str(exc), "error_code": "network_error"}
        except Exception as exc:
            logger.error("KAP batch detail error: %s", exc)
            return {"error": str(exc)}


_tool_cache = {}


def _initialize_tool_cache() -> None:
    _tool_cache["search_web"] = DDGSSearchTool()
    _tool_cache["search_news"] = DDGSNewsTool()
    _tool_cache["search_google_news"] = GoogleNewsTool()
    _tool_cache["fetch_url_content"] = FetchUrlContentTool()
    _tool_cache["summarize_url_content"] = SummarizeUrlContentTool()
    _tool_cache["yfinance_search"] = YFinanceSearchTool()
    _tool_cache["yfinance_sector_analysis"] = YFinanceSectorTool()
    _tool_cache["yfinance_company_data"] = YFinanceCompanyTool()
    _tool_cache["yfinance_overview"] = YFinanceOverviewTool()
    _tool_cache["yfinance_index_data"] = YFinanceIndexDataTool()
    _tool_cache["yfinance_price_history"] = YFinancePriceHistoryTool()
    _tool_cache["yfinance_dividends"] = YFinanceDividendsTool()
    _tool_cache["yfinance_analyst"] = YFinanceAnalystTool()
    _tool_cache["yfinance_earnings"] = YFinanceEarningsTool()
    _tool_cache["yfinance_ownership"] = YFinanceOwnershipTool()
    _tool_cache["yfinance_sustainability"] = YFinanceSustainabilityTool()
    _tool_cache["kap_search_disclosures"] = KapSearchDisclosuresTool()
    _tool_cache["kap_get_disclosure_detail"] = KapGetDisclosureDetailTool()
    _tool_cache["kap_batch_disclosure_details"] = KapBatchDisclosureDetailsTool()
    _tool_cache["search_memory"] = SearchMemoryTool()
    _tool_cache["get_tool_quality_metrics"] = ToolQualityMetricsTool()
    _tool_cache["get_tool_health_metrics"] = ToolHealthMetricsTool()
    _tool_cache["report"] = ReportTool()
    _tool_cache["python_exec"] = PythonExecTool()


def get_tool(tool_name: str) -> Optional[ToolBase]:
    if not _tool_cache:
        with _tool_init_lock:
            if not _tool_cache:
                _initialize_tool_cache()
    return _tool_cache.get(tool_name)


async def execute_tool(
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    console: Optional[Console] = None,
    max_chars: Optional[int] = None,
) -> Any:
    safe_args = args if isinstance(args, dict) else {}
    logger.info(
        "Tool execution started: tool=%s args_keys=%s args_size=%d",
        tool_name,
        sorted(list(safe_args.keys())),
        _safe_json_size(safe_args),
    )
    tool = get_tool(tool_name)
    if not tool:
        result = {
            "error": f"Tool '{tool_name}' not found.",
            "error_code": "tool_not_found",
            "tool": str(tool_name),
        }
        logger.warning(
            "Tool execution failed: tool=%s reason=not_found", tool_name
        )
        _record_tool_metrics(tool_name, result=result, duration_ms=0)
        return result

    blocked_result = _precheck_tool_health(tool_name)
    if blocked_result:
        logger.warning(
            "Tool execution blocked by health check: tool=%s error=%s",
            tool_name,
            str(blocked_result.get("error", ""))[:160],
        )
        _record_tool_metrics(tool_name, result=blocked_result, duration_ms=0)
        return blocked_result

    tool.console = console

    network_cfg = get_config().get("network", {})
    smart_retry_cfg = network_cfg.get("smart_retry", {})
    retry_enabled = bool(smart_retry_cfg.get("enabled", True))
    max_attempts = max(1, int(network_cfg.get("max_retry_count", 3)))
    base_delay = max(0.05, float(network_cfg.get("retry_delay_seconds", 1.0)))
    jitter_ratio = max(0.0, float(smart_retry_cfg.get("jitter_ratio", 0.25)))
    max_delay = max(base_delay, float(smart_retry_cfg.get("max_delay_seconds", 15.0)))

    final_result: Any = None
    duration_ms = 0

    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            attempt_result = await tool.execute(**(args or {}))
        except Exception as exc:
            attempt_result = {
                "error": f"Error executing tool {tool_name}: {str(exc)}",
                "error_code": "execution_error",
                "tool": tool_name,
            }

        attempt_result = tool.sanitize_output(attempt_result, max_chars=max_chars)
        duration_ms = int((time.perf_counter() - started) * 1000)

        result_count = _infer_result_count(attempt_result)
        logger.info(
            "Tool execution finished: tool=%s attempt=%d/%d duration_ms=%d result_type=%s result_count=%s has_error=%s",
            tool_name,
            attempt,
            max_attempts,
            duration_ms,
            type(attempt_result).__name__,
            result_count,
            bool(isinstance(attempt_result, dict) and attempt_result.get("error")),
        )
        if isinstance(attempt_result, dict):
            logger.debug(
                "Tool result preview: tool=%s keys=%s",
                tool_name,
                sorted(list(attempt_result.keys()))[:20],
            )

        should_retry = (
            retry_enabled
            and attempt < max_attempts
            and _is_retryable_tool_error(attempt_result)
        )
        if not should_retry:
            final_result = attempt_result
            break

        retry_after = attempt_result.get("retry_after") if isinstance(attempt_result, dict) else None
        try:
            retry_after_value = float(retry_after) if retry_after is not None else 0.0
        except (TypeError, ValueError):
            retry_after_value = 0.0

        backoff_delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
        wait_seconds = max(backoff_delay, retry_after_value)
        jitter = wait_seconds * jitter_ratio * random.uniform(0.0, 1.0)
        total_wait = min(max_delay, wait_seconds + jitter)

        logger.warning(
            "Tool retry scheduled: tool=%s attempt=%d/%d wait=%.2fs error=%s",
            tool_name,
            attempt,
            max_attempts,
            total_wait,
            str(attempt_result.get("error", ""))[:120] if isinstance(attempt_result, dict) else "unknown",
        )
        await asyncio.sleep(total_wait)

    if final_result is None:
        final_result = {
            "error": f"Tool execution failed after retries: {tool_name}",
            "error_code": "execution_error",
            "tool": tool_name,
        }

    _record_tool_metrics(tool_name, result=final_result, duration_ms=duration_ms)
    _record_tool_health(
        tool_name,
        success=not (isinstance(final_result, dict) and final_result.get("error")),
        duration_ms=duration_ms,
        error_code=(final_result.get("error_code") if isinstance(final_result, dict) else None),
    )
    logger.info(
        "Tool execution recorded: tool=%s success=%s duration_ms=%d",
        tool_name,
        not (isinstance(final_result, dict) and final_result.get("error")),
        duration_ms,
    )
    return final_result
