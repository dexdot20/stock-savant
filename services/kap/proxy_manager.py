from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from core import get_standard_logger

from .config import PROXY_ERROR_THRESHOLD, PROXY_FILE_PATH, PROXY_RESET_INTERVAL

logger = get_standard_logger(__name__)


def _mask_proxy_url(proxy_url: Optional[str]) -> str:
    if not proxy_url:
        return "direct"
    if "@" not in proxy_url:
        return proxy_url

    try:
        if "://" in proxy_url:
            scheme, rest = proxy_url.split("://", 1)
            if "@" in rest:
                _, host_port = rest.rsplit("@", 1)
                return f"{scheme}://<hidden>:<hidden>@{host_port}"
        return proxy_url
    except Exception:
        return "<error>"


@dataclass
class _ProxyEntry:
    url: str
    error_count: int = 0
    disabled_at: Optional[float] = field(default=None)

    @property
    def is_active(self) -> bool:
        return self.disabled_at is None

    def try_reset(self, reset_interval: float) -> None:
        if self.disabled_at is not None:
            if (time.monotonic() - self.disabled_at) >= reset_interval:
                self.error_count = 0
                self.disabled_at = None


class ProxyManager:
    def __init__(
        self,
        proxy_file: Path = PROXY_FILE_PATH,
        error_threshold: int = PROXY_ERROR_THRESHOLD,
        reset_interval: float = PROXY_RESET_INTERVAL,
    ) -> None:
        self._lock = threading.Lock()
        self._threshold = error_threshold
        self._reset_interval = reset_interval
        self._proxies: list[_ProxyEntry] = []
        self._index = 0
        self._load(proxy_file)

    def _load(self, proxy_file: Path) -> None:
        if not proxy_file.exists():
            logger.warning("KAP proxy file not found: %s", proxy_file)
            return
        for line in proxy_file.read_text(encoding="utf-8").splitlines():
            url = line.strip()
            if url and not url.startswith("#"):
                self._proxies.append(_ProxyEntry(url=url))
        logger.info("KAP proxy pool loaded: %d proxies (%s)", len(self._proxies), proxy_file)

    def _reset_eligible(self) -> None:
        for proxy in self._proxies:
            proxy.try_reset(self._reset_interval)

    def get_next(self) -> Optional[str]:
        with self._lock:
            if not self._proxies:
                return None
            self._reset_eligible()
            for _ in range(len(self._proxies)):
                self._index %= len(self._proxies)
                entry = self._proxies[self._index]
                self._index += 1
                if entry.is_active:
                    return entry.url
            return None

    def mark_error(self, proxy_url: str) -> None:
        with self._lock:
            for proxy in self._proxies:
                if proxy.url != proxy_url:
                    continue
                proxy.error_count += 1
                if proxy.error_count >= self._threshold:
                    proxy.disabled_at = time.monotonic()
                    logger.warning(
                        "KAP proxy disabled: %s (%d errors, threshold=%d)",
                        _mask_proxy_url(proxy_url),
                        proxy.error_count,
                        self._threshold,
                    )
                else:
                    logger.debug(
                        "KAP proxy error recorded: %s (%d/%d)",
                        _mask_proxy_url(proxy_url),
                        proxy.error_count,
                        self._threshold,
                    )
                break

    def mark_success(self, proxy_url: str) -> None:
        with self._lock:
            for proxy in self._proxies:
                if proxy.url == proxy_url:
                    proxy.error_count = 0
                    break

    def status(self) -> list[dict]:
        with self._lock:
            self._reset_eligible()
            result = []
            for proxy in self._proxies:
                entry = {
                    "url": proxy.url,
                    "active": proxy.is_active,
                    "error_count": proxy.error_count,
                }
                if proxy.disabled_at is not None:
                    elapsed = time.monotonic() - proxy.disabled_at
                    entry["disabled_seconds_ago"] = round(elapsed, 1)
                    entry["reset_in_seconds"] = max(
                        0.0,
                        round(self._reset_interval - elapsed, 1),
                    )
                result.append(entry)
            return result

    def __len__(self) -> int:
        return len(self._proxies)


def fetch_with_retry(
    method: str,
    url: str,
    *,
    headers: dict,
    timeout: int,
    proxy_manager: Optional[ProxyManager] = None,
    max_retries: int = 3,
    **request_kwargs,
) -> httpx.Response:
    has_proxies = proxy_manager is not None and len(proxy_manager) > 0
    attempts = max_retries if has_proxies else 1
    tried: set[Optional[str]] = set()
    last_exc: Optional[Exception] = None

    for _ in range(attempts):
        proxy_url = proxy_manager.get_next() if has_proxies else None
        if proxy_url in tried:
            if None in tried:
                break
            proxy_url = None
        tried.add(proxy_url)

        client_kwargs = {
            "headers": headers,
            "timeout": timeout,
            "follow_redirects": True,
        }
        if proxy_url:
            client_kwargs["proxy"] = proxy_url

        try:
            with httpx.Client(**client_kwargs) as client:
                response = getattr(client, method.lower())(url, **request_kwargs)
                response.raise_for_status()
            if proxy_url and proxy_manager:
                proxy_manager.mark_success(proxy_url)
            logger.debug(
                "KAP %s %s succeeded via %s",
                method.upper(),
                url,
                _mask_proxy_url(proxy_url),
            )
            return response
        except Exception as exc:
            last_exc = exc
            if proxy_url and proxy_manager:
                proxy_manager.mark_error(proxy_url)
            logger.warning(
                "KAP %s %s failed via %s: %s",
                method.upper(),
                url,
                _mask_proxy_url(proxy_url),
                exc,
            )

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("KAP request failed without an exception")