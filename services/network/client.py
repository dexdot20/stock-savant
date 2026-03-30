"""
Network Services - Unified HTTP Client, Proxy Management, and Retry Logic

This module consolidates all network operations in one place (DRY/KISS principle):
- HTTP Client: Request management, timeout/retry mechanisms
- Proxy Manager: SOCKS5 proxy management
- Retry Decorators: Retry mechanism with fixed backoff

MERGED FROM:
- client.py (HTTPClient)
- proxy.py (ProxyManager)
- retry.py (retry_fixed decorator)

Features:
- Detailed logging (recorded to log.txt)
- Automatic error management and reporting
- Rate limiting and retry mechanism
"""

from __future__ import annotations

import time
import asyncio
import random
import functools
import ssl
from datetime import datetime
from typing import Any, Dict, Optional, List, Callable
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.exceptions import (
    RequestException,
    ConnectionError,
    Timeout,
    HTTPError,
    TooManyRedirects,
    InvalidURL,
)

from core import (
    get_standard_logger,
    PaymentRequiredError,
    RateLimitError,
    APIError,
)
from core.paths import resolve_path, get_app_root
from config import DEFAULT_USER_AGENTS

_log = get_standard_logger(__name__)

# ───────────────────────────────
# RETRY DECORATORS
# ───────────────────────────────


def retry_fixed(max_attempts: int = 3, delay: float = 1.0):
    """Plain retry decorator with fixed delay.

    Args:
        max_attempts: Total number of attempts.
        delay: Wait time between attempts (seconds).
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__qualname__
            last_exc: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    _log.debug(
                        "Retry attempt %d/%d: %s", attempt, max_attempts, func_name
                    )
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        _log.info(
                            "Retry successful - %s (attempt %d/%d)",
                            func_name,
                            attempt,
                            max_attempts,
                        )
                    return result
                except PaymentRequiredError:
                    # Do not retry for 402 Payment Required
                    _log.warning("PaymentRequiredError - skipping retry: %s", func_name)
                    raise
                except Exception as e:
                    last_exc = e
                    if attempt == max_attempts:
                        _log.error(
                            "Retry failed - %s all attempts exhausted (%d/%d): %s",
                            func_name,
                            attempt,
                            max_attempts,
                            str(e)[:200],
                        )
                        break
                    _log.debug(
                        "Retry attempt %d failed: %s. Waiting %.2fs before retry",
                        attempt,
                        str(e)[:100],
                        delay,
                    )
                    time.sleep(delay)
            if last_exc:
                raise last_exc
            raise Exception("Retry failed")

        return wrapper

    return decorator


def retry_smart(max_attempts: int = 3, base_delay: float = 2.0):
    """Smart retry decorator.
    For rate limit errors (RateLimitError), waits for the specified duration.
    For other errors, applies exponential backoff.
    APIError is not retried since it's thrown after all fallback models have been tried —
    retrying would just repeat the same models from the beginning.
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__qualname__
            last_exc: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except (PaymentRequiredError, APIError, KeyboardInterrupt):
                    raise
                except RateLimitError as e:
                    last_exc = e
                    if attempt == max_attempts:
                        break

                    wait_time = e.retry_after or (base_delay * (2 ** (attempt - 1)))
                    # Safety limit: no more than 120 seconds wait
                    wait_time = min(float(wait_time), 120.0)

                    _log.warning(
                        "Rate limit exceeded (%s), waiting %.1f seconds... (Attempt %d/%d)",
                        func_name,
                        wait_time,
                        attempt,
                        max_attempts,
                    )
                    time.sleep(wait_time)
                except Exception as e:
                    last_exc = e
                    if attempt == max_attempts:
                        break

                    wait_time = base_delay * (2 ** (attempt - 1))
                    wait_time = min(float(wait_time), 60.0)  # Normal errors max 60s

                    _log.debug(
                        "Error occurred (%s): %s. Retrying after %.1f seconds. (Attempt %d/%d)",
                        func_name,
                        str(e)[:100],
                        wait_time,
                        attempt,
                        max_attempts,
                    )
                    time.sleep(wait_time)

            if last_exc:
                raise last_exc
            raise Exception("Retry failure")

        return wrapper

    return decorator


def retry_smart_async(max_attempts: int = 3, base_delay: float = 2.0):
    """Smart ASYNC retry decorator.
    APIError is not retried since it's thrown after all fallback models have been tried —
    retrying would just repeat the same models from the beginning.
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            func_name = func.__qualname__
            last_exc: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except (
                    PaymentRequiredError,
                    APIError,
                    KeyboardInterrupt,
                    asyncio.CancelledError,
                ):
                    raise
                except RateLimitError as e:
                    last_exc = e
                    if attempt == max_attempts:
                        break

                    wait_time = e.retry_after or (base_delay * (2 ** (attempt - 1)))
                    wait_time = min(float(wait_time), 120.0)

                    _log.warning(
                        "Rate limit exceeded (%s), waiting %.1f seconds (ASYNC)... (Attempt %d/%d)",
                        func_name,
                        wait_time,
                        attempt,
                        max_attempts,
                    )
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    last_exc = e
                    if attempt == max_attempts:
                        break

                    wait_time = base_delay * (2 ** (attempt - 1))
                    wait_time = min(float(wait_time), 60.0)

                    _log.debug(
                        "Error occurred (%s): %s. Async retrying after %.1f seconds. (Attempt %d/%d)",
                        func_name,
                        str(e)[:100],
                        wait_time,
                        attempt,
                        max_attempts,
                    )
                    await asyncio.sleep(wait_time)

            if last_exc:
                raise last_exc
            raise Exception("Retry failure (Async)")

        return wrapper

    return decorator


# ───────────────────────────────
# PROXY MANAGER
# ───────────────────────────────

# Proxy step types for step-specific proxy control
PROXY_STEP_MARKET_DATA = "market_data"
PROXY_STEP_NEWS_SEARCH = "news_search"
PROXY_STEP_CONTENT_EXTRACTION = "content_extraction"


class ProxyManager:
    """Simple class for SOCKS5 proxy management.

    Step-specific proxy control:
        - market_data: Proxy usage for YFinance API calls
        - news_search: Proxy usage for Google News search operations
        - content_extraction: Proxy usage for article content extraction

    From .env file for step-based control:
        PROXY_ENABLED=true          # Enable proxy globally
        PROXY_MARKET_DATA=true      # Proxy for market data
        PROXY_NEWS_SEARCH=true      # Proxy for news search
        PROXY_CONTENT_EXTRACTION=true  # Proxy for content extraction
    """

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, step: Optional[str] = None
    ):
        self.config = config or {}
        self.logger = get_standard_logger(__name__)
        self.step = step  # Which step this is used for

        # Basic settings
        self._global_enabled = self.config.get("enabled", False)
        self.proxy_file = resolve_path(
            self.config.get("file", "proxies.txt"), get_app_root()
        )
        self.max_failures = self.config.get("max_failures", 3)

        # Step-based proxy settings - read default values from config for all stages
        self._step_settings = {
            PROXY_STEP_MARKET_DATA: self.config.get("market_data", True),
            PROXY_STEP_NEWS_SEARCH: self.config.get("news_search", True),
            PROXY_STEP_CONTENT_EXTRACTION: self.config.get("content_extraction", True),
        }

        # Calculate enabled status
        self.enabled = self._calculate_enabled()

        # Proxy list and status
        self.proxies: List[str] = []
        self.failed_proxies: Dict[str, int] = {}
        self.last_load_time = 0
        self.load_interval = 300  # 5 minutes

        # Log detailed configuration
        step_info = f", Step: {self.step}" if self.step else ""
        stage_settings = (
            ", ".join([f"{k}={v}" for k, v in self._step_settings.items()])
            if self._global_enabled
            else "disabled"
        )
        self.logger.info(
            "ProxyManager initialized - Global: %s, File: %s, Stages: [%s]%s",
            self._global_enabled,
            self.proxy_file,
            stage_settings,
            step_info,
        )

        # Initial loading
        if self.enabled:
            self._load_proxies()

    def _calculate_enabled(self) -> bool:
        """Calculate whether proxy is enabled.

        Logic:
        1. If global enabled False -> proxy disabled
        2. If step is specified and that step is False -> proxy disabled
        3. Otherwise -> proxy enabled
        """
        if not self._global_enabled:
            return False

        if self.step and self.step in self._step_settings:
            return self._step_settings[self.step]

        return True

    def is_step_enabled(self, step: str) -> bool:
        """Returns whether proxy is enabled for a specific step."""
        if not self._global_enabled:
            return False
        return self._step_settings.get(step, True)

    def _load_proxies(self) -> None:
        """Load proxies from file. Supports different proxy formats (HTTP, SOCKS4/5)."""
        start_time = datetime.now()

        try:
            proxy_path = Path(self.proxy_file)
            if not proxy_path.exists():
                self.logger.warning("Proxy file not found: %s", self.proxy_file)
                return

            with open(proxy_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            valid_proxies = []
            invalid_count = 0
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parsed_proxy = self._parse_proxy_line(line)
                if parsed_proxy:
                    valid_proxies.append(parsed_proxy)
                else:
                    invalid_count += 1
                    self.logger.debug("Invalid proxy format: %s", line)

            if not valid_proxies:
                self.logger.warning(
                    "No valid proxies found in file (File: %s)", self.proxy_file
                )
                return

            self.proxies = valid_proxies
            load_duration = (datetime.now() - start_time).total_seconds()

            self.logger.info(
                "Proxy loading completed - Valid: %d, Invalid: %d, Duration: %.3fs",
                len(self.proxies),
                invalid_count,
                load_duration,
            )

        except Exception as exc:
            self.logger.error("Proxy loading error: %s", exc, exc_info=True)

    def _parse_proxy_line(self, line: str) -> Optional[str]:
        """Parse proxy line and convert to standard URL format.

        Supported formats:
        1. protocol://user:pass@host:port (Full format)
        2. protocol://host:port
        3. host:port:user:pass (Common list format - LunaProxy, Bright Data etc.)
        4. user:pass:host:port (Some providers)
        5. user:pass@host:port (Protocol-less auth)
        6. host:port (Simple format)

        Default protocol: socks5h (to prevent DNS leaks)
        """
        line = line.strip()
        if not line:
            return None

        # 1. Already in full format (contains protocol)
        if "://" in line:
            # Standardization: socks5 -> socks5h (remote dns)
            if line.startswith("socks5://"):
                return line.replace("socks5://", "socks5h://", 1)
            return line

        # 2. Format containing @: user:pass@host:port
        if "@" in line:
            # If protocol is not at beginning, add default (socks5h: remote DNS support)
            return f"socks5h://{line}"

        # 3. Colon-based splitting
        # We don't limit splitting because colons can appear in usernames (rare but possible)
        # However, residential proxies typically use host:port:user:pass format.
        parts = [p.strip() for p in line.split(":")]

        # host:port:user:pass OR user:pass:host:port
        if len(parts) == 4:
            p1, p2, p3, p4 = parts
            # Check which is the port (usually numeric)
            if p2.isdigit():  # host:port:user:pass (Residential/LunaProxy standard)
                host, port, user, pw = p1, p2, p3, p4
                return f"socks5h://{user}:{pw}@{host}:{port}"
            elif p4.isdigit():  # user:pass:host:port
                user, pw, host, port = p1, p2, p3, p4
                return f"socks5h://{user}:{pw}@{host}:{port}"
            else:
                # Even if port is not numeric, try first version (host:port:user:pass is most common)
                return f"socks5h://{p3}:{p4}@{p1}:{p2}"

        # host:port format
        if len(parts) == 2:
            host, port = parts
            # If port is numeric, it's host:port, otherwise user:pass (or error)
            if port.isdigit():
                return f"socks5h://{host}:{port}"

        return None

    def _should_reload_proxies(self) -> bool:
        """Check whether the proxy file needs to be reloaded."""
        return time.time() - self.last_load_time > self.load_interval

    def get_proxy(self) -> Optional[str]:
        """Return a randomly selected working SOCKS5 proxy."""
        if not self.enabled:
            return None

        # Periodic reload check
        if self._should_reload_proxies():
            self._load_proxies()
            self.last_load_time = time.time()

        if not self.proxies:
            return None

        # Filter out failed proxies
        available_proxies = [
            proxy
            for proxy in self.proxies
            if self.failed_proxies.get(proxy, 0) < self.max_failures
        ]

        if not available_proxies:
            self.logger.warning("No available SOCKS5 proxies remaining")
            return None

        # Random selection
        selected_proxy = random.choice(available_proxies)
        return selected_proxy

    def mark_proxy_failed(self, proxy: str) -> None:
        """Mark a proxy as failed."""
        self.failed_proxies[proxy] = self.failed_proxies.get(proxy, 0) + 1

        failure_count = self.failed_proxies[proxy]
        if failure_count >= self.max_failures:
            self.logger.warning("Proxy disabled: %s", proxy)

    def mark_proxy_success(self, proxy: str) -> None:
        """Mark a proxy as successful and reset its failure counter."""
        if proxy in self.failed_proxies:
            del self.failed_proxies[proxy]

    def get_proxy_stats(self) -> Dict[str, Any]:
        """Return basic proxy statistics."""
        total_proxies = len(self.proxies)
        failed_count = len(
            [
                p
                for p in self.proxies
                if self.failed_proxies.get(p, 0) >= self.max_failures
            ]
        )
        available_count = total_proxies - failed_count

        return {
            "enabled": self.enabled,
            "total_proxies": total_proxies,
            "available_proxies": available_count,
            "failed_proxies": failed_count,
        }

    @staticmethod
    def format_proxy_for_requests(proxy: str) -> Dict[str, str]:
        """
        Return proxy in requests library format.

        Centralized proxy formatting (DRY principle).
        Supports SOCKS4/5 and HTTP/HTTPS proxies.

        Args:
            proxy: Proxy URL (socks5://ip:port, http://ip:port, etc.)

        Returns:
            Dict with 'http' and 'https' keys for requests library
        """
        # Support standard protocols
        return {"http": proxy, "https": proxy}

    @staticmethod
    def normalize_proxy_for_aiohttp(proxy: str) -> str:
        """Normalize proxy URL for aiohttp.

        aiohttp_socks expects socks5/socks4. Converts socks5h to socks5.
        """
        if proxy.startswith("socks5h://"):
            return proxy.replace("socks5h://", "socks5://", 1)
        return proxy


# ───────────────────────────────
# HTTP Client
# ───────────────────────────────


class HTTPClient:
    """Simple, centralized HTTP client.

    Notes:
    - KISS/YAGNI: Minimal feature set; only necessary methods.
    - DRY: Retry and timeout logic centralized.
    - Detailed logging: All requests are logged to log.txt.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limit_delay: float = 0.1,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay
        self.logger = get_standard_logger(__name__)

        # Request statistics
        self._request_count = 0
        self._success_count = 0
        self._failure_count = 0

        # Fallback timeout values (KISS: simple but effective)
        self.fallback_timeouts = self._build_fallback_timeouts(timeout)

        # Create session with SSL context (prevents scraping SSL errors)
        self.session = requests.Session()
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = True  # Keep hostname verification enabled
        ssl_context.verify_mode = ssl.CERT_REQUIRED  # Keep certificate verification enabled
        adapter = requests.adapters.HTTPAdapter(ssl_context=ssl_context)
        self.session.mount("https://", adapter)
        self.session.mount("http://", requests.adapters.HTTPAdapter())

        # Default headers - prevents API calls from being flagged as bot
        self.default_headers = self._get_modern_headers()

        self.logger.info(
            "HTTPClient started - Timeout: %ds, MaxRetries: %d, RetryDelay: %.1fs",
            timeout,
            max_retries,
            retry_delay,
        )

    def _build_fallback_timeouts(self, timeout: int) -> List[int]:
        base_timeout = max(1, int(timeout))
        candidates = [max(base_timeout // 2, 1), max(base_timeout // 4, 1), 5]
        fallback_timeouts: List[int] = []
        seen = {base_timeout}

        for candidate in candidates:
            safe_candidate = max(1, int(candidate))
            if safe_candidate in seen:
                continue
            seen.add(safe_candidate)
            fallback_timeouts.append(safe_candidate)

        return fallback_timeouts

    def _get_modern_headers(self, url: Optional[str] = None) -> Dict[str, str]:
        """
        Generate modern browser headers (Client Hints, Sec-Fetch, etc.) and Referer info.
        """
        ua = random.choice(DEFAULT_USER_AGENTS)

        # Basic modern headers
        headers = {
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "DNT": "1",
        }

        # Client Hints (for Chromium-based user agents)
        if "Chrome" in ua:
            headers.update(
                {
                    "Sec-Ch-Ua": '"Google Chrome";v="130", "Chromium";v="130", "Not?A_Brand";v="99"',
                    "Sec-Ch-Ua-Mobile": "?0",
                    "Sec-Ch-Ua-Platform": (
                        '"Windows"'
                        if "Windows" in ua
                        else '"macOS"' if "Mac" in ua else '"Linux"'
                    ),
                }
            )

        # Dynamic Referer management
        if url:
            domain = urlparse(url).netloc
            if domain:
                # Choose random referer (search engine or home page)
                referers = [
                    f"https://{domain}/",
                    "https://www.google.com/",
                    "https://www.bing.com/",
                    "https://duckduckgo.com/",
                ]
                headers["Referer"] = random.choice(referers)
                headers["Sec-Fetch-Site"] = (
                    "cross-site" if "google" in headers["Referer"] else "same-origin"
                )

        return headers

    # ───────────────────────────────
    # JSON GET
    # ───────────────────────────────
    def get_json(
        self, url: str, headers: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Make a GET request, trying fallback timeouts on timeout."""
        self._request_count += 1
        self.logger.debug("HTTP GET starting: %s", url[:100])

        # Generate dynamic modern headers
        merged_headers = self._get_modern_headers(url)
        if headers:
            merged_headers.update(headers)

        result = self._request_with_fallback("GET", url, headers=merged_headers)

        if result is not None:
            self._success_count += 1
            self.logger.debug("HTTP GET successful: %s", url[:100])
        else:
            self._failure_count += 1
            self.logger.warning("HTTP GET failed: %s", url[:100])

        return result

    def _request_with_fallback(
        self, method: str, url: str, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Make request with fallback timeouts only for timeout-driven failures."""
        start_time = datetime.now()

        # Try with main timeout
        result, allow_fallback = self._make_request(
            method, url, self.timeout, **kwargs
        )
        if result is not None or not allow_fallback:
            return result

        # Try with fallback timeouts
        for i, fallback_timeout in enumerate(self.fallback_timeouts):
            self.logger.info(
                "Attempting fallback timeout %d/%d (%ds): %s",
                i + 1,
                len(self.fallback_timeouts),
                fallback_timeout,
                url[:80],
            )
            result, allow_fallback = self._make_request(
                method, url, fallback_timeout, **kwargs
            )
            if result is not None:
                elapsed = (datetime.now() - start_time).total_seconds()
                self.logger.info(
                    "Fallback successful - %s (timeout: %ds, total time: %.2fs)",
                    url[:60],
                    fallback_timeout,
                    elapsed,
                )
                return result
            if not allow_fallback:
                break

        elapsed = (datetime.now() - start_time).total_seconds()
        self.logger.error(
            "All timeout values failed: %s (total time: %.2fs)",
            url[:80],
            elapsed,
        )
        return None

    def _make_request(
        self, method: str, url: str, timeout: int, **kwargs
    ) -> tuple[Optional[Dict[str, Any]], bool]:
        """Make a request with a single timeout value.

        Returns:
            tuple[result, allow_shorter_timeout_fallback]
        """
        retries = self.max_retries
        for attempt in range(retries):
            attempt_start = datetime.now()
            try:
                self.logger.debug(
                    "%s request attempt %d/%d - URL: %s, Timeout: %ds",
                    method,
                    attempt + 1,
                    retries,
                    url[:80],
                    timeout,
                )

                if method == "GET":
                    resp: requests.Response = self.session.get(
                        url, timeout=timeout, **kwargs
                    )
                else:  # POST
                    resp: requests.Response = self.session.post(
                        url, timeout=timeout, **kwargs
                    )

                # Rate limit check
                if resp.status_code == 429:
                    wait_time: float = max(self.retry_delay, 1.0) * (attempt + 1) * 2
                    self.logger.warning(
                        "Rate limit (429) received - URL: %s, Waiting: %.1fs",
                        url[:60],
                        wait_time,
                    )
                    time.sleep(wait_time)
                    continue

                # Special handling for 403 Forbidden error
                if resp.status_code == 403:
                    self.logger.error(
                        "403 Forbidden - API access denied: %s (attempt %d/%d)",
                        url[:60],
                        attempt + 1,
                        retries,
                    )
                    # Refresh entire header set and retry
                    if attempt < retries - 1:
                        new_headers = self._get_modern_headers(url)
                        if "headers" in kwargs:
                            new_headers.update(kwargs["headers"])
                        kwargs["headers"] = new_headers

                        self.logger.info("Header set refreshed, retrying")
                        time.sleep(self.retry_delay * 2)  # Longer wait
                        continue
                    else:
                        return None, False

                # Special handling for 402 Payment Required error
                if resp.status_code == 402:
                    self.logger.error(
                        "402 Payment Required - Payment required for API: %s", url[:60]
                    )
                    raise PaymentRequiredError(
                        "Payment required for API. Please check your API key.",
                        context="HTTPClient._make_request",
                        details={"url": url[:100], "status_code": 402},
                    )

                resp.raise_for_status()

                # Parse JSON
                data = resp.json()
                attempt_duration = (datetime.now() - attempt_start).total_seconds()

                self.logger.debug(
                    "%s successful - URL: %s, Status: %d, Time: %.3fs",
                    method,
                    url[:60],
                    resp.status_code,
                    attempt_duration,
                )

                time.sleep(self.rate_limit_delay)
                return data, False

            except Timeout:
                attempt_duration = (datetime.now() - attempt_start).total_seconds()
                if attempt == retries - 1:
                    self.logger.warning(
                        "Timeout (%ds) all attempts failed: %s (time: %.2fs)",
                        timeout,
                        url[:60],
                        attempt_duration,
                    )
                    return None, True
                self.logger.debug(
                    "Timeout attempt %d/%d: %s", attempt + 1, retries, url[:60]
                )
                time.sleep(self.retry_delay)

            except (ConnectionError, HTTPError) as e:
                attempt_duration = (datetime.now() - attempt_start).total_seconds()
                if attempt == retries - 1:
                    self.logger.error(
                        "%s connection/HTTP error all attempts failed: %s - %s (time: %.2fs)",
                        method,
                        url[:60],
                        str(e)[:100],
                        attempt_duration,
                    )
                    return None, False
                self.logger.debug(
                    "Connection error attempt %d/%d: %s",
                    attempt + 1,
                    retries,
                    str(e)[:80],
                )
                time.sleep(self.retry_delay)

            except (TooManyRedirects, InvalidURL) as e:
                # These errors shouldn't be retried
                self.logger.error(
                    "%s URL/redirect error: %s - %s", method, url[:60], str(e)[:100]
                )
                return None, False

            except RequestException as e:
                if attempt == retries - 1:
                    self.logger.error(
                        "%s request error all attempts failed: %s - %s",
                        method,
                        url[:60],
                        str(e)[:100],
                    )
                    return None, False
                self.logger.debug(
                    "Request error attempt %d/%d: %s", attempt + 1, retries, str(e)[:80]
                )
                time.sleep(self.retry_delay)

            except (ValueError, KeyError) as e:
                # JSON parsing or invalid parameter errors
                if attempt == retries - 1:
                    self.logger.error(
                        "JSON parse error all attempts failed: %s - %s",
                        url[:60],
                        str(e)[:100],
                    )
                    return None, False
                self.logger.debug(
                    "JSON parse error attempt %d/%d: %s",
                    attempt + 1,
                    retries,
                    str(e)[:80],
                )
                time.sleep(self.retry_delay)

        return None, False

    # ───────────────────────────────
    # JSON POST
    # ───────────────────────────────
    def post_json(
        self,
        url: str,
        json_payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make a POST request, trying fallback timeouts on timeout."""
        # Merge standard headers
        if headers:
            merged_headers = {**self.default_headers, **headers}
        else:
            merged_headers = self.default_headers.copy()

        result = self._request_with_fallback(
            "POST", url, json=json_payload, headers=merged_headers
        )
        return result if result is not None else {}


# ───────────────────────────────
# Network Utility Functions
# ───────────────────────────────


def create_http_client_from_config() -> HTTPClient:
    """Create HTTP client from configuration."""
    try:
        from config import get_config

        config = get_config()

        # Get values from network configuration
        network_cfg = config.get("network", {})
        general_timeout = network_cfg.get("request_timeout_seconds", 30)
        general_max_retries = network_cfg.get("max_retry_count", 3)
        general_retry_delay = network_cfg.get("retry_delay_seconds", 1.0)

        return HTTPClient(
            timeout=general_timeout,
            max_retries=general_max_retries,
            retry_delay=general_retry_delay,
            rate_limit_delay=network_cfg.get("rate_limit_delay", 0.1),
        )
    except Exception as e:
        _log.warning(
            f"Could not create HTTP client from config, using default values: {e}"
        )
        return HTTPClient()


# ───────────────────────────────
# PUBLIC API
# ───────────────────────────────

__all__ = [
    "HTTPClient",
    "ProxyManager",
    "retry_fixed",
    "retry_smart",
    "retry_smart_async",
    "create_http_client_from_config",
    "PROXY_STEP_MARKET_DATA",
    "PROXY_STEP_NEWS_SEARCH",
    "PROXY_STEP_CONTENT_EXTRACTION",
]
