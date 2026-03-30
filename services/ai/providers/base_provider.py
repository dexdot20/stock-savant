"""Provider common helpers."""

import json
import asyncio
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import aiohttp

from core import APIError, RateLimitError
from core.logging import get_ai_debug_logger, ai_debug_enabled


class BaseAIProvider(ABC):
    """Abstract AI provider contract for sync and async completion requests."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    def call(
        self,
        prompt: List[Dict[str, str]],
        request_type: str,
        provider_name: str,
        *,
        context: "ProviderContext",
        model_override: Optional[str] = None,
        timeout_override: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def call_async(
        self,
        prompt: List[Dict[str, str]],
        request_type: str,
        provider_name: str,
        *,
        context: "ProviderContext",
        model_override: Optional[str] = None,
        timeout_override: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        pass


@dataclass
class ProviderContext:
    """Common context used in provider calls."""

    logger: Any
    providers_cfg: Dict[str, Any]
    model_configs: Dict[str, Any]
    request_timeout: float

    def log_response(
        self, response_dict: Dict[str, Any], request_type: str, model: str
    ) -> None:
        """Log AI response for debugging."""
        self.logger.debug(
            "AI response: type=%s, model=%s, len=%d",
            request_type,
            model,
            len(str(response_dict.get("content", ""))),
        )


def post_provider_request(
    provider_label: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json_payload: Optional[Dict[str, Any]] = None,
    timeout_override: Optional[float] = None,
    request_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Async HTTP wrapper for synchronous use."""
    return _run_async_request(
        post_provider_request_async(
            provider_label,
            url,
            headers=headers,
            params=params,
            json_payload=json_payload,
            timeout_override=timeout_override,
            request_timeout=request_timeout,
        )
    )


async def post_provider_request_async(
    provider_label: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json_payload: Optional[Dict[str, Any]] = None,
    timeout_override: Optional[float] = None,
    request_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    timeout_seconds = timeout_override or request_timeout or 30
    if timeout_seconds <= 0:
        timeout_seconds = 30

    debug_logger = None
    if ai_debug_enabled():
        debug_logger = get_ai_debug_logger()
        _log_raw_api_request(
            debug_logger, provider_label, url, headers, json_payload, timeout_seconds
        )

    _MAX_CONNECT_ATTEMPTS = 2
    _CONNECT_RETRY_DELAY = 1.0

    for attempt in range(1, _MAX_CONNECT_ATTEMPTS + 1):
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            async with aiohttp.ClientSession(
                timeout=timeout, max_line_size=32768, max_field_size=32768
            ) as session:
                async with session.post(
                    url,
                    headers=headers,
                    params=params,
                    json=json_payload,
                ) as response:
                    status_code = response.status
                    response_headers = dict(response.headers)
                    response_text = await response.text()

                    if debug_logger:
                        _log_raw_api_response_payload(
                            debug_logger,
                            provider_label,
                            status_code,
                            response_headers,
                            response_text,
                        )

                    if status_code >= 400:
                        message = extract_http_error_from_payload(
                            response_text, f"HTTP {status_code}"
                        )
                        if debug_logger:
                            _log_raw_api_error(
                                debug_logger, provider_label, "HTTPError", message
                            )

                        if status_code == 429:
                            retry_after = response_headers.get("Retry-After")
                            retry_after_val = None
                            if retry_after and retry_after.isdigit():
                                retry_after_val = int(retry_after)
                            raise RateLimitError(
                                f"{provider_label} rate limit exceeded: {message}",
                                retry_after=retry_after_val,
                            )

                        raise APIError(
                            f"{provider_label} HTTP error: {message}",
                            status_code=status_code,
                        )

                    try:
                        return await response.json()
                    except (aiohttp.ContentTypeError, json.JSONDecodeError) as exc:
                        if debug_logger:
                            _log_raw_api_error(
                                debug_logger,
                                provider_label,
                                "JSONDecodeError",
                                str(exc),
                            )
                        raise APIError(
                            f"{provider_label} JSON response could not be parsed: {exc}"
                        )

        except (RateLimitError, APIError):
            raise

        except asyncio.TimeoutError:  # pragma: no cover - network error tracking
            detail = f"timeout ({timeout_seconds:.0f}s)"
            if debug_logger:
                _log_raw_api_error(debug_logger, provider_label, "TimeoutError", detail)
            if attempt < _MAX_CONNECT_ATTEMPTS:
                await asyncio.sleep(_CONNECT_RETRY_DELAY)
                continue
            raise APIError(
                f"{provider_label} connection timeout ({timeout_seconds:.0f}s)"
            )

        except aiohttp.ClientError as exc:  # pragma: no cover - network error tracking
            detail = str(exc) or repr(exc) or type(exc).__name__
            if debug_logger:
                _log_raw_api_error(
                    debug_logger, provider_label, "ConnectionError", detail
                )
            if attempt < _MAX_CONNECT_ATTEMPTS:
                await asyncio.sleep(_CONNECT_RETRY_DELAY)
                continue
            raise APIError(f"{provider_label} connection error: {detail}")


def _run_async_request(coro: Any) -> Dict[str, Any]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # If a loop is already running, we have a problem because we can't call asyncio.run()
    # To prevent 'coroutine was never awaited' warning, we must close/dispose the coro
    try:
        coro.close()
    except (AttributeError, RuntimeError):
        pass

    raise APIError(
        "Active event loop detected; post_provider_request_async should be used."
    )


def _log_raw_api_request(
    debug_logger: Any,
    provider_label: str,
    url: str,
    headers: Optional[Dict[str, str]],
    json_payload: Optional[Dict[str, Any]],
    timeout: float,
) -> None:
    """Log raw API request sent to the provider."""
    separator = "=" * 100
    debug_logger._logger.info(separator)
    debug_logger._logger.info("[RAW API REQUEST] >>> SENDING DATA >>>")
    debug_logger._logger.info("Provider: %s", provider_label)
    debug_logger._logger.info("URL: %s", url)
    debug_logger._logger.info("Timeout: %ss", timeout)

    # Headers (API key'i maskele)
    if headers:
        safe_headers = {}
        for k, v in headers.items():
            if "authorization" in k.lower() or "api" in k.lower() or "key" in k.lower():
                safe_headers[k] = v[:20] + "..." if len(v) > 20 else "[MASKED]"
            else:
                safe_headers[k] = v
        debug_logger._logger.info(
            f"Headers: {json.dumps(safe_headers, ensure_ascii=False, indent=2)}"
        )

    # Full JSON payload - ALL DATA HERE
    if json_payload:
        debug_logger._logger.info("--- FULL REQUEST PAYLOAD (ALL DATA SENT TO AI) ---")
        debug_logger._logger.info(">>>")
        try:
            payload_str = json.dumps(
                json_payload, ensure_ascii=False, indent=2, default=str
            )
            for line in payload_str.split("\n"):
                debug_logger._logger.info("    %s", line)
        except Exception as e:
            debug_logger._logger.info("    [Payload serialize edilemedi: %s]", e)
            debug_logger._logger.info("    %s", str(json_payload))
        debug_logger._logger.info("<<<")
        debug_logger._logger.info("--- REQUEST PAYLOAD END ---")

    debug_logger._logger.info(separator)


def _log_raw_api_response_payload(
    debug_logger: Any,
    provider_label: str,
    status_code: int,
    headers: Dict[str, Any],
    response_text: str,
) -> None:
    """Log raw API response (aiohttp) from the provider."""
    separator = "=" * 100
    debug_logger._logger.info(separator)
    debug_logger._logger.info("[RAW API RESPONSE] <<< RECEIVED DATA <<<")
    debug_logger._logger.info("Provider: %s", provider_label)
    debug_logger._logger.info("Status Code: %s", status_code)
    debug_logger._logger.info("Response Headers: %s", headers)

    debug_logger._logger.info("--- FULL RESPONSE BODY (ALL DATA RECEIVED FROM AI) ---")
    debug_logger._logger.info(">>>")
    try:
        response_json = json.loads(response_text)
        response_str = json.dumps(
            response_json, ensure_ascii=False, indent=2, default=str
        )
        for line in response_str.split("\n"):
            debug_logger._logger.info("    %s", line)
    except Exception:
        debug_logger._logger.info("    [Raw Text Response]")
        for line in response_text.split("\n"):
            debug_logger._logger.info("    %s", line)
    debug_logger._logger.info("<<<")
    debug_logger._logger.info("--- RESPONSE BODY END ---")
    debug_logger._logger.info(separator)


def _log_raw_api_error(
    debug_logger: Any,
    provider_label: str,
    error_type: str,
    error_message: str,
) -> None:
    """Log API error details."""
    separator = "!" * 100
    debug_logger._logger.info(separator)
    debug_logger._logger.info("[RAW API ERROR]")
    debug_logger._logger.info("Provider: %s", provider_label)
    debug_logger._logger.info("Error Type: %s", error_type)
    debug_logger._logger.info("Error Message: %s", error_message)
    debug_logger._logger.info(separator)


def extract_http_error_from_payload(payload_text: str, fallback: str) -> str:
    try:
        payload = json.loads(payload_text)
    except Exception:
        return fallback

    if isinstance(payload, dict):
        error_section = payload.get("error")
        if isinstance(error_section, dict):
            message = error_section.get("message")
            if message:
                return str(message)
    return fallback


def infer_provider_from_model_name(model_name: Optional[str]) -> Optional[str]:
    if not model_name or not isinstance(model_name, str):
        return None

    lowered = model_name.lower()
    if "deepseek" in lowered:
        return "deepseek"

    if lowered in {"deepseek-chat", "deepseek-reasoner"}:
        return "deepseek"

    return None
