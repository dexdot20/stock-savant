"""OpenRouter provider integration."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import aiohttp

from config import is_configured_secret
from core import APIError, RateLimitError

from .base_provider import (
    BaseAIProvider,
    ProviderContext,
    post_provider_request,
    post_provider_request_async,
)
from .cache_usage import extract_prompt_cache_usage, log_prompt_cache_usage


@dataclass
class _StreamState:
    on_reasoning_delta: Optional[Callable[[str], None]] = None
    on_content_delta: Optional[Callable[[str], None]] = None
    finalizer: Optional[Callable[[], None]] = None
    reasoning_streamed: bool = False
    content_streamed: bool = False


class _ConsoleStreamRenderer:
    def __init__(self, console: Any, *, request_type: str, model: str) -> None:
        self._console = console
        self._request_type = request_type
        self._model = model
        self._started = False
        self._reasoning_started = False
        self._content_started = False

    def emit_reasoning(self, text: str) -> None:
        self._emit(text, kind="reasoning")

    def emit_content(self, text: str) -> None:
        self._emit(text, kind="content")

    def finalize(self) -> None:
        if not self._started:
            return
        try:
            self._console.file.write("\n")
            self._console.file.flush()
        except Exception:
            pass

    def _emit(self, text: str, *, kind: str) -> None:
        if not text:
            return

        try:
            if not self._started:
                header = f"\nAI Stream | {self._request_type} | {self._model}\n"
                self._console.file.write(header)
                self._started = True

            if kind == "reasoning" and not self._reasoning_started:
                self._console.file.write("Reasoning\n")
                self._reasoning_started = True

            if kind == "content" and not self._content_started:
                if self._reasoning_started:
                    self._console.file.write("\n\nAnswer\n")
                else:
                    self._console.file.write("Answer\n")
                self._content_started = True

            self._console.file.write(text)
            self._console.file.flush()
        except Exception:
            pass


def _resolve_stream_state(
    request_type: str,
    model: str,
    *,
    console: Optional[Any] = None,
    allow_reasoning_stream: bool = True,
    allow_content_stream: bool = True,
    on_reasoning_delta: Optional[Callable[[str], None]] = None,
    on_content_delta: Optional[Callable[[str], None]] = None,
) -> _StreamState:
    finalizer = None
    if console is not None:
        renderer = _ConsoleStreamRenderer(
            console,
            request_type=request_type,
            model=model,
        )
        if allow_reasoning_stream:
            on_reasoning_delta = on_reasoning_delta or renderer.emit_reasoning
        if allow_content_stream:
            on_content_delta = on_content_delta or renderer.emit_content
        if allow_reasoning_stream or allow_content_stream:
            finalizer = renderer.finalize

    return _StreamState(
        on_reasoning_delta=on_reasoning_delta,
        on_content_delta=on_content_delta,
        finalizer=finalizer,
    )


def _stream_enabled(stream_state: _StreamState) -> bool:
    return bool(stream_state.on_reasoning_delta or stream_state.on_content_delta)


def _stringify_openrouter_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (int, float, bool)):
        return str(value)

    if isinstance(value, dict):
        preferred_keys = (
            "text",
            "summary",
            "content",
            "value",
            "message",
            "output_text",
            "reasoning",
            "reasoning_content",
            "refusal",
            "arguments",
            "data",
        )
        parts: List[str] = []
        for key in preferred_keys:
            if key in value:
                text = _stringify_openrouter_value(value.get(key))
                if text:
                    parts.append(text)
        if parts:
            return "\n".join(parts)
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)

    if isinstance(value, list):
        parts = [_stringify_openrouter_value(item) for item in value]
        normalized_parts = [part for part in parts if part]
        return "\n".join(normalized_parts).strip()

    return str(value).strip()


def _stringify_openrouter_stream_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, (int, float, bool)):
        return str(value)

    if isinstance(value, dict):
        preferred_keys = (
            "text",
            "summary",
            "content",
            "value",
            "message",
            "output_text",
            "reasoning",
            "reasoning_content",
            "refusal",
            "arguments",
            "data",
        )
        parts: List[str] = []
        for key in preferred_keys:
            if key in value:
                text = _stringify_openrouter_stream_value(value.get(key))
                if text:
                    parts.append(text)
        if parts:
            return "".join(parts)
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)

    if isinstance(value, list):
        parts = [_stringify_openrouter_stream_value(item) for item in value]
        return "".join(part for part in parts if part)

    return str(value)


def _reasoning_details_to_text(details: Any, *, preserve_spacing: bool = False) -> str:
    if not isinstance(details, list):
        return ""

    stringify = (
        _stringify_openrouter_stream_value
        if preserve_spacing
        else _stringify_openrouter_value
    )
    parts: List[str] = []
    for item in details:
        if not isinstance(item, dict):
            continue
        detail_type = str(item.get("type") or "").strip().lower()
        if detail_type == "reasoning.text":
            text = stringify(item.get("text"))
        elif detail_type == "reasoning.summary":
            text = stringify(item.get("summary"))
        else:
            text = ""
        if text:
            parts.append(text)
    if preserve_spacing:
        return "".join(parts)
    return "\n".join(parts).strip()


def _emit_stream_delta(stream_state: _StreamState, delta: Dict[str, Any]) -> None:
    content_delta = _stringify_openrouter_stream_value(delta.get("content"))
    reasoning_delta = _stringify_openrouter_stream_value(delta.get("reasoning"))
    if not reasoning_delta:
        reasoning_delta = _reasoning_details_to_text(
            delta.get("reasoning_details"),
            preserve_spacing=True,
        )

    if reasoning_delta:
        stream_state.reasoning_streamed = True
        if stream_state.on_reasoning_delta:
            stream_state.on_reasoning_delta(reasoning_delta)
    if content_delta:
        stream_state.content_streamed = True
        if stream_state.on_content_delta:
            stream_state.on_content_delta(content_delta)


async def _post_openrouter_stream_request_async(
    url: str,
    *,
    headers: Dict[str, str],
    json_payload: Dict[str, Any],
    timeout_seconds: float,
    stream_state: _StreamState,
) -> Dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None

    try:
        async with aiohttp.ClientSession(
            timeout=timeout,
            max_line_size=32768,
            max_field_size=32768,
        ) as session:
            async with session.post(url, headers=headers, json=json_payload) as response:
                response_text = await response.text() if response.status >= 400 else None

                if response.status >= 400:
                    message = response_text or f"HTTP {response.status}"
                    if response.status == 429:
                        retry_after = response.headers.get("Retry-After")
                        retry_after_val = (
                            int(retry_after)
                            if retry_after and retry_after.isdigit()
                            else None
                        )
                        raise RateLimitError(
                            f"OpenRouter rate limit exceeded: {message}",
                            retry_after=retry_after_val,
                        )
                    raise APIError(
                        f"OpenRouter HTTP error: {message}",
                        status_code=response.status,
                    )

                async for raw_line in response.content:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue

                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue

                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(event.get("usage"), dict):
                        usage = event.get("usage")

                    choices = event.get("choices") or []
                    if not choices:
                        continue

                    choice = choices[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    if isinstance(delta, dict):
                        content_delta = _stringify_openrouter_stream_value(
                            delta.get("content")
                        )
                        reasoning_delta = _stringify_openrouter_stream_value(
                            delta.get("reasoning")
                        )
                        if not reasoning_delta:
                            reasoning_delta = _reasoning_details_to_text(
                                delta.get("reasoning_details"),
                                preserve_spacing=True,
                            )
                        if content_delta:
                            content_parts.append(content_delta)
                        if reasoning_delta:
                            reasoning_parts.append(reasoning_delta)
                        _emit_stream_delta(stream_state, delta)
    except (RateLimitError, APIError):
        raise
    except asyncio.TimeoutError as exc:
        raise APIError(
            f"OpenRouter connection timeout ({timeout_seconds:.0f}s)"
        ) from exc
    except aiohttp.ClientError as exc:
        detail = str(exc) or repr(exc) or type(exc).__name__
        raise APIError(f"OpenRouter connection error: {detail}") from exc
    finally:
        if stream_state.finalizer:
            stream_state.finalizer()

    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    if not content and not reasoning:
        raise APIError("OpenRouter returned empty stream response", auto_log=False)

    response: Dict[str, Any] = {
        "choices": [
            {
                "message": {
                    "content": content,
                    "reasoning": reasoning,
                },
                "finish_reason": finish_reason,
            }
        ]
    }
    response["_streaming"] = {
        "reasoning_streamed": stream_state.reasoning_streamed,
        "content_streamed": stream_state.content_streamed,
    }
    if usage:
        response["usage"] = usage
    return response


class OpenRouterProvider(BaseAIProvider):
    @property
    def provider_name(self) -> str:
        return "openrouter"

    def call(
        self,
        prompt: List[Dict[str, str]],
        request_type: str,
        provider_name: str,
        *,
        context: ProviderContext,
        model_override: Optional[str] = None,
        timeout_override: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return call_openrouter(
            prompt,
            request_type,
            provider_name,
            context=context,
            model_override=model_override,
            timeout_override=timeout_override,
            **kwargs,
        )

    async def call_async(
        self,
        prompt: List[Dict[str, str]],
        request_type: str,
        provider_name: str,
        *,
        context: ProviderContext,
        model_override: Optional[str] = None,
        timeout_override: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return await call_openrouter_async(
            prompt,
            request_type,
            provider_name,
            context=context,
            model_override=model_override,
            timeout_override=timeout_override,
            **kwargs,
        )


def _prepare_openrouter_payload(
    prompt: List[Dict[str, str]],
    request_type: str,
    context: ProviderContext,
    model_override: Optional[str] = None,
    **kwargs: Any,
) -> tuple[str, Dict[str, str], Dict[str, Any], str]:
    provider_cfg = context.providers_cfg.get("openrouter", {})
    api_key = provider_cfg.get("api_key")
    if not is_configured_secret(api_key):
        raise APIError("OpenRouter API key is not defined.")
    api_key = str(api_key).strip()

    base_url = str(provider_cfg.get("base_url") or "https://openrouter.ai/api/v1").strip()
    model = model_override or str(provider_cfg.get("default_model") or "").strip()
    if not model:
        raise APIError(
            "OpenRouter model is not configured. ai.models.<request_type>.model veya ai.providers.openrouter.default_model alanını ayarlayın."
        )

    normalized_base_url = base_url.rstrip("/")
    if normalized_base_url.endswith("/chat/completions"):
        url = normalized_base_url
    else:
        url = f"{normalized_base_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    http_referer = str(provider_cfg.get("http_referer") or "").strip()
    if http_referer:
        headers["HTTP-Referer"] = http_referer

    app_title = str(provider_cfg.get("app_title") or "").strip()
    if app_title:
        headers["X-OpenRouter-Title"] = app_title

    payload: Dict[str, Any] = {
        "model": model,
        "messages": prompt,
    }

    for param in (
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
    ):
        value = provider_cfg.get(param)
        if value is not None:
            payload[param] = value

    max_tokens = provider_cfg.get("max_tokens")
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    reasoning_cfg = provider_cfg.get("reasoning", {})
    if isinstance(reasoning_cfg, dict) and bool(reasoning_cfg.get("enabled", False)):
        if request_type == "reasoner":
            reasoning_payload: Dict[str, Any] = {"enabled": True}
            effort = reasoning_cfg.get("effort")
            if effort:
                reasoning_payload["effort"] = str(effort)
            reasoning_max_tokens = reasoning_cfg.get("max_tokens")
            if reasoning_max_tokens is not None:
                reasoning_payload["max_tokens"] = int(reasoning_max_tokens)
            exclude = reasoning_cfg.get("exclude")
            if exclude is not None:
                reasoning_payload["exclude"] = bool(exclude)
            payload["reasoning"] = reasoning_payload

    response_format = kwargs.get("response_format")
    if not (isinstance(response_format, dict) and response_format.get("type")):
        response_format = provider_cfg.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type"):
        payload["response_format"] = response_format

    prompt_caching_cfg = provider_cfg.get("prompt_caching")
    if not isinstance(prompt_caching_cfg, dict):
        prompt_caching_cfg = {}
    prompt_caching_enabled = bool(prompt_caching_cfg.get("enabled", True))

    cache_control = kwargs.get("cache_control")
    if cache_control is None:
        cache_control = prompt_caching_cfg.get("cache_control")
    if prompt_caching_enabled and isinstance(cache_control, dict) and cache_control:
        payload["cache_control"] = cache_control

    provider_preferences = kwargs.get("provider")
    if not isinstance(provider_preferences, dict):
        provider_preferences = provider_cfg.get("provider")
    if isinstance(provider_preferences, dict) and provider_preferences:
        if (
            prompt_caching_enabled
            and prompt_caching_cfg.get("preserve_sticky_routing", True)
            and "order" in provider_preferences
        ):
            provider_preferences = {
                key: value
                for key, value in provider_preferences.items()
                if key != "order"
            }
        if provider_preferences:
            payload["provider"] = provider_preferences

    for key in ("tools", "tool_choice", "parallel_tool_calls"):
        value = kwargs.get(key)
        if value is not None:
            payload[key] = value

    return url, headers, payload, model


def _process_openrouter_response(
    data: Any, context: ProviderContext, model: str, request_type: str
) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise APIError("Unexpected response received from OpenRouter.", auto_log=False)

    if "error" in data:
        error_info = data.get("error")
        if isinstance(error_info, dict):
            message = error_info.get("message", "Unknown error")
            code = error_info.get("code")
            code_text = str(code).strip()

            if code == 429 or code_text == "429" or "rate limit" in message.lower():
                raise RateLimitError(f"OpenRouter rate limit: {message}")

            raise APIError(f"OpenRouter returned error: {message}", auto_log=False)

        raise APIError(
            "OpenRouter returned error: invalid error payload",
            auto_log=False,
        )

    choices = data.get("choices", [])
    if not choices:
        context.logger.warning(
            "OpenRouter response has no choices. Data keys: %s", list(data.keys())
        )
        raise APIError(
            "OpenRouter returned empty response (no choices)",
            auto_log=False,
        )

    message = choices[0].get("message", {})
    content = _stringify_openrouter_value(message.get("content"))
    tool_calls = message.get("tool_calls")
    reasoning = _stringify_openrouter_value(message.get("reasoning"))
    if not reasoning:
        reasoning = _reasoning_details_to_text(message.get("reasoning_details"))
    if not reasoning:
        reasoning = _stringify_openrouter_value(message.get("reasoning_content"))
    refusal = _stringify_openrouter_value(message.get("refusal"))
    finish_reason = choices[0].get("finish_reason")
    streaming_meta = (
        data.get("_streaming") if isinstance(data.get("_streaming"), dict) else {}
    )

    if not content and not tool_calls and reasoning:
        context.logger.warning(
            "OpenRouter returned empty content; using reasoning fallback. finish_reason=%s, model=%s",
            finish_reason,
            model,
        )
        content = reasoning

    if not content and not tool_calls:
        diagnostic_parts = [f"Finish reason: {finish_reason}"]
        if refusal:
            diagnostic_parts.append(f"refusal={refusal[:240]}")
        elif reasoning:
            diagnostic_parts.append(f"reasoning={reasoning[:240]}")

        raise APIError(
            "OpenRouter returned empty response. " + "; ".join(diagnostic_parts),
            auto_log=False,
        )

    response_dict = {
        "content": content,
        "tool_calls": tool_calls,
        "reasoning": reasoning,
        "refusal": refusal,
        "confidence": None,
        "model": model,
        "finish_reason": finish_reason,
        "reasoning_streamed": bool(streaming_meta.get("reasoning_streamed", False)),
        "content_streamed": bool(streaming_meta.get("content_streamed", False)),
    }

    usage = data.get("usage")
    if usage:
        cache_usage = extract_prompt_cache_usage(usage)
        response_dict["usage"] = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "reasoning_tokens": usage.get("reasoning_tokens", 0),
            **cache_usage,
        }
        prompt_caching_cfg = context.providers_cfg.get("openrouter", {}).get(
            "prompt_caching", {}
        )
        if bool(prompt_caching_cfg.get("log_usage", True)):
            log_prompt_cache_usage(
                context.logger,
                "OpenRouter",
                model,
                cache_usage,
            )

    context.log_response(response_dict, request_type, model)
    context.logger.info(
        "OpenRouter response received: %s characters, tools: %s",
        len(str(content)),
        bool(tool_calls),
    )
    return response_dict


def call_openrouter(
    prompt: List[Dict[str, str]],
    request_type: str,
    provider_name: str,
    *,
    context: ProviderContext,
    model_override: Optional[str] = None,
    timeout_override: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    del provider_name
    provider_cfg = context.providers_cfg.get("openrouter", {})
    console_stream_cfg = provider_cfg.get("console_stream", {})
    stream_enabled = bool(console_stream_cfg.get("enabled", False))
    url, headers, payload, model = _prepare_openrouter_payload(
        prompt,
        request_type,
        context,
        model_override,
        **kwargs,
    )

    stream_state = _resolve_stream_state(
        request_type,
        model,
        console=kwargs.get("console") if stream_enabled else None,
        allow_reasoning_stream=bool(console_stream_cfg.get("reasoning", False)),
        allow_content_stream=bool(console_stream_cfg.get("content", True)),
        on_reasoning_delta=kwargs.get("on_reasoning_delta"),
        on_content_delta=kwargs.get("on_content_delta"),
    )

    if _stream_enabled(stream_state):
        payload["stream"] = True
        timeout_seconds = timeout_override or context.request_timeout or 30
        data = _run_openrouter_stream_request(
            url,
            headers=headers,
            json_payload=payload,
            timeout_seconds=float(timeout_seconds),
            stream_state=stream_state,
        )
        return _process_openrouter_response(data, context, model, request_type)

    data = post_provider_request(
        "OpenRouter",
        url,
        headers=headers,
        json_payload=payload,
        timeout_override=timeout_override,
        request_timeout=context.request_timeout,
    )
    return _process_openrouter_response(data, context, model, request_type)


async def call_openrouter_async(
    prompt: List[Dict[str, str]],
    request_type: str,
    provider_name: str,
    *,
    context: ProviderContext,
    model_override: Optional[str] = None,
    timeout_override: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    del provider_name
    provider_cfg = context.providers_cfg.get("openrouter", {})
    console_stream_cfg = provider_cfg.get("console_stream", {})
    stream_enabled = bool(console_stream_cfg.get("enabled", False))
    url, headers, payload, model = _prepare_openrouter_payload(
        prompt,
        request_type,
        context,
        model_override,
        **kwargs,
    )

    stream_state = _resolve_stream_state(
        request_type,
        model,
        console=kwargs.get("console") if stream_enabled else None,
        allow_reasoning_stream=bool(console_stream_cfg.get("reasoning", False)),
        allow_content_stream=bool(console_stream_cfg.get("content", True)),
        on_reasoning_delta=kwargs.get("on_reasoning_delta"),
        on_content_delta=kwargs.get("on_content_delta"),
    )

    if _stream_enabled(stream_state):
        payload["stream"] = True
        timeout_seconds = timeout_override or context.request_timeout or 30
        data = await _post_openrouter_stream_request_async(
            url,
            headers=headers,
            json_payload=payload,
            timeout_seconds=float(timeout_seconds),
            stream_state=stream_state,
        )
        return _process_openrouter_response(data, context, model, request_type)

    data = await post_provider_request_async(
        "OpenRouter",
        url,
        headers=headers,
        json_payload=payload,
        timeout_override=timeout_override,
        request_timeout=context.request_timeout,
    )
    return _process_openrouter_response(data, context, model, request_type)


def _run_openrouter_stream_request(
    url: str,
    *,
    headers: Dict[str, str],
    json_payload: Dict[str, Any],
    timeout_seconds: float,
    stream_state: _StreamState,
) -> Dict[str, Any]:
    from .base_provider import _run_async_request

    return _run_async_request(
        _post_openrouter_stream_request_async(
            url,
            headers=headers,
            json_payload=json_payload,
            timeout_seconds=timeout_seconds,
            stream_state=stream_state,
        )
    )