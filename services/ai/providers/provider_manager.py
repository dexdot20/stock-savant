"""AI provider request orchestration."""

import time
from typing import Any, Callable, Dict, List, Optional

from config import is_configured_secret
from core import APIError, RateLimitError

from .base_provider import BaseAIProvider, ProviderContext
from .deepseek_provider import DeepSeekProvider
from .openrouter_provider import OpenRouterProvider
from .provider_metadata import get_provider_display_name

ProviderCallable = Callable[..., Dict[str, Any]]


class ProviderManager:
    """Manage AI provider calls and fallback chains."""

    _API_KEY_FIELDS = {
        "deepseek": "api_key",
        "openrouter": "api_key",
    }

    def __init__(
        self,
        logger: Any,
        providers_cfg: Dict[str, Any],
        model_configs: Dict[str, Any],
        provider_order: List[str],
        request_timeout: int,
    ) -> None:
        self.logger = logger
        self.providers_cfg = providers_cfg or {}
        self.model_configs = model_configs or {}
        self.provider_order = provider_order
        self.request_timeout = request_timeout
        self._context = ProviderContext(
            logger=self.logger,
            providers_cfg=self.providers_cfg,
            model_configs=self.model_configs,
            request_timeout=float(self.request_timeout),
        )
        self._provider_clients: Dict[str, BaseAIProvider] = {
            "deepseek": DeepSeekProvider(),
            "openrouter": OpenRouterProvider(),
        }

    def send_request(
        self,
        prompt: List[Dict[str, str]],
        request_type: str,
        timeout_override: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        provider_candidates = self._resolve_provider_candidates(request_type)
        if not provider_candidates:
            raise APIError("AI providers are not configured.")

        errors: List[str] = []
        availability_issues: List[str] = []
        last_rate_limit_exc: Optional[RateLimitError] = None
        if timeout_override is not None:
            try:
                effective_timeout = max(1.0, float(timeout_override))
            except (TypeError, ValueError):
                effective_timeout = float(self.request_timeout)
        else:
            effective_timeout = float(self.request_timeout)

        self.logger.info(
            "AI request starting: type=%s, providers=%s, timeout=%.1fs",
            request_type,
            provider_candidates,
            effective_timeout,
        )
        self.logger.debug(
            "AI request payload summary: type=%s messages=%d chars=%d kwargs_keys=%s",
            request_type,
            len(prompt),
            sum(len(str(item)) for item in prompt),
            sorted(list(kwargs.keys())),
        )

        # Model config'den sağlayıcı override değerlerini oku
        _model_cfg = self.model_configs.get(request_type)
        _model_provider = (
            _model_cfg.get("provider") if isinstance(_model_cfg, dict) else None
        )
        _fallback_provider = (
            _model_cfg.get("fallback_provider")
            if isinstance(_model_cfg, dict)
            else None
        )

        for provider_name in provider_candidates:
            if not self._is_provider_enabled(provider_name):
                provider_issue = self._build_provider_unavailable_message(provider_name)
                availability_issues.append(provider_issue)
                self.logger.warning("Provider unavailable: %s", provider_issue)
                continue

            handler = self._get_handler(provider_name)
            if handler is None:
                handler_issue = (
                    f"{self._get_provider_display_name(provider_name)}: request handler not found."
                )
                errors.append(handler_issue)
                self.logger.warning("Handler not found: %s", provider_name)
                continue

            # Ana model ve fallback model listesi oluştur
            models_to_try = self._get_models_with_fallback(provider_name, request_type)

            for model_index, model_override in enumerate(models_to_try):
                is_fallback = model_index > 0
                try:
                    display = self._get_provider_display_name(provider_name)
                    model_display = model_override or "default"
                    fallback_info = " (fallback)" if is_fallback else ""
                    self.logger.info(
                        "Trying provider: %s, model: %s%s, timeout=%.1fs",
                        display,
                        model_display,
                        fallback_info,
                        effective_timeout,
                    )
                    start_time = time.time()

                    response = handler(
                        prompt,
                        request_type,
                        provider_name,
                        model_override=model_override,
                        timeout_override=effective_timeout,
                        provider_order_override=(
                            _fallback_provider if is_fallback else _model_provider
                        ),
                        **kwargs,
                    )

                    elapsed = time.time() - start_time
                    self.logger.info(
                        "Provider succeeded: %s, model: %s%s (%.2fs, %s chars)",
                        display,
                        model_display,
                        fallback_info,
                        elapsed,
                        len(str(response.get("content", ""))),
                    )
                    self.logger.debug(
                        "Provider response summary: provider=%s model=%s%s content_chars=%d tool_calls=%d reasoning_chars=%d",
                        provider_name,
                        model_display,
                        fallback_info,
                        len(str(response.get("content", ""))),
                        len(response.get("tool_calls") or []),
                        len(str(response.get("reasoning", ""))),
                    )
                    return response

                except RateLimitError as exc:
                    last_rate_limit_exc = exc
                    model_info = f" (model: {model_override})" if model_override else ""
                    error_message = f"{self._get_provider_display_name(provider_name)}{model_info}: {exc}"
                    errors.append(error_message)
                    self.logger.warning(
                        "AI provider rate limit (%s%s): %s",
                        provider_name,
                        model_info,
                        exc,
                    )
                    continue

                except APIError as exc:
                    model_info = f" (model: {model_override})" if model_override else ""
                    error_message = f"{self._get_provider_display_name(provider_name)}{model_info}: {exc}"
                    errors.append(error_message)
                    self.logger.warning(
                        "AI provider API error (%s%s): %s",
                        provider_name,
                        model_info,
                        exc,
                        extra={
                            "request_type": request_type,
                            "provider": provider_name,
                            "model": model_override,
                        },
                    )
                    # Fallback modelleri denemeye devam et
                    continue

                except TimeoutError as exc:
                    model_info = f" (model: {model_override})" if model_override else ""
                    error_message = f"{self._get_provider_display_name(provider_name)}{model_info}: Timeout ({effective_timeout}s)"
                    errors.append(error_message)
                    self.logger.error(
                        "AI provider timeout (%s%s): %s",
                        provider_name,
                        model_info,
                        exc,
                        extra={
                            "request_type": request_type,
                            "provider": provider_name,
                            "model": model_override,
                            "timeout": effective_timeout,
                        },
                    )
                    # Fallback modelleri denemeye devam et
                    continue

                except Exception as exc:  # pragma: no cover - savunmacı
                    model_info = f" (model: {model_override})" if model_override else ""
                    error_message = f"{self._get_provider_display_name(provider_name)}{model_info}: {exc}"
                    errors.append(error_message)
                    self.logger.error(
                        "AI provider unexpected error (%s%s): %s",
                        provider_name,
                        model_info,
                        exc,
                        exc_info=True,
                        extra={
                            "request_type": request_type,
                            "provider": provider_name,
                            "model": model_override,
                        },
                    )
                    # Fallback modelleri denemeye devam et
                    continue

        if errors:
            error_summary = "; ".join(errors)
            self.logger.error(
                "All AI providers failed: %s",
                error_summary,
                extra={"request_type": request_type, "errors": errors},
            )

            if last_rate_limit_exc:
                # En az bir rate limit hatası varsa, bunu RateLimitError olarak fırlat
                # ki üst katman (orchestrator) akıllı retry yapabilsin.
                raise RateLimitError(
                    f"All AI providers failed (Rate Limit): {error_summary}",
                    retry_after=last_rate_limit_exc.retry_after,
                )

            raise APIError(error_summary)

        if availability_issues:
            raise APIError(
                "; ".join(availability_issues),
                details={
                    "request_type": request_type,
                    "providers": provider_candidates,
                },
            )

        raise APIError(
            "No active AI provider was found or all attempts failed."
        )

    async def send_request_async(
        self,
        prompt: List[Dict[str, str]],
        request_type: str,
        timeout_override: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Perform an asynchronous AI provider request."""
        provider_candidates = self._resolve_provider_candidates(request_type)
        if not provider_candidates:
            raise APIError("AI providers are not configured.")

        errors: List[str] = []
        availability_issues: List[str] = []
        last_rate_limit_exc: Optional[RateLimitError] = None
        if timeout_override is not None:
            try:
                effective_timeout = max(1.0, float(timeout_override))
            except (TypeError, ValueError):
                effective_timeout = float(self.request_timeout)
        else:
            effective_timeout = float(self.request_timeout)

        self.logger.info(
            "AI ASYNC request starting: type=%s, providers=%s, timeout=%.1fs",
            request_type,
            provider_candidates,
            effective_timeout,
        )

        # Model config'den sağlayıcı override değerlerini oku
        _model_cfg = self.model_configs.get(request_type)
        _model_provider = (
            _model_cfg.get("provider") if isinstance(_model_cfg, dict) else None
        )
        _fallback_provider = (
            _model_cfg.get("fallback_provider")
            if isinstance(_model_cfg, dict)
            else None
        )

        for provider_name in provider_candidates:
            if not self._is_provider_enabled(provider_name):
                provider_issue = self._build_provider_unavailable_message(provider_name)
                availability_issues.append(provider_issue)
                self.logger.warning("Provider unavailable: %s", provider_issue)
                continue

            # Asenkron handler al (yoksa senkron handler'ı wrap et)
            handler = self._get_async_handler(provider_name)
            if handler is None:
                handler_issue = (
                    f"{self._get_provider_display_name(provider_name)}: async request handler not found."
                )
                errors.append(handler_issue)
                self.logger.warning("Async handler not found: %s", provider_name)
                continue

            models_to_try = self._get_models_with_fallback(provider_name, request_type)

            for model_index, model_override in enumerate(models_to_try):
                is_fallback = model_index > 0
                try:
                    display = self._get_provider_display_name(provider_name)
                    model_display = model_override or "default"
                    fallback_info = " (fallback)" if is_fallback else ""
                    self.logger.info(
                        "Trying provider (ASYNC): %s, model: %s%s, timeout=%.1fs",
                        display,
                        model_display,
                        fallback_info,
                        effective_timeout,
                    )
                    start_time = time.time()

                    response = await handler(
                        prompt,
                        request_type,
                        provider_name,
                        model_override=model_override,
                        timeout_override=effective_timeout,
                        provider_order_override=(
                            _fallback_provider if is_fallback else _model_provider
                        ),
                        **kwargs,
                    )

                    elapsed = time.time() - start_time
                    self.logger.info(
                        "Provider succeeded: %s, model: %s%s (%.2fs, %s chars)",
                        display,
                        model_display,
                        fallback_info,
                        elapsed,
                        len(str(response.get("content", ""))),
                    )
                    return response

                except RateLimitError as exc:
                    last_rate_limit_exc = exc
                    model_info = f" (model: {model_override})" if model_override else ""
                    error_message = f"{self._get_provider_display_name(provider_name)}{model_info}: {exc}"
                    errors.append(error_message)
                    self.logger.warning(
                        "AI provider rate limit (%s%s): %s",
                        provider_name,
                        model_info,
                        exc,
                    )
                    continue

                except APIError as exc:
                    model_info = f" (model: {model_override})" if model_override else ""
                    error_message = f"{self._get_provider_display_name(provider_name)}{model_info}: {exc}"
                    errors.append(error_message)
                    self.logger.warning(
                        "AI provider API error (%s%s): %s",
                        provider_name,
                        model_info,
                        exc,
                        extra={
                            "request_type": request_type,
                            "provider": provider_name,
                            "model": model_override,
                        },
                    )
                    continue

                except TimeoutError as exc:
                    model_info = f" (model: {model_override})" if model_override else ""
                    error_message = f"{self._get_provider_display_name(provider_name)}{model_info}: Timeout ({effective_timeout}s)"
                    errors.append(error_message)
                    self.logger.error(
                        "AI provider timeout (%s%s): %s",
                        provider_name,
                        model_info,
                        exc,
                        extra={
                            "request_type": request_type,
                            "provider": provider_name,
                            "model": model_override,
                            "timeout": effective_timeout,
                        },
                    )
                    continue

                except Exception as exc:
                    model_info = f" (model: {model_override})" if model_override else ""
                    error_message = f"{self._get_provider_display_name(provider_name)}{model_info}: {exc}"
                    errors.append(error_message)
                    self.logger.error(
                        "AI provider unexpected error (%s%s): %s",
                        provider_name,
                        model_info,
                        exc,
                        exc_info=True,
                        extra={
                            "request_type": request_type,
                            "provider": provider_name,
                            "model": model_override,
                        },
                    )
                    continue

        if errors:
            error_summary = "; ".join(errors)
            if last_rate_limit_exc:
                raise RateLimitError(
                    f"All AI providers failed asynchronously (Rate Limit): {error_summary}",
                    retry_after=last_rate_limit_exc.retry_after,
                )
            raise APIError(error_summary)

        if availability_issues:
            raise APIError(
                "; ".join(availability_issues),
                details={
                    "request_type": request_type,
                    "providers": provider_candidates,
                },
            )

        raise APIError(
            "No active AI provider was found or all attempts failed (Async)."
        )

    def get_primary_enabled_provider(self, request_type: str) -> Optional[str]:
        for candidate in self._resolve_provider_candidates(request_type):
            if self._is_provider_enabled(candidate):
                return candidate
        return None

    def get_default_provider(self) -> str:
        if self.provider_order:
            return self.provider_order[0]
        if self.providers_cfg:
            return next(iter(self.providers_cfg.keys()))
        return "deepseek"

    def display_name(self, provider_name: str) -> str:
        return self._get_provider_display_name(provider_name)

    def _resolve_provider_candidates(self, request_type: str) -> List[str]:
        candidates: List[str] = []
        model_cfg = self.model_configs.get(request_type)

        if isinstance(model_cfg, dict):
            candidates.extend(self._coerce_provider_list(model_cfg.get("provider")))
            candidates.extend(
                self._coerce_provider_list(
                    model_cfg.get("fallback_provider")
                    or model_cfg.get("fallback_providers")
                )
            )

        candidates.extend(self.provider_order)

        if not candidates:
            candidates.extend(self.providers_cfg.keys())

        if not candidates:
            candidates.append("deepseek")

        deduped: List[str] = []
        for candidate in candidates:
            if candidate and candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _is_provider_enabled(self, provider_name: str) -> bool:
        provider_cfg = self.providers_cfg.get(provider_name, {})
        if not provider_cfg:
            self.logger.debug(f"Provider {provider_name}: config not found")
            return False

        enabled = provider_cfg.get("enabled")
        if enabled is False:
            self.logger.debug(f"Provider {provider_name}: enabled=False")
            return False

        key_field = self._API_KEY_FIELDS.get(provider_name)
        if key_field:
            api_key = provider_cfg.get(key_field)
            is_valid = is_configured_secret(api_key)
            self.logger.debug(
                f"Provider {provider_name}: api_key={'[SET]' if is_valid else '[MISSING/INVALID]'} "
                f"(length={len(api_key) if api_key else 0})"
            )
            if is_valid:
                return True

        if enabled is True and not key_field:
            self.logger.debug(f"Provider {provider_name}: enabled=True")
            return True

        self.logger.debug(
            f"Provider {provider_name}: disabled (API key missing or invalid)"
        )
        return False

    def _build_provider_unavailable_message(self, provider_name: str) -> str:
        display = self._get_provider_display_name(provider_name)
        provider_cfg = self.providers_cfg.get(provider_name, {})
        if not provider_cfg:
            return f"{display}: provider configuration not found."

        enabled = provider_cfg.get("enabled")
        if enabled is False:
            return f"{display}: disabled in configuration."

        key_field = self._API_KEY_FIELDS.get(provider_name)
        if key_field:
            api_key = provider_cfg.get(key_field)
            if not is_configured_secret(api_key):
                return (
                    f"{display}: API key missing or invalid. "
                    f"Check ai.providers.{provider_name}.{key_field} in config/settings.py."
                )

        return f"{display}: currently unavailable."

    def _get_handler(self, provider_name: str) -> Optional[ProviderCallable]:
        provider = self._provider_clients.get(provider_name)
        if not isinstance(provider, BaseAIProvider):
            return None
        return lambda prompt, request_type, name, **kwargs: provider.call(
            prompt,
            request_type,
            name,
            context=self._context,
            **kwargs,
        )

    def _get_async_handler(self, provider_name: str) -> Optional[Callable[..., Any]]:
        """Dönüş tipi async olan handler döner."""
        provider = self._provider_clients.get(provider_name)
        if not isinstance(provider, BaseAIProvider):
            return None
        return lambda prompt, request_type, name, **kwargs: provider.call_async(
            prompt,
            request_type,
            name,
            context=self._context,
            **kwargs,
        )

    def _get_models_with_fallback(
        self, provider_name: str, request_type: str
    ) -> List[Optional[str]]:
        """Birincil model ve fallback modelleri içeren liste döndürür.

        Args:
            provider_name: Sağlayıcı adı (örn: "deepseek")
            request_type: İstek tipi (örn: "news", "reasoner")

        Returns:
            Denenecek modellerin listesi. İlk eleman birincil model, diğerleri fallback.
        """
        models: List[Optional[str]] = []
        model_cfg = self.model_configs.get(request_type)

        if not model_cfg:
            return [None]  # Varsayılan model kullan

        # String formatında model tanımı
        if isinstance(model_cfg, str):
            models.append(model_cfg)
            return models

        if not isinstance(model_cfg, dict):
            return [None]

        primary_provider = model_cfg.get("provider")

        # Birincil modeli ekle
        preferred_model = model_cfg.get("model")
        if preferred_model and (not primary_provider or primary_provider == provider_name):
            models.append(preferred_model)

        # Fallback modelleri ekle
        fallback_models = model_cfg.get("fallback") or model_cfg.get("fallback_models")
        if isinstance(fallback_models, dict):
            self._append_models(models, fallback_models.get(provider_name))
        elif provider_name == primary_provider and isinstance(fallback_models, list):
            # Liste formatında fallback desteği
            self._append_models(models, fallback_models)
        elif provider_name == primary_provider and isinstance(fallback_models, str):
            # String fallback desteği (virgül ile ayrılmış olabilir)
            self._append_models(models, fallback_models)

        return models if models else [None]

    def _get_provider_display_name(self, provider_name: Optional[str]) -> str:
        return get_provider_display_name(provider_name)

    @staticmethod
    def _coerce_provider_list(value: Any) -> List[str]:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @staticmethod
    def _append_models(models: List[Optional[str]], value: Any) -> None:
        if isinstance(value, str):
            candidates = [part.strip() for part in value.split(",") if part.strip()]
        elif isinstance(value, list):
            candidates = [str(item).strip() for item in value if str(item).strip()]
        else:
            candidates = []

        for candidate in candidates:
            if candidate not in models:
                models.append(candidate)
