import logging
import unittest
from io import StringIO

from rich.console import Console

from commands.config import validate_config
from core import APIError
from services.ai.providers.base_provider import BaseAIProvider, ProviderContext
from services.ai.providers.provider_manager import ProviderManager


class _StubProvider(BaseAIProvider):
    def __init__(self, name: str, *, error: Exception | None = None) -> None:
        self._name = name
        self._error = error

    @property
    def provider_name(self) -> str:
        return self._name

    def call(
        self,
        prompt,
        request_type,
        provider_name,
        *,
        context: ProviderContext,
        model_override=None,
        timeout_override=None,
        **kwargs,
    ):
        del prompt, request_type, timeout_override, kwargs
        if self._error is not None:
            raise self._error
        model = (
            model_override
            or context.providers_cfg.get(provider_name, {}).get("default_model")
            or "default"
        )
        return {
            "content": f"{provider_name}:{model}",
            "tool_calls": [],
            "reasoning": "",
            "model": model,
        }

    async def call_async(
        self,
        prompt,
        request_type,
        provider_name,
        *,
        context: ProviderContext,
        model_override=None,
        timeout_override=None,
        **kwargs,
    ):
        return self.call(
            prompt,
            request_type,
            provider_name,
            context=context,
            model_override=model_override,
            timeout_override=timeout_override,
            **kwargs,
        )


class ProviderManagerResolutionTests(unittest.TestCase):
    def _make_manager(self, model_configs):
        manager = ProviderManager(
            logger=logging.getLogger("provider-manager-test"),
            providers_cfg={
                "deepseek": {
                    "api_key": "deepseek-key",
                    "default_model": "deepseek-chat",
                },
                "openrouter": {
                    "api_key": "openrouter-key",
                    "default_model": "openai/gpt-4o-mini",
                },
            },
            model_configs=model_configs,
            provider_order=["deepseek", "openrouter"],
            request_timeout=30,
        )
        manager._provider_clients = {
            "deepseek": _StubProvider("deepseek"),
            "openrouter": _StubProvider("openrouter"),
        }
        return manager

    def test_request_type_provider_order_overrides_global_order(self) -> None:
        manager = self._make_manager(
            {
                "news": {
                    "provider": "openrouter",
                    "model": "openai/gpt-5-mini",
                    "fallback_provider": "deepseek",
                }
            }
        )

        self.assertEqual(
            manager._resolve_provider_candidates("news"),
            ["openrouter", "deepseek"],
        )

    def test_fallback_provider_uses_provider_specific_model(self) -> None:
        manager = self._make_manager(
            {
                "reasoner": {
                    "provider": "openrouter",
                    "model": "openai/gpt-5-mini",
                    "fallback_provider": "deepseek",
                    "fallback_models": {"deepseek": "deepseek-reasoner"},
                }
            }
        )
        manager._provider_clients["openrouter"] = _StubProvider(
            "openrouter",
            error=APIError("primary failed"),
        )

        response = manager.send_request(
            [{"role": "user", "content": "hello"}],
            request_type="reasoner",
        )

        self.assertEqual(response["content"], "deepseek:deepseek-reasoner")

    def test_multiple_fallback_models_are_tried_in_order(self) -> None:
        class _SelectiveStubProvider(_StubProvider):
            def call(self, *args, **kwargs):
                model_override = kwargs.get("model_override")
                if model_override in {"deepseek-chat", "deepseek-reasoner"}:
                    raise APIError(f"{model_override} failed")
                return super().call(*args, **kwargs)

        manager = self._make_manager(
            {
                "summarizer": {
                    "provider": "deepseek",
                    "model": "deepseek-chat",
                    "fallback_models": [
                        "deepseek-reasoner",
                        "deepseek-chat-lite",
                    ],
                }
            }
        )
        manager._provider_clients["deepseek"] = _SelectiveStubProvider("deepseek")

        response = manager.send_request(
            [{"role": "user", "content": "hello"}],
            request_type="summarizer",
        )

        self.assertEqual(response["content"], "deepseek:deepseek-chat-lite")


class ConfigVisibilityTests(unittest.TestCase):
    def test_validate_config_mentions_openrouter_by_display_name(self) -> None:
        stream = StringIO()
        console = Console(file=stream, force_terminal=False, color_system=None)

        valid = validate_config(
            {
                "ai": {
                    "providers": {
                        "deepseek": {"api_key": "deepseek-key"},
                        "openrouter": {"api_key": ""},
                    },
                    "models": {
                        "news": {"provider": "deepseek", "model": "deepseek-chat"},
                        "reasoner": {
                            "provider": "openrouter",
                            "model": "openai/gpt-5-mini",
                        },
                    },
                }
            },
            console=console,
        )

        self.assertTrue(valid)
        self.assertIn("OpenRouter API key is not set", stream.getvalue())


if __name__ == "__main__":
    unittest.main()