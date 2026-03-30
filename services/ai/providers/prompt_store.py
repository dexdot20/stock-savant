"""Prompt management helper module."""

from dataclasses import dataclass
from typing import Any, Dict

import yaml

from core.paths import get_app_root

PROMPTS_FILE = "data/prompts.yaml"


@dataclass
class PromptStore:
    """Safely loads and caches prompt templates."""

    config: Dict[str, Any]
    logger: Any

    def __post_init__(self) -> None:
        self._prompts = self._load_prompts()

    def get(self, key: str) -> str:
        return self._prompts.get(key, "")

    def _build_prompt_variables(self) -> Dict[str, str]:
        ai_cfg = self.config.get("ai", {}) if isinstance(self.config, dict) else {}
        tool_cfg = ai_cfg.get("tools", {}) if isinstance(ai_cfg, dict) else {}
        agent_tool_cfg = (
            ai_cfg.get("agent_tool_limits", {}) if isinstance(ai_cfg, dict) else {}
        )
        retry_cfg = (
            self.config.get("network", {}).get("smart_retry", {})
            if isinstance(self.config, dict)
            else {}
        )

        parallel_limit = int(
            agent_tool_cfg.get(
                "max_parallel_tools", tool_cfg.get("parallel_limit", 2)
            )
        )
        error_rate_threshold = float(
            ai_cfg.get(
                "adaptive_tool_error_rate_threshold",
                retry_cfg.get("adaptive_error_rate_threshold", 0.30),
            )
        )

        return {
            "tool_parallel_limit": str(max(1, parallel_limit)),
            "adaptive_tool_error_rate_threshold": f"{max(0.0, error_rate_threshold):.2f}",
        }

    def _load_prompts(self) -> Dict[str, str]:
        files_cfg = (
            self.config.get("files", {}) if isinstance(self.config, dict) else {}
        )
        prompts_path = files_cfg.get("prompts_file") or PROMPTS_FILE

        from pathlib import Path

        app_root = get_app_root()
        cwd = Path.cwd()

        # Search order: CWD, APP_ROOT
        candidate_paths = []
        path_obj = Path(prompts_path)

        if path_obj.is_absolute():
            candidate_paths.append(path_obj)
        else:
            candidate_paths.append(cwd / prompts_path)
            candidate_paths.append(app_root / prompts_path)

        final_path = None
        for p in candidate_paths:
            if p.exists():
                final_path = p
                break

        if not final_path:
            self.logger.warning(
                "Prompts file not found. Searched: %s",
                [str(p) for p in candidate_paths],
            )
            return {}

        try:
            with open(final_path, "r", encoding="utf-8") as file:
                loaded = yaml.safe_load(file) or {}
                self.logger.info("Prompts loaded: %s", final_path)
        except FileNotFoundError:
            self.logger.warning(
                "Prompts file not found (despite fallback options): %s", final_path
            )
            return {}
        except yaml.YAMLError as exc:
            self.logger.error("Prompts file YAML error: %s", exc)
            return {}
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("Prompts file could not be loaded: %s", exc)
            return {}

        if not isinstance(loaded, dict):
            self.logger.warning("Prompts file in unexpected format: %s", prompts_path)
            return {}

        partials_raw = loaded.get("_partials", {})
        partials: Dict[str, str] = {}
        if isinstance(partials_raw, dict):
            for key, value in partials_raw.items():
                if isinstance(key, str) and isinstance(value, str):
                    partials[key] = value

        prompts: Dict[str, str] = {}
        prompt_vars = self._build_prompt_variables()
        for key, value in loaded.items():
            if key == "_partials":
                continue
            if isinstance(key, str) and isinstance(value, str):
                prompt_value = value
                for partial_key, partial_value in partials.items():
                    prompt_value = prompt_value.replace(
                        f"{{{{{partial_key}}}}}", partial_value
                    )
                for var_key, var_value in prompt_vars.items():
                    prompt_value = prompt_value.replace(f"{{{var_key}}}", var_value)
                prompts[key] = prompt_value
        return prompts
