"""Central runtime defaults for the application.

This module is intentionally conservative:
- keep only settings that are read by the current codebase,
- resolve environment-backed secrets once here,
- preserve a few legacy constants because older modules still import them directly.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

from core.paths import get_app_root, get_runtime_dir

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional bootstrap dependency guard
    load_dotenv = None


def _load_env_file() -> None:
    """Load the local .env file when python-dotenv is available."""
    if load_dotenv is not None:
        load_dotenv(get_app_root() / ".env")


def _load_user_agents() -> list[str]:
    """Load user-agent strings from user-agents.txt with a safe fallback list."""
    agents_path = get_app_root() / "user-agents.txt"
    if agents_path.exists():
        try:
            with open(agents_path, "r", encoding="utf-8") as file_handle:
                agents = [line.strip() for line in file_handle if line.strip()]
            if agents:
                return agents
        except OSError:
            pass

    return [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    ]


_load_env_file()


# Legacy constants kept for modules that still import them directly instead of
# reading the config dictionary at runtime.
CACHE_FILE = str(get_runtime_dir() / "stock_cache.db")
NA_VALUE = "N/A"
LOGISTIC_SCALING_FACTOR = 2
DEFAULT_USER_AGENTS = _load_user_agents()
PROMPTS_FILE = "data/prompts.yaml"

_MISSING_SECRET_VALUES = frozenset(
    {"", "None", "null", "your_api_key_here", "your_key_here"}
)


def _get_env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def is_configured_secret(value: Any) -> bool:
    """Return whether a config secret-like value is meaningfully configured."""
    if not isinstance(value, str):
        return False
    return value.strip() not in _MISSING_SECRET_VALUES


# Keep this dictionary focused on values that are consumed in the current code.
# When a module needs a new runtime knob, add it here instead of hiding it behind
# an in-code fallback. That keeps behavior discoverable for non-technical users.
DEFAULT_CONFIG: Dict[str, Any] = {
    # Root-level switches kept for legacy call sites in CLI and analysis flows.
    "cache_enabled": True,
    "use_company_keywords": True,
    "max_workers": 4,
    "terminal_debug": True,

    # Shared HTTP behavior for news search, content extraction and generic tools.
    # The previous 300s timeout was too forgiving and could stall the whole flow.
    "network": {
        "request_timeout_seconds": 45,
        "max_retry_count": 3,
        "retry_delay_seconds": 1.0,
        "verify_ssl": True,
        "smart_retry": {
            "enabled": True,
            "jitter_ratio": 0.20,
            "max_delay_seconds": 20.0,
        },
        "health_check": {
            "enabled": True,
            "degraded_after_failures": 3,
            "unhealthy_after_failures": 6,
            "tool_cooldown_seconds": 180,
        },
    },

    # Cache namespaces all share the same schema. TTLs are tuned so volatile
    # news/search payloads refresh faster while heavier analysis still benefits
    # from caching within the same trading session.
    "cache": {
        "analysis_daily": {"max_entries": 512, "ttl_seconds": 21600},
        "tools_web_search": {"max_entries": 256, "ttl_seconds": 900},
        "tools_news_search": {"max_entries": 256, "ttl_seconds": 600},
        "tools_summary": {"max_entries": 128, "ttl_seconds": 1800},
        "rag_embeddings": {"max_entries": 2048, "ttl_seconds": 43200},
    },

    # File paths used by prompt loading and legacy disk cache consumers.
    "files": {
        "prompts_file": PROMPTS_FILE,
        "cache_file": CACHE_FILE,
        "json_indent": 2,
    },

    "kap": {
        "timeout_seconds": 20,
        "default_batch_workers": 5,
        "member_map_cache_ttl_seconds": 86400,
        "proxy_file": "proxies.txt",
        "proxy_error_threshold": 3,
        "proxy_reset_interval_seconds": 300,
    },
    "kap_intelligence": {
        "default_days": 30,
        "default_limit": 5,
        "alert_days": 7,
        "min_alert_severity_score": 70,
    },

    "ai": {
        "debug": False,
        "optimize_shares_history": True,
        "output_language": "English",
        "reporting": {
            "readable_mode_default": True,
            "max_reason_points": 3,
        },
        # Python tool execution must stay strict enough for safety but not so low
        # that simple financial calculations get cut off too early.
        "python_exec": {
            "enabled": True,
            "default_timeout_seconds": 10,
            "max_timeout_seconds": 25,
            "max_output_chars": 12000,
            "max_memory_mb": 256,
            "max_file_size_mb": 2,
            "sandbox_subdir": "python_exec",
            "blocked_modules": [
                "aiohttp",
                "builtins",
                "ctypes",
                "ftplib",
                "http",
                "importlib",
                "multiprocessing",
                "os",
                "pathlib",
                "requests",
                "shutil",
                "signal",
                "socket",
                "ssl",
                "subprocess",
                "sys",
                "telnetlib",
                "threading",
                "urllib",
                "webbrowser",
            ],
        },
        "report_tool": {
            "enabled": True,
            "redact_sensitive_values": True,
            "max_field_chars": 4000,
            "max_context_items": 20,
            "max_reports_per_process": 200,
            "dedupe_window": 50,
        },

        # Tool output truncation keeps the agent context window healthy.
        "tool_output": {
            "enabled": True,
            "default_max_chars": 6000,
            "per_tool": {
                "yfinance_company_data": 15000,
                "fetch_url_content": 12000,
                "summarize_url_content": 14000,
                "search_memory": 16000,
                "python_exec": 12000,
                "report": 4000,
                "yfinance_overview": 5000,
                "yfinance_index_data": 7000,
                "yfinance_price_history": 7000,
                "yfinance_dividends": 5000,
                "yfinance_analyst": 7000,
                "yfinance_earnings": 6000,
                "yfinance_ownership": 7000,
                "yfinance_sustainability": 5000,
            },
        },

        "search_memory_preview_chars": {
            "default": 1400,
            "analysis": 2400,
            "pre_research": 1800,
            "news": 1400,
        },

        # RAG retrieval stays enabled by default because several agent flows assume
        # the vector store exists. A slightly higher top_k gives better recall
        # without materially inflating prompt size.
        "rag": {
            "enabled": True,
            "embedding_model": "intfloat/multilingual-e5-large",
            "embedding_batch_size": 32,
            "normalize_embeddings": True,
            "huggingface": {
                "hf_token": _get_env_str("HF_TOKEN", ""),
                "suppress_hub_warnings": True,
                "suppress_model_load_report": True,
                "suppress_http_logs": True,
            },
            "top_k": 6,
            "candidate_pool": 18,
            "analysis_chunk_size": 1800,
            "analysis_chunk_overlap": 220,
            "pre_research_chunk_size": 1800,
            "pre_research_chunk_overlap": 220,
            "news_chunk_size": 1800,
            "news_chunk_overlap": 220,
            "reranker": {
                "enabled": False,
                "model": "BAAI/bge-reranker-v2-m3",
                "weight": 0.65,
                "candidate_pool": 18,
            },
            "query_expansion": {
                "enabled": True,
                "weight": 0.35,
                "warm_start_enabled": True,
                "timeout_seconds": 15,
                "hypothesis_max_chars": 900,
                "fallback_template": (
                    "Factual research notes about {query}. Include dated catalysts, "
                    "financial metrics, management commentary, risks, and prior analysis context."
                ),
            },
            "collections": {
                "analysis": "analysis_history",
                "pre_research": "pre_research_reports",
                "news": "news_articles",
            },
        },

        "agent_steps": {
            "news": 30,
            "pre_research": 15,
            "pre_research_depth_mode": "standard",
            "comparison": 30,
            "continuation_increment": 10,
        },

        "agent_tool_limits": {
            "max_parallel_tools": 4,
        },

        # Working-memory thresholds are intentionally explicit because they shape
        # the quality/cost trade-off of long agent runs.
        "working_memory": {
            "max_facts": 72,
            "adaptive_max_facts": 96,
            "adaptive_usage_step": 0.25,
            "max_sources": 30,
            "max_questions": 20,
            "max_contradictions": 20,
            "max_evidence_records": 24,
            "consolidation_threshold": 45,
            "fact_similarity_threshold": 0.82,
            "fact_similarity_window": 12,
            "consolidation_similarity_ratio": 0.35,
            "consolidation_cache_size": 32,
            "importance_recalc_interval_seconds": 120,
            "refresh_context_top_k": 5,
            "refresh_context_recent_days": 120,
            "refresh_context_window": 1,
            "shared_pool_enabled": True,
            "shared_pool_search_top_k": 4,
            "shared_pool_similarity_threshold": 0.6,
        },

        "agent_reflection": {
            "enabled": True,
            "suppress_reasoning_panel_after_stream": True,
            "interval_steps": 3,
            "max_facts": 4,
            "max_questions": 3,
            "max_contradictions": 2,
        },

        # Context-budget safeguards for long-running conversations.
        "token_budget": {
            "enabled": True,
            "history_token_limit": 96000,
            "immediate_context_ratio": 0.5,
            "working_memory_rescue_ratio": 0.5,
            "encoding": "cl100k_base",
        },

        # summarize_url_content uses these limits to keep page condensation focused.
        "url_content_pruning": {
            "chunk_size_chars": 2200,
            "max_selected_chunks": 3,
            "max_expanded_chunks": 5,
            "lead_chunk_count": 2,
            "min_selected_chars": 1200,
            "min_focus_score": 2.2,
            "min_salience_score": 0.45,
            "coverage_ratio_floor": 0.18,
            "fallback_fulltext_chars": 10000,
        },

        # MODEL & PROVIDER SEÇİMİ — açıklama ve örnekler
        # - Bu bölümdeki `models` öğeleri için `provider` alanı burada tanımladığınız
        #   sağlayıcı isimlerinden birini işaret eder (ör. "openrouter", "deepseek").
        # - `fallback_models` ile provider-özgü model yedeklemeleri tanımlanabilir.
        # - OpenRouter gibi gateway'lerde route/selection parametreleri kullanarak
        #   isteklerin hangi alt-sağlayıcı(lar)a gideceğini belirleyebilirsiniz.
        #   Kesin alan isimleri ve davranış için OpenRouter dokümanına bakın:
        #   https://openrouter.ai/docs/guides/routing/provider-selection
        # Örnek kullanım:
        # DEFAULT_CONFIG['ai']['models']['news'] = {
        #     "provider": "openrouter",
        #     "model": "x-ai/grok-4.1-fast",
        #     "fallback_models": {"openrouter": ["minimax/minimax-m2.5"]},
        #     "routing_overrides": {
        #         "preferred_providers": ["openai", "x-ai"],
        #         "force_provider": None,
        #     },
        # }
        #
        # Not: API anahtarlarını doğrudan buraya koymayın — ortam değişkenlerini kullanın.
        "models": {
            "news": {
                "provider": "openrouter",
                "model": "x-ai/grok-4.1-fast",
                "fallback_models": {
                  "openrouter": [
                        "minimax/minimax-m2.5",
                        "minimax/minimax-m2.7",
                        "x-ai/grok-4.1-fast",
                    ],
                },
            },
            "reasoner": {
                "provider": "deepseek",
                "model": "deepseek-reasoner",
                "fallback_models": {
                    "openrouter": [
                        "moonshotai/kimi-k2.5",
                    ],
                },
            },
            "pre_research": {
                "provider": "openrouter",
                "model": "x-ai/grok-4.1-fast",
                "fallback_models": {
                    "openrouter": [
                        "minimax/minimax-m2.5",
                        "minimax/minimax-m2.7",
                        "x-ai/grok-4.1-fast",
                    ],
                },
            },
            "comparison": {
                "provider": "deepseek",
                "model": "deepseek-reasoner",
                "fallback_models": {
                    "openrouter": [
                        "moonshotai/kimi-k2.5",
                    ],
                },
            },
            "summarizer": {
                "provider": "openrouter",
                "model": "openai/gpt-oss-120b",
                "fallback_models": {
                    "openrouter": [
                        "xiaomi/mimo-v2-flash",
                        "z-ai/glm-4.7-flash",
                    ],
                },
            },
        },

        # PROVIDER YAPILANDIRMA BÖLÜMÜ (örnekler ve açıklamalar)
        # - Her provider girdisi kimlik doğrulama bilgileri (`api_key`), `base_url`,
        #   `default_model` ve isteğe bağlı yapılandırma (timeout, retry_policy, routing)
        #   içermelidir.
        # - OpenRouter kullanıyorsanız burada `routing` veya `provider_selection` benzeri
        #   alanlarla tercihlerinizi belirtebilirsiniz. Uygulamanın HTTP katmanı bu
        #   alanları OpenRouter istek gövdesine maplemelidir.
        # - `prompt_caching.preserve_sticky_routing` True ise cache uyumu için aynı
        #   provider seçimi korunur.
        # Örnek (detaylı):
        # "openrouter": {
        #     "api_key": _get_env_str("OPENROUTER_API_KEY"),
        #     "base_url": "https://openrouter.ai/api/v1",
        #     "default_model": "x-ai/grok-4.1-fast",
        #     "timeout_seconds": 30,
        #     "retry_policy": {"max_attempts": 3, "initial_delay": 0.5, "backoff_factor": 2.0},
        #     "routing": {
        #         "strategy": "preferred",  # "preferred" | "force" | "fallback"
        #         "preferred_providers": ["openai", "x-ai"],
        #         "fallback_order": ["openai", "anthropic"],
        #         "force_provider": None,
        #     },
        #     "model_map": {"chat": "gpt-4o-mini", "reasoning": "grok-4.1-fast"},
        #     "prompt_caching": {"enabled": True, "preserve_sticky_routing": True},
        # },
        "providers": {
            "deepseek": {
                "api_key": _get_env_str("DEEPSEEK_API_KEY"),
                "base_url": "https://api.deepseek.com",
                "default_model": "deepseek-chat",
                "console_stream": {
                    "enabled": True,
                    "content": True,
                    "reasoning": True,
                },
                "temperature": 0.8,
                "max_tokens": 8192,
                "top_p": 0.7,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "reasoning": {
                    "enabled": True,
                    "max_tokens": 8192,
                },
                "prompt_caching": {
                    "enabled": True,
                    "log_usage": True,
                },
            },
            "openrouter": {
                "api_key": _get_env_str("OPENROUTER_API_KEY"),
                "base_url": "https://openrouter.ai/api/v1",
                "default_model": "",
                "http_referer": _get_env_str("OPENROUTER_HTTP_REFERER"),
                "app_title": _get_env_str("OPENROUTER_APP_TITLE"),
                "console_stream": {
                    "enabled": True,
                    "content": True,
                    "reasoning": True,
                },
                "temperature": 0.8,
                "max_tokens": 8192,
                "top_p": 0.7,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "reasoning": {
                    "enabled": True,
                    "effort": "medium",
                    "max_tokens": 4096,
                    "exclude": False,
                },
                "provider": {
                    "require_parameters": False,
                },
                "prompt_caching": {
                    "enabled": True,
                    "preserve_sticky_routing": True,
                    "log_usage": True,
                },
                # OpenRouter için varsayılan routing seçenekleri.
                # Bu yapı, uygulama tarafından OpenRouter'ın `provider` parametresine
                # çevrilip istek gövdesine eklenmelidir. Aşağıdaki alanlar OpenRouter
                # dokümanındaki `provider` objesiyle eşleşir (bkz. provider-selection).
                # Yorum satırlarında gerçek dünya örnekleri gösterilmiştir.
                #
                # Örnek provider objesi (istek başına gönderilecek):
                # provider_override = {
                #     "order": ["openai", "anthropic"],
                #     "allow_fallbacks": True,
                #     "require_parameters": False,
                #     "data_collection": "deny",  # only use ZDR/non-training endpoints
                #     "zdr": True,
                #     "only": ["azure"],
                #     "ignore": ["deepinfra"],
                #     "quantizations": ["fp8"],
                #     "sort": {"by": "throughput", "partition": "none"},
                #     "preferred_min_throughput": {"p90": 50},
                #     "preferred_max_latency": {"p90": 3},
                #     "max_price": {"prompt": 1.0, "completion": 2.0},
                # }
                #
                # Basit varsayılanlar (uygulama bu değerleri ihtiyaç halinde açıktan
                # üzerine yazmalıdır):
                "routing_defaults": {
                    "order": [],
                    "allow_fallbacks": True,
                    "require_parameters": False,
                    "data_collection": "allow",
                    "zdr": False,
                    # Varsayılan olarak OpenRouter isteklerini throughput'a
                    # öncelik verecek şekilde ayarlıyoruz. Bu, OpenRouter'ın
                    # sağlayıcı sıralamasını "throughput" bazlı yapar ve
                    # global en yüksek throughput'u seçmeye çalışır.
                    # Partition: "none" seçimi, tüm modeller arasındaki
                    # endpoint'leri throughput'a göre karşılaştırır (model
                    # gruplaması yapılmaz).
                    "sort": {"by": "throughput", "partition": "none"},
                },
            },
        },
    },

    # These knobs are consumed directly by the news search layer.
    "web_search": {
        "total_searches": 3,
        "search_delay_seconds": 2,
        "max_concurrent_requests": 4,
        "min_article_length": 100,
        "interactive_selection": False,
        "ddgs_region": "wt-wt",
        "ddgs_safesearch": "off",
    },

    # Separate from web_search because extraction quality and extraction caching
    # have very different tuning concerns.
    "content_extraction": {
        "use_readability": True,
        "cache_ttl_seconds": 1800,
        "cache_max_entries": 256,
    },

    # ArticleProcessor reads this section directly for async/threadpool fan-out.
    "news_processing": {
        "max_workers": 4,
    },

    "proxy": {
        "enabled": True,
        "file": "proxies.txt",
        "max_failures": 3,
        "market_data": True,
        "news_search": True,
        "content_extraction": False,
    },

    # API defaults intentionally remain minimal because api/main.py currently uses
    # only host/port/workers and allowed network rules.
    "api": {
        "host": "0.0.0.0",
        "port": 8001,
        "workers": 1,
        "user_agents": DEFAULT_USER_AGENTS,
        "allowed_networks": [
            "127.0.0.0/8",
            "::1/128",
            "192.168.0.0/24",
        ],
    },

    "logging": {
        "console_level": "WARNING",
        "file_level": "INFO",
        "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "file": "borsa.log",
    },

    # This block consolidates market-data behavior that previously lived behind
    # setdefault fallbacks in the service layer.
    "market_data": {
        "provider": "yfinance",
        "history_period": "1y",
        "history_interval": "1d",
        "default_cache_ttl_hours": 12,
        "api_request_timeout": 8.0,
        "api_total_timeout": 25.0,
        "api_max_retries": 3,
        "api_retry_delay": 0.75,
        "api_max_backoff_delay": 10.0,
        "api_circuit_breaker_threshold": 5,
        "api_circuit_breaker_timeout": 60.0,
        "history_repair_enabled": True,
        "history_repair_non_us_only": True,
        "yfinance_news_limit": 5,
        "prefer_proxy_for_yfinance": False,
    },

    "portfolio": {
        "protection": {
            "enabled": True,
            "max_positions": 20,
            "max_single_position_pct": 40.0,
            "max_total_cost": 0,
        },
        "risk": {
            "max_sector_weight_pct": 55.0,
            "low_confidence_threshold_pct": 55.0,
        },
    },
}


def get_default_config() -> Dict[str, Any]:
    """Return a deep copy of the default configuration."""
    return copy.deepcopy(DEFAULT_CONFIG)


def get_config() -> Dict[str, Any]:
    """Return a fresh deep copy of the application configuration."""
    return copy.deepcopy(DEFAULT_CONFIG)
