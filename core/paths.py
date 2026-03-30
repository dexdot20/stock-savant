"""Application path helpers and Nuitka compiled-binary compatibility."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Union

PathLike = Union[str, Path]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_frozen() -> bool:
    """Return True when running as a compiled Nuitka/onefile binary."""
    return bool(getattr(sys, "frozen", False) or getattr(sys, "nuitka", False))


def get_app_root() -> Path:
    """Return the application root directory where source assets live.

    Resolution order (Nuitka --standalone mode):
    1. sys.argv[0] parent directory (preferred)
    2. sys.executable parent directory (fallback)

    Development mode: returns the project root (two levels above this file).
    """
    # 1. Compiled binary — use executable location
    if _is_frozen():
        if hasattr(sys, "argv") and sys.argv and sys.argv[0]:
            return Path(sys.argv[0]).resolve().parent
        return Path(sys.executable).resolve().parent

    # 2. Development environment — this file lives at core/paths.py
    return Path(__file__).resolve().parent.parent


def get_runtime_dir() -> Path:
    r"""Return the writable runtime directory, creating it if necessary.

    Resolution order:
    1. STRUCT_RUNTIME_DIR environment variable (explicit override)
    2. Application root returned by get_app_root() (default; portable)
    3. %LOCALAPPDATA%\Struct fallback if the app root is not writable
    """
    custom = os.getenv("STRUCT_RUNTIME_DIR")
    if custom:
        runtime_dir = Path(custom).expanduser()
        runtime_dir.mkdir(parents=True, exist_ok=True)
        return runtime_dir

    # Default: same directory as the executable (portable layout)
    runtime_dir = get_app_root()

    try:
        # Ensure directory exists and perform a write-permission test
        runtime_dir.mkdir(parents=True, exist_ok=True)
        test_file = runtime_dir / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except (IOError, PermissionError):
            raise PermissionError("No write permission to the application directory")

    except (IOError, PermissionError, OSError):
        # Fall back to %LOCALAPPDATA%\Struct when the app root is read-only
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
        if base:
            runtime_dir = Path(base) / "Struct"
            runtime_dir.mkdir(parents=True, exist_ok=True)

    return runtime_dir


def resolve_path(path: PathLike, base_dir: Optional[Path] = None) -> Path:
    """Resolve a relative path against base_dir (defaults to app root)."""
    base_dir = base_dir or get_app_root()
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return base_dir / resolved


def get_data_path(relative_path: PathLike) -> Path:
    """Return the absolute path for an in-application data asset."""
    return resolve_path(relative_path, get_app_root())


def get_instance_dir() -> Path:
    """Return the shared writable instance directory."""
    return _ensure_dir(get_runtime_dir() / "instance")


def get_history_dir() -> Path:
    """Return the directory used for persisted analysis history."""
    return _ensure_dir(get_instance_dir() / "history")


def get_analysis_cache_dir() -> Path:
    """Return the directory used for daily analysis cache files."""
    return _ensure_dir(get_instance_dir() / "cache" / "analysis_daily")


def get_favorites_path() -> Path:
    """Return the path to the persisted favorites JSON file."""
    return get_instance_dir() / "favorites.json"


def get_portfolio_path() -> Path:
    """Return the path to the persisted portfolio JSON file."""
    return get_instance_dir() / "portfolio.json"


def get_sessions_path() -> Path:
    """Return the directory that stores Free Mode session snapshots."""
    return _ensure_dir(get_instance_dir() / "sessions")


def get_ai_reports_path() -> Path:
    """Return the append-only JSONL path for developer-facing AI reports."""
    logs_dir = _ensure_dir(get_instance_dir() / "logs")
    return logs_dir / "ai_reports.jsonl"


def get_python_exec_dir() -> Path:
    """Return the writable sandbox root for bounded Python execution."""
    return _ensure_dir(get_instance_dir() / "python_exec")


def get_alerts_path() -> Path:
    """Return the path to the persistent price-alerts JSON file."""
    alerts_path = get_instance_dir() / "alerts.json"
    return alerts_path


def get_tool_health_path() -> Path:
    """Return the path to the persistent tool-health JSON snapshot."""
    health_dir = _ensure_dir(get_instance_dir() / "health")
    return health_dir / "tool_health.json"


def get_kap_state_path() -> Path:
    """Return the path to the persistent KAP intelligence state file."""
    kap_dir = _ensure_dir(get_instance_dir() / "kap")
    return kap_dir / "kap_state.json"


def get_investor_profile_path() -> Path:
    """Return the path to the persisted investor profile JSON file."""
    profile_dir = _ensure_dir(get_instance_dir() / "profiles")
    return profile_dir / "investor_profile.json"


def ensure_parent_dir(path: PathLike) -> Path:
    """Ensure the parent directory for a file path exists."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_json_file(path: PathLike, default_payload: Any) -> Path:
    """Ensure a JSON file exists with a default payload when missing."""
    resolved = ensure_parent_dir(path)
    if not resolved.exists():
        with open(resolved, "w", encoding="utf-8") as handle:
            json.dump(default_payload, handle, ensure_ascii=False, indent=2)
    return resolved
