"""Core Logging - Central logging standard.

This module provides the application-wide logger.
All logs are written both to console and to log.txt.
"""

import logging
import os
import sys
import threading
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional, Any, Dict

# ═══════════════════════════════════════════════════════════════════════════════
# LOG CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

from .paths import get_runtime_dir, get_app_root

# Log file location - log.txt in the writable runtime directory
LOG_FILE_PATH = get_runtime_dir() / "log.txt"


# Load configuration
def _load_logging_config():
    # Environment variables take precedence; otherwise use defaults
    return {
        "format": os.environ.get(
            "LOGGING_FORMAT", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ),
        "date_format": os.environ.get("LOGGING_DATE_FORMAT", "%Y-%m-%d %H:%M:%S"),
        "console_level": os.environ.get("LOGGING_CONSOLE_LEVEL", "WARNING"),
        "file_level": os.environ.get("LOGGING_FILE_LEVEL", "INFO"),
    }


_logging_cfg = _load_logging_config()
LOG_FORMAT = _logging_cfg["format"]
LOG_DATE_FORMAT = _logging_cfg["date_format"]
LOG_LEVEL = logging.DEBUG  # Internal: capture all levels, handlers filter appropriately
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5  # Keep 5 backup files

# Thread-safe initialization lock
_logger_lock = threading.Lock()
_initialized_loggers: Dict[str, logging.Logger] = {}
_root_configured = False


class DetailedFormatter(logging.Formatter):
    """Provide detailed, optionally colorized log formatting."""

    LEVEL_COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = False):
        super().__init__(LOG_FORMAT, LOG_DATE_FORMAT)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        # Apply the base format
        formatted = super().format(record)

        # Append exception information if present
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)

        if record.exc_text:
            formatted = f"{formatted}\n{record.exc_text}"

        # Append stack information if present
        if record.stack_info:
            formatted = f"{formatted}\n{record.stack_info}"

        # Apply color output only for the console
        if self.use_colors and record.levelno in self.LEVEL_COLORS:
            color = self.LEVEL_COLORS[record.levelno]
            formatted = f"{color}{formatted}{self.RESET}"

        return formatted


def _setup_root_logger() -> None:
    """Configure the root logger once."""
    global _root_configured

    if _root_configured:
        return

    with _logger_lock:
        if _root_configured:
            return

        root_logger = logging.getLogger()
        root_logger.setLevel(LOG_LEVEL)

        # Clear existing handlers
        root_logger.handlers.clear()

        # Resolve log levels from the environment
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        # Read values from config, falling back to environment variables
        console_level_str = _logging_cfg["console_level"].upper()
        file_level_str = _logging_cfg["file_level"].upper()

        console_level = level_map.get(console_level_str, logging.WARNING)
        file_level = level_map.get(file_level_str, logging.DEBUG)

        # ═══════════════════════════════════════════════════════════════════════
        # FILE HANDLER - All logs are written to log.txt
        # ═══════════════════════════════════════════════════════════════════════
        try:
            # Ensure the log directory exists
            LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                LOG_FILE_PATH,
                maxBytes=MAX_LOG_SIZE,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(file_level)
            file_handler.setFormatter(DetailedFormatter(use_colors=False))
            root_logger.addHandler(file_handler)

            # Startup log entry
            separator = "=" * 80
            start_msg = f"\n{separator}\n"
            start_msg += f"  STRUCT CLI - APPLICATION STARTED\n"
            start_msg += f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            start_msg += f"  Python: {sys.version}\n"
            start_msg += f"  Platform: {sys.platform}\n"
            start_msg += f"  Working Directory: {os.getcwd()}\n"
            start_msg += f"  App Root: {get_app_root()}\n"
            start_msg += f"  Runtime Dir: {get_runtime_dir()}\n"
            start_msg += f"  Log File: {LOG_FILE_PATH}\n"
            start_msg += (
                f"  Settings File: {get_app_root() / 'config/settings.py'}\n"
            )
            start_msg += f"  Log Levels: Console={console_level_str}, File={file_level_str}\n"
            start_msg += f"{separator}"

            file_handler.emit(
                logging.LogRecord(
                    name="SYSTEM",
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=start_msg,
                    args=(),
                    exc_info=None,
                )
            )

        except Exception as e:
            # Report to console if the file handler cannot be created
            print(f"[WARNING] Failed to create log file: {e}", file=sys.stderr)

        # ═══════════════════════════════════════════════════════════════════════
        # CONSOLE HANDLER - Rich used for better terminal output and Status compatibility
        # ═══════════════════════════════════════════════════════════════════════
        try:
            from rich.logging import RichHandler
            from .console import console

            console_handler = RichHandler(
                console=console,
                level=console_level,
                show_time=True,
                show_path=False,
                markup=True,
                rich_tracebacks=True,
            )
            # Use a simpler format for RichHandler because it adds its own time and level
            console_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        except ImportError:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(console_level)
            console_handler.setFormatter(DetailedFormatter(use_colors=True))

        root_logger.addHandler(console_handler)

        # ═══════════════════════════════════════════════════════════════════════
        # THIRD-PARTY LOGGER CONFIGURATION
        # ═══════════════════════════════════════════════════════════════════════
        # Suppress noisy third-party logs already handled by the app.
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("h11").setLevel(logging.WARNING)
        logging.getLogger("h2").setLevel(logging.WARNING)
        logging.getLogger("aiohttp").setLevel(logging.WARNING)
        logging.getLogger("curl_cffi").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(
            logging.WARNING
        )

        _root_configured = True


def get_standard_logger(name: str = "borsa_ai") -> logging.Logger:
    """Return a configured standard logger instance.

    All logs are automatically written to log.txt.

    Args:
        name: Logger name (default: "borsa_ai")

    Returns:
        Configured logger instance
    """
    # Configure the root logger
    _setup_root_logger()

    # Check cached loggers
    if name in _initialized_loggers:
        return _initialized_loggers[name]

    with _logger_lock:
        if name in _initialized_loggers:
            return _initialized_loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(LOG_LEVEL)

        _initialized_loggers[name] = logger
        return logger


def log_exception(
    logger: logging.Logger,
    message: str,
    exc: Exception,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log an exception with detailed context."""
    error_details = [
        f"ERROR: {message}",
        f"Type: {type(exc).__name__}",
        f"Message: {str(exc)}",
    ]

    if context:
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        error_details.append(f"Context: {context_str}")

    # Capture the stack trace
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    error_details.append(f"Stack Trace:\n{''.join(tb)}")

    logger.error("\n".join(error_details))


def log_operation_start(
    logger: logging.Logger,
    operation: str,
    details: Optional[Dict[str, Any]] = None,
) -> datetime:
    """Log the operation start and return the start time."""
    detail_str = ""
    if details:
        detail_str = " | " + ", ".join([f"{k}={v}" for k, v in details.items()])

    logger.info(f"▶▶ STARTED: {operation}{detail_str}")
    return datetime.now()


def log_operation_end(
    logger: logging.Logger,
    operation: str,
    start_time: datetime,
    success: bool = True,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Log the operation end."""
    elapsed = (datetime.now() - start_time).total_seconds()
    status = "✓ COMPLETED" if success else "✗ FAILED"

    detail_str = ""
    if details:
        detail_str = " | " + ", ".join([f"{k}={v}" for k, v in details.items()])

    log_method = logger.info if success else logger.error
    log_method(f"◀◀ {status}: {operation} ({elapsed:.3f}s){detail_str}")


# ═══════════════════════════════════════════════════════════════════════════════
# AI DEBUG LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

# We control AI debug mode here.
# Defining it in code instead of .env prevents users from changing it after compiling to an .exe.
AI_DEBUG = False

AI_DEBUG_LOG_FILE_PATH = get_runtime_dir() / "ai_debug.log"
_ai_debug_logger: Optional[logging.Logger] = None


def ai_debug_enabled() -> bool:
    """Check whether AI debug mode is enabled.

    This reads the AI_DEBUG constant defined in code.
    """
    return AI_DEBUG


class AIDebugLogger:
    """Custom logger wrapper for AI debug operations."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)


def get_ai_debug_logger() -> AIDebugLogger:
    """Return the AI debug logger (Singleton pattern).

    Creates it if it does not already exist.
    """
    global _ai_debug_logger

    if _ai_debug_logger is not None:
        return AIDebugLogger(_ai_debug_logger)

    logger = logging.getLogger("ai_debug")
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    try:
        file_handler = logging.FileHandler(
            AI_DEBUG_LOG_FILE_PATH, mode="a", encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.propagate = False
    except Exception as e:
        sys.stderr.write(f"AI Debug logger could not be created: {e}\n")

    _ai_debug_logger = logger
    return AIDebugLogger(_ai_debug_logger)


# Public API
__all__ = [
    "get_standard_logger",
    "log_exception",
    "log_operation_start",
    "log_operation_end",
    "DetailedFormatter",
    "LOG_FILE_PATH",
    "get_ai_debug_logger",
    "ai_debug_enabled",
    "AIDebugLogger",
]
