"""
Core Exceptions - Application-wide exception types
===================================================

This module contains all exception classes used by the Struct application.
It is designed for centralized exception management.

Features:
- Automatic logging support
- Contextual error information
- Exception chaining
- Detailed error reporting
"""

import traceback
from datetime import datetime
from typing import Optional, Any, Dict, List
import threading

from .logging import get_standard_logger

# Dedicated logger for the exception module
_exception_logger = get_standard_logger("exceptions")

# Thread-safe monotonic error counter
_error_counter_lock = threading.Lock()
_error_counter = 0


def _next_error_seq() -> int:
    global _error_counter
    with _error_counter_lock:
        _error_counter += 1
        return _error_counter


class BorsaException(Exception):
    """
    Base exception class for the Struct application.

    All application exceptions inherit from this class.
    It provides automatic logging and detailed error reporting.
    """

    def __init__(
        self,
        message: str,
        context: Optional[str] = None,
        original_error: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
        auto_log: bool = True,
    ):
        """
        Args:
            message: Error message
            context: Context where the error occurred (e.g. "MarketDataService.fetch_data")
            original_error: Original exception, if any
            details: Additional error details (dict)
            auto_log: Whether to log automatically (default: True)
        """
        self.message = message
        self.context = context
        self.original_error = original_error
        self.details = details or {}
        self.timestamp = datetime.now()
        self.error_id = self._generate_error_id()

        # Build the full error message
        self.full_message = self._build_full_message()
        super().__init__(self.full_message)

        # Otomatik loglama
        if auto_log:
            self._log_exception()

    def _generate_error_id(self) -> str:
        """Generates a unique, monotonically increasing error ID."""
        seq = _next_error_seq()
        return f"ERR-{self.timestamp.strftime('%Y%m%d%H%M%S')}-{seq:06d}"

    def _build_full_message(self) -> str:
        """Build the full error message."""
        parts = []

        if self.context:
            parts.append(f"[{self.context}]")

        parts.append(self.message)

        if self.original_error:
            parts.append(
                f"(Original error: {type(self.original_error).__name__}: {self.original_error})"
            )

        return " ".join(parts)

    def _log_exception(self) -> None:
        """Logs the exception to the log file."""
        log_lines = [
            "",
            "╔" + "═" * 78 + "╗",
            f"║ ERROR REPORT - {self.error_id}".ljust(79) + "║",
            "╠" + "═" * 78 + "╣",
            f"║ Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}".ljust(79) + "║",
            f"║ Type: {type(self).__name__}".ljust(79) + "║",
            f"║ Context: {self.context or 'Not specified'}".ljust(79) + "║",
        ]

        # Split message into lines
        msg_lines = self.message.split("\n")
        log_lines.append(f"║ Message: {msg_lines[0][:60]}".ljust(79) + "║")
        for line in msg_lines[1:]:
            log_lines.append(f"║          {line[:60]}".ljust(79) + "║")

        # Add details
        if self.details:
            log_lines.append("╠" + "─" * 78 + "╣")
            log_lines.append("║ Details:".ljust(79) + "║")
            for key, value in self.details.items():
                val_str = str(value)[:55]
                log_lines.append(f"║   • {key}: {val_str}".ljust(79) + "║")

        # Add original error if exists
        if self.original_error:
            log_lines.append("╠" + "─" * 78 + "╣")
            log_lines.append("║ Original Error:".ljust(79) + "║")
            log_lines.append(
                f"║   Type: {type(self.original_error).__name__}".ljust(79) + "║"
            )
            orig_msg = str(self.original_error)[:60]
            log_lines.append(f"║   Message: {orig_msg}".ljust(79) + "║")

            # Stack trace
            if self.original_error.__traceback__:
                log_lines.append("║   Stack Trace:".ljust(79) + "║")
                tb_lines = traceback.format_exception(
                    type(self.original_error),
                    self.original_error,
                    self.original_error.__traceback__,
                )
                for tb_line in tb_lines:
                    for sub_line in tb_line.strip().split("\n"):
                        truncated = sub_line[:70]
                        log_lines.append(f"║     {truncated}".ljust(79) + "║")

        log_lines.append("╚" + "═" * 78 + "╝")
        log_lines.append("")

        _exception_logger.error("\n".join(log_lines))

    def to_dict(self) -> Dict[str, Any]:
        """Convert the exception to a dictionary."""
        return {
            "error_id": self.error_id,
            "type": type(self).__name__,
            "message": self.message,
            "context": self.context,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "original_error": str(self.original_error) if self.original_error else None,
        }

    def get_user_message(self) -> str:
        """Returns a safe message to be displayed to the user."""
        # Filter sensitive information
        return f"An error occurred: {self.message} (Error ID: {self.error_id})"


class ConfigError(BorsaException):
    """Configuration-related errors."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        expected_value: Optional[str] = None,
        actual_value: Optional[Any] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if config_key:
            details["config_key"] = config_key
        if expected_value:
            details["expected_value"] = expected_value
        if actual_value is not None:
            details["actual_value"] = str(actual_value)[:100]

        super().__init__(message, details=details, **kwargs)


class APIError(BorsaException):
    """Errors related to API calls.

    Examples:
    - HTTP errors
    - Rate limiting
    - Timeout
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        response_body: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if status_code:
            details["status_code"] = status_code
        if endpoint:
            details["endpoint"] = endpoint
        if response_body:
            details["response_body"] = response_body[:500]

        super().__init__(message, details=details, **kwargs)


class PaymentRequiredError(APIError):
    """
    API payment required errors (402 Payment Required).
    """

    def __init__(self, message: str = "Payment required for API access", **kwargs):
        kwargs.setdefault("status_code", 402)
        super().__init__(message, **kwargs)


class RateLimitError(APIError):
    """
    Rate limiting errors (429 Too Many Requests).
    """

    def __init__(
        self,
        message: str = "API rate limit exceeded",
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault("status_code", 429)
        details = kwargs.pop("details", {})
        if retry_after:
            details["retry_after_seconds"] = retry_after
        super().__init__(message, details=details, **kwargs)
        self.retry_after = retry_after


class ValidationError(BorsaException):
    """Data validation errors.

    Examples:
    - Invalid ticker symbol
    - Missing required field
    - Format error
    """

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        validation_rules: Optional[List[str]] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if field_name:
            details["field_name"] = field_name
        if invalid_value is not None:
            details["invalid_value"] = str(invalid_value)[:100]
        if validation_rules:
            details["validation_rules"] = ", ".join(validation_rules)

        super().__init__(message, details=details, **kwargs)


class CacheError(BorsaException):
    """Errors related to cache operations.

    Examples:
    - Cache file could not be read
    - Invalid cache data
    - Insufficient disk space
    """

    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        cache_path: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if cache_key:
            details["cache_key"] = cache_key
        if cache_path:
            details["cache_path"] = cache_path

        super().__init__(message, details=details, **kwargs)


class DataProcessingError(BorsaException):
    """Data processing errors.

    Examples:
    - JSON parse error
    - Data conversion error
    - Missing data
    """

    def __init__(
        self,
        message: str,
        data_type: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if data_type:
            details["data_type"] = data_type
        if operation:
            details["operation"] = operation

        super().__init__(message, details=details, **kwargs)


class AIWorkflowError(BorsaException):
    """AI workflow errors.

    Examples:
    - AI provider access error
    - Prompt processing error
    - Model response error
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        phase: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if provider:
            details["provider"] = provider
        if model:
            details["model"] = model
        if phase:
            details["phase"] = phase

        super().__init__(message, details=details, **kwargs)


class NetworkError(BorsaException):
    """Network connection errors.

    Examples:
    - Connection refused
    - Timeout
    - DNS resolution error
    """

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if url:
            details["url"] = url
        if timeout:
            details["timeout_seconds"] = timeout

        super().__init__(message, details=details, **kwargs)


class DatabaseError(BorsaException):
    """Database operation errors.

    Examples:
    - Connection error
    - Query error
    - Transaction error
    """

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        table: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if query:
            # Truncate the query for safety
            details["query"] = query[:200] + "..." if len(query) > 200 else query
        if table:
            details["table"] = table

        super().__init__(message, details=details, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTION HANDLER UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def handle_exception(
    exc: Exception,
    context: Optional[str] = None,
    reraise: bool = True,
    fallback_value: Any = None,
) -> Any:
    """
    Handle exceptions centrally.

    Args:
        exc: Exception to handle
        context: Error context
        reraise: Whether to re-raise the exception
        fallback_value: Value returned when handling fails

    Returns:
        fallback_value if reraise is False

    Raises:
        BorsaException: If reraise=True and exc is not already a BorsaException

    Usage:
        try:
            risky_operation()
        except Exception as e:
            return handle_exception(e, "MyClass.my_method", reraise=False, fallback_value=None)
    """
    # No need to log again if it is already a BorsaException
    if isinstance(exc, BorsaException):
        if reraise:
            raise
        return fallback_value

    # Convert other exceptions to BorsaException
    wrapped = BorsaException(
        message=str(exc),
        context=context,
        original_error=exc,
    )

    if reraise:
        raise wrapped from exc

    return fallback_value


def safe_execute(
    func,
    *args,
    context: Optional[str] = None,
    fallback_value: Any = None,
    log_errors: bool = True,
    **kwargs,
) -> Any:
    """
    Execute a function safely.

    Args:
        func: Function to execute
        *args: Positional function arguments
        context: Error context
        fallback_value: Value returned on error
        log_errors: Whether to log errors
        **kwargs: Keyword function arguments

    Returns:
        Function result or fallback_value

    Usage:
        result = safe_execute(risky_function, arg1, arg2, context="MyService", fallback_value=[])
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            if not isinstance(e, BorsaException):
                _exception_logger.warning(
                    "safe_execute failed [%s.%s]: %s: %s",
                    getattr(func, "__module__", "?"),
                    getattr(func, "__name__", "?"),
                    type(e).__name__,
                    e,
                )
            # BorsaException instances are already auto-logged on construction.
        return fallback_value


# Public API
__all__ = [
    "BorsaException",
    "ConfigError",
    "APIError",
    "PaymentRequiredError",
    "RateLimitError",
    "ValidationError",
    "CacheError",
    "DataProcessingError",
    "AIWorkflowError",
    "NetworkError",
    "DatabaseError",
    "handle_exception",
    "safe_execute",
]
