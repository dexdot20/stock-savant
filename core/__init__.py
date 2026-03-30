"""
Core Package - Core Infrastructure Modules
===========================================

This package contains the core infrastructure components of the Struct application:
- Exception classes (exceptions.py) - Detailed error management
- Logging standard (logging.py) - File-based logging (log.txt)

Usage:
    >>> from core import BorsaException, get_standard_logger
    >>> logger = get_standard_logger(__name__)
    >>> logger.info("Operation succeeded")
    >>> raise APIError("API call failed", status_code=500)
"""

from .exceptions import (
    BorsaException,
    ConfigError,
    APIError,
    PaymentRequiredError,
    RateLimitError,
    ValidationError,
    CacheError,
    DataProcessingError,
    AIWorkflowError,
    NetworkError,
    DatabaseError,
    handle_exception,
    safe_execute,
)
from .logging import (
    get_standard_logger,
    log_exception,
    log_operation_start,
    log_operation_end,
    LOG_FILE_PATH,
)
from .console import console

# Public API
__all__ = [
    # Console
    "console",
    # Exceptions
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
    # Exception helpers
    "handle_exception",
    "safe_execute",
    # Logging
    "get_standard_logger",
    "log_exception",
    "log_operation_start",
    "log_operation_end",
    "LOG_FILE_PATH",
]
