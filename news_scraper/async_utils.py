"""
Async helpers - safely run coroutines from sync calls.
"""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine


def run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run coroutine in synchronous context.

    Note: If an active event loop is running, use async methods instead of sync wrapper.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Active event loop detected; use async methods instead.")
