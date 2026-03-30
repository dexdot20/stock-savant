"""YFinance adapter package."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types

import requests


def _install_requests_fallback_shims() -> None:
    """Install compatibility shims only when ``curl_cffi`` is unavailable."""
    original_session_init = requests.Session.__init__

    def _patched_session_init(self, **kwargs):
        kwargs.pop("impersonate", None)
        original_session_init(self, **kwargs)

    requests.Session.__init__ = _patched_session_init

    if not hasattr(requests.cookies.RequestsCookieJar, "jar"):
        requests.cookies.RequestsCookieJar.jar = property(lambda self: self)

    if "curl_cffi" in sys.modules and sys.modules["curl_cffi"] is not None:
        return

    fake_requests = types.ModuleType("fake_requests")
    for attr in dir(requests):
        try:
            if attr not in ("Session", "session"):
                setattr(fake_requests, attr, getattr(requests, attr))
        except AttributeError:
            pass

    fake_requests.Session = requests.Session
    fake_session_module = types.ModuleType("session")
    fake_session_module.Session = requests.Session
    fake_requests.session = fake_session_module

    dummy_curl = types.ModuleType("curl_cffi")
    dummy_curl.requests = fake_requests
    sys.modules["curl_cffi"] = dummy_curl


if importlib.util.find_spec("curl_cffi") is None:
    _install_requests_fallback_shims()


from .adapter import YFinanceAdapterMixin

__all__ = ["YFinanceAdapterMixin"]


# --------------------------------------------------------------------------
# LOGGING FIX: Suppress expected 404 errors from yfinance
# --------------------------------------------------------------------------
class YFinanceFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        # Suppress "Not Found" errors for specific modules (e.g., Turkish stocks often lack these)
        if "404 Client Error" in msg and (
            "upgradeDowngradeHistory" in msg or "esgScores" in msg
        ):
            return False
        # Suppress "possibly delisted" errors (we handle these gracefully in our code)
        if "possibly delisted" in msg:
            return False
        return True


logging.getLogger("yfinance").addFilter(YFinanceFilter())
