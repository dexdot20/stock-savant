"""
services.kap.proxy_manager birim testleri.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from services.kap.proxy_manager import ProxyManager, fetch_with_retry


# --------------------------------------------------------------------------- #
# Yardımcı
# --------------------------------------------------------------------------- #

def _make_proxy_file(tmp_path: Path, lines: list[str]) -> Path:
    proxy_file = tmp_path / "proxies.txt"
    proxy_file.write_text("\n".join(lines), encoding="utf-8")
    return proxy_file


# --------------------------------------------------------------------------- #
# ProxyManager — yükleme
# --------------------------------------------------------------------------- #

class TestProxyManagerLoad:
    def test_empty_file_no_proxies(self, tmp_path):
        f = _make_proxy_file(tmp_path, [])
        mgr = ProxyManager(proxy_file=f)
        assert len(mgr) == 0

    def test_missing_file_no_error(self, tmp_path):
        mgr = ProxyManager(proxy_file=tmp_path / "nonexistent.txt")
        assert len(mgr) == 0

    def test_comments_and_blank_lines_ignored(self, tmp_path):
        f = _make_proxy_file(
            tmp_path,
            ["# yorum", "", "http://proxy1:8080", "  ", "http://proxy2:3128"],
        )
        mgr = ProxyManager(proxy_file=f)
        assert len(mgr) == 2

    def test_proxy_urls_loaded_correctly(self, tmp_path):
        urls = ["http://proxy1:8080", "http://user:pass@proxy2:3128"]
        f = _make_proxy_file(tmp_path, urls)
        mgr = ProxyManager(proxy_file=f)
        statuses = mgr.status()
        assert [p["url"] for p in statuses] == urls


# --------------------------------------------------------------------------- #
# ProxyManager — rotasyon
# --------------------------------------------------------------------------- #

class TestProxyManagerRotation:
    def test_get_next_round_robin(self, tmp_path):
        f = _make_proxy_file(tmp_path, ["http://p1:80", "http://p2:80", "http://p3:80"])
        mgr = ProxyManager(proxy_file=f)
        results = [mgr.get_next() for _ in range(6)]
        assert results == [
            "http://p1:80", "http://p2:80", "http://p3:80",
            "http://p1:80", "http://p2:80", "http://p3:80",
        ]

    def test_get_next_returns_none_when_empty(self, tmp_path):
        f = _make_proxy_file(tmp_path, [])
        mgr = ProxyManager(proxy_file=f)
        assert mgr.get_next() is None

    def test_get_next_skips_disabled(self, tmp_path):
        f = _make_proxy_file(tmp_path, ["http://p1:80", "http://p2:80"])
        mgr = ProxyManager(proxy_file=f, error_threshold=1)
        mgr.mark_error("http://p1:80")
        for _ in range(4):
            assert mgr.get_next() == "http://p2:80"

    def test_get_next_returns_none_all_disabled(self, tmp_path):
        f = _make_proxy_file(tmp_path, ["http://p1:80"])
        mgr = ProxyManager(proxy_file=f, error_threshold=1)
        mgr.mark_error("http://p1:80")
        assert mgr.get_next() is None


# --------------------------------------------------------------------------- #
# ProxyManager — hata ve devre dışı bırakma
# --------------------------------------------------------------------------- #

class TestProxyManagerErrors:
    def test_mark_error_increments_count(self, tmp_path):
        f = _make_proxy_file(tmp_path, ["http://p1:80"])
        mgr = ProxyManager(proxy_file=f, error_threshold=3)
        mgr.mark_error("http://p1:80")
        mgr.mark_error("http://p1:80")
        status = mgr.status()[0]
        assert status["error_count"] == 2
        assert status["active"] is True

    def test_mark_error_disables_at_threshold(self, tmp_path):
        f = _make_proxy_file(tmp_path, ["http://p1:80"])
        mgr = ProxyManager(proxy_file=f, error_threshold=3)
        for _ in range(3):
            mgr.mark_error("http://p1:80")
        assert mgr.status()[0]["active"] is False

    def test_mark_success_resets_error_count(self, tmp_path):
        f = _make_proxy_file(tmp_path, ["http://p1:80"])
        mgr = ProxyManager(proxy_file=f, error_threshold=3)
        mgr.mark_error("http://p1:80")
        mgr.mark_error("http://p1:80")
        mgr.mark_success("http://p1:80")
        assert mgr.status()[0]["error_count"] == 0

    def test_unknown_proxy_mark_error_no_crash(self, tmp_path):
        f = _make_proxy_file(tmp_path, ["http://p1:80"])
        mgr = ProxyManager(proxy_file=f)
        mgr.mark_error("http://unknown:80")


# --------------------------------------------------------------------------- #
# ProxyManager — reset (zaman tabanlı)
# --------------------------------------------------------------------------- #

class TestProxyManagerReset:
    def test_disabled_proxy_resets_after_interval(self, tmp_path):
        f = _make_proxy_file(tmp_path, ["http://p1:80"])
        mgr = ProxyManager(proxy_file=f, error_threshold=1, reset_interval=0.05)
        mgr.mark_error("http://p1:80")
        assert mgr.get_next() is None
        time.sleep(0.1)
        assert mgr.get_next() == "http://p1:80"

    def test_status_shows_reset_in_seconds(self, tmp_path):
        f = _make_proxy_file(tmp_path, ["http://p1:80"])
        mgr = ProxyManager(proxy_file=f, error_threshold=1, reset_interval=60.0)
        mgr.mark_error("http://p1:80")
        status = mgr.status()[0]
        assert "reset_in_seconds" in status
        assert status["reset_in_seconds"] <= 60.0


# --------------------------------------------------------------------------- #
# fetch_with_retry
# --------------------------------------------------------------------------- #

class TestFetchWithRetry:
    def test_success_no_proxy(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("services.kap.proxy_manager.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            result = fetch_with_retry("get", "http://example.com", headers={}, timeout=5)
            assert result is mock_resp

    def test_proxy_marked_success_on_ok(self, tmp_path):
        f = _make_proxy_file(tmp_path, ["http://p1:80"])
        mgr = ProxyManager(proxy_file=f)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("services.kap.proxy_manager.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            fetch_with_retry("get", "http://example.com", headers={}, timeout=5, proxy_manager=mgr)
            assert mgr.status()[0]["error_count"] == 0

    def test_proxy_marked_error_on_failure_then_retry(self, tmp_path):
        import httpx as _httpx

        f = _make_proxy_file(tmp_path, ["http://p1:80", "http://p2:80"])
        mgr = ProxyManager(proxy_file=f, error_threshold=3)
        call_count = 0
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        def fake_client_factory(**kwargs):
            nonlocal call_count
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)

            def fake_get(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise _httpx.ConnectError("bağlantı yok")
                return mock_resp

            mock_client.get = fake_get
            return mock_client

        with patch("services.kap.proxy_manager.httpx.Client", side_effect=fake_client_factory):
            result = fetch_with_retry(
                "get", "http://example.com", headers={}, timeout=5,
                proxy_manager=mgr, max_retries=3
            )
            assert result is mock_resp
            assert call_count == 2
            assert mgr.status()[0]["error_count"] == 1
