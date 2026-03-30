"""
_load_member_map() cache katmanları için kapsamlı birim testler:
  - Bellek cache hızlı yol
  - Disk cache (taze / süresi dolmuş / eski format / bozuk JSON)
  - Atomic disk yazma (.tmp → rename)
  - force_refresh tüm katmanları atlar
  - Thread-safe double-checked locking (cache stampede koruması)
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import services.kap.scraper.member_scraper as ms
from services.kap.config import MEMBER_MAP_CACHE_TTL

from tests.kap_test_utils import build_member_rsc_html


# --------------------------------------------------------------------------- #
# Yardımcı
# --------------------------------------------------------------------------- #

SAMPLE_MAP = {"AKBNK": "oid-001", "THYAO": "oid-002"}


def _reset_cache():
    ms._member_map_cache = None
    ms._member_map_loaded_at = 0.0


def _write_cache_file(path: Path, data: dict, age_seconds: float = 0.0):
    payload = {"updated_at": time.time() - age_seconds, "data": data}
    path.write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 1. Bellek cache hızlı yol
# --------------------------------------------------------------------------- #

class TestMemoryCache:
    def setup_method(self):
        _reset_cache()

    def test_returns_memory_cache_without_disk_or_network(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ms, "MEMBER_MAP_CACHE_FILE", tmp_path / "cache.json")
        ms._member_map_cache = SAMPLE_MAP
        ms._member_map_loaded_at = time.time()

        result = ms._load_member_map()
        assert result is SAMPLE_MAP

    def test_memory_cache_expires_after_ttl(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(ms, "MEMBER_MAP_CACHE_FILE", cache_file)
        _write_cache_file(cache_file, SAMPLE_MAP, age_seconds=0.0)

        ms._member_map_cache = {"OLD": "stale"}
        ms._member_map_loaded_at = time.time() - MEMBER_MAP_CACHE_TTL - 1

        result = ms._load_member_map()
        assert result == SAMPLE_MAP


# --------------------------------------------------------------------------- #
# 2. Disk cache
# --------------------------------------------------------------------------- #

class TestDiskCache:
    def setup_method(self):
        _reset_cache()

    def test_fresh_disk_cache_used(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(ms, "MEMBER_MAP_CACHE_FILE", cache_file)
        _write_cache_file(cache_file, SAMPLE_MAP, age_seconds=10.0)

        result = ms._load_member_map()
        assert result == SAMPLE_MAP

    def test_expired_disk_cache_triggers_fetch(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(ms, "MEMBER_MAP_CACHE_FILE", cache_file)
        _write_cache_file(cache_file, SAMPLE_MAP, age_seconds=MEMBER_MAP_CACHE_TTL + 1)

        fresh_map = {"GARAN": "oid-003"}
        mock_resp = MagicMock()
        mock_resp.text = build_member_rsc_html(fresh_map)

        with patch("services.kap.scraper.member_scraper.fetch_with_retry", return_value=mock_resp):
            result = ms._load_member_map()

        assert result == fresh_map

    def test_old_format_cache_triggers_fetch(self, tmp_path, monkeypatch):
        """Eski format (saf dict, updated_at yok) → geçersiz sayılır."""
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(ms, "MEMBER_MAP_CACHE_FILE", cache_file)
        cache_file.write_text(json.dumps(SAMPLE_MAP), encoding="utf-8")

        fresh_map = {"GARAN": "oid-003"}
        mock_resp = MagicMock()
        mock_resp.text = build_member_rsc_html(fresh_map)

        with patch("services.kap.scraper.member_scraper.fetch_with_retry", return_value=mock_resp):
            result = ms._load_member_map()

        assert result == fresh_map

    def test_corrupted_cache_file_triggers_fetch(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(ms, "MEMBER_MAP_CACHE_FILE", cache_file)
        cache_file.write_text("bozuk json {{{{", encoding="utf-8")

        fresh_map = {"ISCTR": "oid-004"}
        mock_resp = MagicMock()
        mock_resp.text = build_member_rsc_html(fresh_map)

        with patch("services.kap.scraper.member_scraper.fetch_with_retry", return_value=mock_resp):
            result = ms._load_member_map()

        assert result == fresh_map


# --------------------------------------------------------------------------- #
# 3. Atomic disk yazma
# --------------------------------------------------------------------------- #

class TestAtomicDiskWrite:
    def setup_method(self):
        _reset_cache()

    def test_cache_file_written_with_metadata(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(ms, "MEMBER_MAP_CACHE_FILE", cache_file)

        fresh_map = {"AKBNK": "oid-001"}
        mock_resp = MagicMock()
        mock_resp.text = build_member_rsc_html(fresh_map)

        with patch("services.kap.scraper.member_scraper.fetch_with_retry", return_value=mock_resp):
            ms._load_member_map()

        assert cache_file.exists()
        stored = json.loads(cache_file.read_text(encoding="utf-8"))
        assert "updated_at" in stored
        assert "data" in stored
        assert stored["data"] == fresh_map

    def test_tmp_file_cleaned_up_after_write(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(ms, "MEMBER_MAP_CACHE_FILE", cache_file)

        mock_resp = MagicMock()
        mock_resp.text = build_member_rsc_html({"X": "y"})

        with patch("services.kap.scraper.member_scraper.fetch_with_retry", return_value=mock_resp):
            ms._load_member_map()

        assert not cache_file.with_suffix(".tmp").exists()


# --------------------------------------------------------------------------- #
# 4. force_refresh
# --------------------------------------------------------------------------- #

class TestForceRefresh:
    def setup_method(self):
        _reset_cache()

    def test_force_refresh_bypasses_fresh_memory_cache(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(ms, "MEMBER_MAP_CACHE_FILE", cache_file)
        ms._member_map_cache = {"STALE": "oid-stale"}
        ms._member_map_loaded_at = time.time()

        fresh_map = {"FRESH": "oid-fresh"}
        mock_resp = MagicMock()
        mock_resp.text = build_member_rsc_html(fresh_map)

        with patch("services.kap.scraper.member_scraper.fetch_with_retry", return_value=mock_resp):
            result = ms._load_member_map(force_refresh=True)

        assert result == fresh_map

    def test_force_refresh_bypasses_fresh_disk_cache(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(ms, "MEMBER_MAP_CACHE_FILE", cache_file)
        _write_cache_file(cache_file, {"DISK": "oid-disk"}, age_seconds=0.0)

        fresh_map = {"NET": "oid-net"}
        mock_resp = MagicMock()
        mock_resp.text = build_member_rsc_html(fresh_map)

        with patch("services.kap.scraper.member_scraper.fetch_with_retry", return_value=mock_resp):
            result = ms._load_member_map(force_refresh=True)

        assert result == fresh_map

    def test_lookup_noninteractive_force_refresh(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(ms, "MEMBER_MAP_CACHE_FILE", cache_file)
        ms._member_map_cache = {"AKBNK": "oid-old"}
        ms._member_map_loaded_at = time.time()

        fresh_map = {"AKBNK": "oid-new"}
        mock_resp = MagicMock()
        mock_resp.text = build_member_rsc_html(fresh_map)

        with patch("services.kap.scraper.member_scraper.fetch_with_retry", return_value=mock_resp):
            result = ms.lookup_member_oid_noninteractive("AKBNK", force_refresh=True)

        assert result["oid"] == "oid-new"
        assert result["found"] is True


# --------------------------------------------------------------------------- #
# 5. Thread-safe double-checked locking (cache stampede)
# --------------------------------------------------------------------------- #

class TestCacheStampede:
    def setup_method(self):
        _reset_cache()

    def test_concurrent_calls_hit_network_only_once(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(ms, "MEMBER_MAP_CACHE_FILE", cache_file)

        call_count = 0

        def slow_fetch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            time.sleep(0.05)
            resp = MagicMock()
            resp.text = build_member_rsc_html({"AKBNK": "oid-001"})
            return resp

        results = []
        errors = []

        def worker():
            try:
                results.append(ms._load_member_map())
            except Exception as e:
                errors.append(e)

        with patch("services.kap.scraper.member_scraper.fetch_with_retry", side_effect=slow_fetch):
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors
        assert len(results) == 10
        assert all(r == {"AKBNK": "oid-001"} for r in results)
        assert call_count == 1
