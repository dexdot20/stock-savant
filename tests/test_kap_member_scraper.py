"""
member_scraper modülü birim testleri.
Cache testleri için test_kap_member_cache.py'ye bakın.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import services.kap.scraper.member_scraper as ms
from services.kap.scraper.member_scraper import (
    _extract_rsc_text,
    list_disclosures,
    lookup_member_oid_noninteractive,
)

from tests.kap_test_utils import SAMPLE_MEMBER_MAP, build_member_rsc_html


# --------------------------------------------------------------------------- #
# Yardımcı
# --------------------------------------------------------------------------- #

def _preset_cache(mapping: dict | None = None) -> None:
    ms._member_map_cache = mapping if mapping is not None else dict(SAMPLE_MEMBER_MAP)
    ms._member_map_loaded_at = time.time()


def _reset_cache() -> None:
    ms._member_map_cache = None
    ms._member_map_loaded_at = 0.0


# --------------------------------------------------------------------------- #
# _extract_rsc_text
# --------------------------------------------------------------------------- #

class TestExtractRscText:
    def test_extracts_plain_content(self):
        html = '<script>self.__next_f.push([1,"hello"])</script>'
        assert _extract_rsc_text(html) == "hello"

    def test_ignores_non_nextf_scripts(self):
        html = '<script>window.foo = 1</script>'
        assert _extract_rsc_text(html) == ""

    def test_multiple_pushes_joined(self):
        html = (
            '<script>self.__next_f.push([1,"alpha"])</script>'
            '<script>self.__next_f.push([1,"beta"])</script>'
        )
        result = _extract_rsc_text(html)
        assert "alpha" in result
        assert "beta" in result

    def test_extracts_member_oid_and_stock_code(self):
        html = (
            '<script>self.__next_f.push([1,'
            '"\\"mkkMemberOid\\":\\"oid-001\\",\\"stockCode\\":\\"akbnk\\""])'
            '</script>'
        )
        result = _extract_rsc_text(html)
        assert '"mkkMemberOid":"oid-001"' in result
        assert '"stockCode":"akbnk"' in result

    def test_empty_html(self):
        assert _extract_rsc_text("") == ""


# --------------------------------------------------------------------------- #
# lookup_member_oid_noninteractive
# --------------------------------------------------------------------------- #

class TestLookupMemberOidNoninteractive:
    def setup_method(self):
        _preset_cache()

    def teardown_method(self):
        _reset_cache()

    def test_exact_match_found(self):
        result = lookup_member_oid_noninteractive("AKBNK")
        assert result["found"] is True
        assert result["oid"] == "oid-001"
        assert result["stock_code"] == "AKBNK"
        assert result["suggestions"] == []

    def test_case_insensitive_lookup(self):
        result = lookup_member_oid_noninteractive("akbnk")
        assert result["found"] is True
        assert result["oid"] == "oid-001"

    def test_not_found_returns_found_false(self):
        result = lookup_member_oid_noninteractive("ZZZZZ")
        assert result["found"] is False
        assert result["oid"] is None

    def test_close_match_returns_suggestions(self):
        result = lookup_member_oid_noninteractive("AKBK")
        assert result["found"] is False
        assert "AKBNK" in result["suggestions"]

    def test_not_found_no_suggestions_for_garbage(self):
        result = lookup_member_oid_noninteractive("XYZQWERTY")
        assert result["found"] is False
        assert result["suggestions"] == []

    def test_stock_code_uppercased_in_response(self):
        result = lookup_member_oid_noninteractive("thyao")
        assert result["stock_code"] == "THYAO"

    def test_force_refresh_fetches_fresh_data(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(ms, "MEMBER_MAP_CACHE_FILE", cache_file)
        _preset_cache({"AKBNK": "oid-old"})

        fresh_html = build_member_rsc_html({"AKBNK": "oid-new"})
        mock_resp = MagicMock()
        mock_resp.text = fresh_html

        with patch("services.kap.scraper.member_scraper.fetch_with_retry", return_value=mock_resp):
            result = lookup_member_oid_noninteractive("AKBNK", force_refresh=True)

        assert result["oid"] == "oid-new"


# --------------------------------------------------------------------------- #
# list_disclosures
# --------------------------------------------------------------------------- #

class TestListDisclosures:
    def _make_raw_entry(self, **overrides) -> dict:
        base = {
            "disclosureIndex": 1001,
            "stockCode": "AKBNK",
            "kapTitle": "AKBANK T.A.Ş.",
            "disclosureClass": "FR",
            "disclosureType": "Mali Tablo",
            "disclosureCategory": "FR",
            "publishDate": "19.02.2026 10:00:00",
            "relatedStocks": "AKBNK",
        }
        base.update(overrides)
        return base

    def _mock_fetch(self, response_data: list):
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        return mock_resp

    def test_returns_disclosure_items(self):
        mock_resp = self._mock_fetch([self._make_raw_entry()])
        with patch("services.kap.scraper.member_scraper.fetch_with_retry", return_value=mock_resp):
            items = list_disclosures(["oid-001"])
        assert len(items) == 1
        assert items[0].disclosure_index == 1001

    def test_empty_api_response(self):
        mock_resp = self._mock_fetch([])
        with patch("services.kap.scraper.member_scraper.fetch_with_retry", return_value=mock_resp):
            items = list_disclosures(["oid-001"])
        assert items == []

    def test_stock_code_from_related_stocks(self):
        entry = self._make_raw_entry(stockCode=None, relatedStocks="THYAO")
        mock_resp = self._mock_fetch([entry])
        with patch("services.kap.scraper.member_scraper.fetch_with_retry", return_value=mock_resp):
            items = list_disclosures(["oid-002"])
        assert items[0].stock_code == "THYAO"

    def test_multiple_related_stocks_kept_as_list(self):
        entry = self._make_raw_entry(relatedStocks="AKBNK, THYAO")
        mock_resp = self._mock_fetch([entry])
        with patch("services.kap.scraper.member_scraper.fetch_with_retry", return_value=mock_resp):
            items = list_disclosures(["oid-001"])
        assert items[0].related_stocks == ["AKBNK", "THYAO"]

    def test_company_title_from_kap_title(self):
        entry = self._make_raw_entry(kapTitle="AKBANK T.A.Ş.", companyTitle=None)
        mock_resp = self._mock_fetch([entry])
        with patch("services.kap.scraper.member_scraper.fetch_with_retry", return_value=mock_resp):
            items = list_disclosures(["oid-001"])
        assert items[0].company_title == "AKBANK T.A.Ş."

    def test_detail_url_set_on_items(self):
        mock_resp = self._mock_fetch([self._make_raw_entry()])
        with patch("services.kap.scraper.member_scraper.fetch_with_retry", return_value=mock_resp):
            items = list_disclosures(["oid-001"])
        assert items[0].detail_url is not None
        assert "1001" in items[0].detail_url

    def test_default_date_range_sent_in_payload(self):
        mock_resp = self._mock_fetch([])
        with patch(
            "services.kap.scraper.member_scraper.fetch_with_retry", return_value=mock_resp
        ) as mock_fetch:
            list_disclosures(["oid-001"])
        _, call_kwargs = mock_fetch.call_args
        payload = call_kwargs.get("json", {})
        assert "fromDate" in payload
        assert "toDate" in payload

    def test_custom_date_range_forwarded(self):
        mock_resp = self._mock_fetch([])
        with patch(
            "services.kap.scraper.member_scraper.fetch_with_retry", return_value=mock_resp
        ) as mock_fetch:
            list_disclosures(["oid-001"], from_date="2025-01-01", to_date="2025-12-31")
        _, call_kwargs = mock_fetch.call_args
        payload = call_kwargs.get("json", {})
        assert payload["fromDate"] == "2025-01-01"
        assert payload["toDate"] == "2025-12-31"

    def test_category_filter_forwarded(self):
        mock_resp = self._mock_fetch([])
        with patch(
            "services.kap.scraper.member_scraper.fetch_with_retry", return_value=mock_resp
        ) as mock_fetch:
            list_disclosures(["oid-001"], category_filter="FR")
        _, call_kwargs = mock_fetch.call_args
        payload = call_kwargs.get("json", {})
        assert payload["disclosureClass"] == "FR"

    def test_all_category_filter_sends_empty_string(self):
        mock_resp = self._mock_fetch([])
        with patch(
            "services.kap.scraper.member_scraper.fetch_with_retry", return_value=mock_resp
        ) as mock_fetch:
            list_disclosures(["oid-001"], category_filter="ALL")
        _, call_kwargs = mock_fetch.call_args
        payload = call_kwargs.get("json", {})
        assert payload["disclosureClass"] == ""

    def test_skips_invalid_entries_gracefully(self):
        bad_entry = {"disclosureIndex": None}
        good_entry = self._make_raw_entry(disclosureIndex=2002)
        mock_resp = self._mock_fetch([bad_entry, good_entry])
        with patch("services.kap.scraper.member_scraper.fetch_with_retry", return_value=mock_resp):
            items = list_disclosures(["oid-001"])
        valid = [i for i in items if i.disclosure_index == 2002]
        assert len(valid) == 1
