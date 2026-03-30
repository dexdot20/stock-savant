"""
services.kap.models birim testleri.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from services.kap.models import (
    DisclosureDetail,
    DisclosureItem,
    SignatureInfo,
    _parse_turkish_datetime,
)


# --------------------------------------------------------------------------- #
# _parse_turkish_datetime
# --------------------------------------------------------------------------- #

class TestParseTurkishDatetime:
    def test_parses_turkish_format(self):
        result = _parse_turkish_datetime("19.02.2026 16:52:20")
        assert result == datetime(2026, 2, 19, 16, 52, 20)

    def test_parses_iso_format_fallback(self):
        result = _parse_turkish_datetime("2026-02-19T16:52:20")
        assert result == datetime(2026, 2, 19, 16, 52, 20)

    def test_returns_none_for_none(self):
        assert _parse_turkish_datetime(None) is None

    def test_returns_none_for_empty_string(self):
        assert _parse_turkish_datetime("") is None

    def test_returns_none_for_invalid_string(self):
        assert _parse_turkish_datetime("not-a-date") is None

    def test_returns_none_for_non_string(self):
        assert _parse_turkish_datetime(12345) is None

    def test_returns_datetime_unchanged(self):
        dt = datetime(2026, 1, 1, 0, 0, 0)
        assert _parse_turkish_datetime(dt) == dt

    @pytest.mark.parametrize("value,expected", [
        ("01.01.2026 00:00:00", datetime(2026, 1, 1, 0, 0, 0)),
        ("31.12.2025 23:59:59", datetime(2025, 12, 31, 23, 59, 59)),
        ("15.07.2024 12:00:00", datetime(2024, 7, 15, 12, 0, 0)),
    ])
    def test_parametrize_turkish_dates(self, value, expected):
        assert _parse_turkish_datetime(value) == expected


# --------------------------------------------------------------------------- #
# DisclosureItem
# --------------------------------------------------------------------------- #

class TestDisclosureItem:
    def test_minimal_valid_item(self):
        item = DisclosureItem.model_validate({"disclosureIndex": 1})
        assert item.disclosure_index == 1

    def test_alias_fields(self):
        item = DisclosureItem.model_validate({
            "disclosureIndex": 2002,
            "stockCode": "THYAO",
            "companyTitle": "Turk Hava Yollari",
        })
        assert item.disclosure_index == 2002
        assert item.stock_code == "THYAO"
        assert item.company_title == "Turk Hava Yollari"

    def test_publish_date_parsed(self):
        item = DisclosureItem.model_validate({
            "disclosureIndex": 1,
            "publishDate": "19.02.2026 10:00:00",
        })
        assert item.publish_date == datetime(2026, 2, 19, 10, 0, 0)

    def test_optional_fields_default_none(self):
        item = DisclosureItem.model_validate({"disclosureIndex": 1})
        assert item.stock_code is None
        assert item.company_title is None
        assert item.related_stocks is None

    def test_related_stocks_as_list(self):
        item = DisclosureItem.model_validate({
            "disclosureIndex": 1,
            "relatedStocks": ["AKBNK", "THYAO"],
        })
        assert item.related_stocks == ["AKBNK", "THYAO"]

    def test_populate_by_name(self):
        item = DisclosureItem(disclosure_index=99, stock_code="GARAN")
        assert item.disclosure_index == 99
        assert item.stock_code == "GARAN"

    def test_detail_url_field(self):
        item = DisclosureItem.model_validate({"disclosureIndex": 1})
        item.detail_url = "https://kap.org.tr/tr/Bildirim/1"
        assert item.detail_url is not None


# --------------------------------------------------------------------------- #
# SignatureInfo
# --------------------------------------------------------------------------- #

class TestSignatureInfo:
    def test_all_optional(self):
        sig = SignatureInfo.model_validate({})
        assert sig.user_name is None
        assert sig.company_name is None

    def test_full_alias_input(self):
        sig = SignatureInfo.model_validate({
            "userName": "Ahmet Yilmaz",
            "title": "CEO",
            "companyName": "AKBANK",
            "signDate": "19.02.2026",
        })
        assert sig.user_name == "Ahmet Yilmaz"
        assert sig.company_name == "AKBANK"
        assert sig.title == "CEO"

    def test_partial_fields(self):
        sig = SignatureInfo.model_validate({"userName": "Fatma"})
        assert sig.user_name == "Fatma"
        assert sig.title is None


# --------------------------------------------------------------------------- #
# DisclosureDetail
# --------------------------------------------------------------------------- #

class TestDisclosureDetail:
    def test_all_optional(self):
        detail = DisclosureDetail.model_validate({})
        assert detail.disclosure_index is None
        assert detail.stock_code is None

    def test_alias_construction(self):
        detail = DisclosureDetail.model_validate({
            "disclosureIndex": 9999,
            "mkkMemberOid": "oid-001",
            "stockCode": "AKBNK",
        })
        assert detail.disclosure_index == 9999
        assert detail.mkk_member_oid == "oid-001"
        assert detail.stock_code == "AKBNK"

    def test_publish_date_parsed(self):
        detail = DisclosureDetail.model_validate({"publishDate": "01.03.2026 09:15:00"})
        assert detail.publish_date == datetime(2026, 3, 1, 9, 15, 0)

    def test_signatures_field(self):
        detail = DisclosureDetail.model_validate({
            "signatures": [{"userName": "A", "title": "CEO"}],
        })
        assert detail.signatures is not None
        assert len(detail.signatures) == 1
        assert detail.signatures[0].user_name == "A"

    def test_disclosure_body_fields(self):
        detail = DisclosureDetail.model_validate({
            "disclosureBody": "<p>İçerik</p>",
            "disclosureBodyText": "İçerik",
        })
        assert detail.disclosure_body == "<p>İçerik</p>"
        assert detail.disclosure_body_text == "İçerik"

    def test_opinion_type_accepts_dict(self):
        detail = DisclosureDetail.model_validate({
            "opinionType": {"id": 1, "label": "Olumlu"},
            "auditType": {"id": 2, "label": "Bağımsız"},
        })
        assert detail.opinion_type == {"id": 1, "label": "Olumlu"}
        assert detail.audit_type == {"id": 2, "label": "Bağımsız"}

    def test_populate_by_name_direct(self):
        detail = DisclosureDetail(
            disclosure_index=1,
            stock_code="THYAO",
            disclosure_body_text="Metin",
        )
        assert detail.stock_code == "THYAO"
        assert detail.disclosure_body_text == "Metin"
