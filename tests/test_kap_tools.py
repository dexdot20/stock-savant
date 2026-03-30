from __future__ import annotations

import asyncio

from services.kap import KapLookupError
from services.kap.models import DisclosureDetail, DisclosureItem
from services.tools import execute_tool


def test_kap_search_tool_serializes_embedded_results(monkeypatch) -> None:
    def fake_search_disclosures(stock_codes, **kwargs):
        assert stock_codes == ["AKBNK"]
        return [
            DisclosureItem.model_validate(
                {
                    "disclosureIndex": 1001,
                    "stockCode": "AKBNK",
                    "companyTitle": "AKBANK",
                    "disclosureClass": "FR",
                    "publishDate": "19.02.2026 10:00:00",
                }
            )
        ]

    monkeypatch.setattr("services.tools.search_disclosures", fake_search_disclosures)

    result = asyncio.run(execute_tool("kap_search_disclosures", {"stock_codes": ["AKBNK"]}))

    assert isinstance(result, list)
    assert result[0]["disclosureIndex"] == 1001
    assert result[0]["stockCode"] == "AKBNK"


def test_kap_search_tool_surfaces_lookup_validation(monkeypatch) -> None:
    def fake_search_disclosures(stock_codes, **kwargs):
        raise KapLookupError({"message": "Some stock codes were not found.", "errors": []})

    monkeypatch.setattr("services.tools.search_disclosures", fake_search_disclosures)

    result = asyncio.run(execute_tool("kap_search_disclosures", {"stock_codes": ["BAD"]}))

    assert result["error_code"] == "validation_error"
    assert result["tool"] == "kap_search_disclosures"


def test_kap_detail_tool_returns_serialized_detail(monkeypatch) -> None:
    monkeypatch.setattr(
        "services.tools.get_disclosure_detail",
        lambda disclosure_index: DisclosureDetail.model_validate(
            {
                "disclosureIndex": disclosure_index,
                "stockCode": "AKBNK",
                "companyTitle": "AKBANK",
                "disclosureClass": "FR",
                "publishDate": "19.02.2026 10:00:00",
            }
        ),
    )

    result = asyncio.run(execute_tool("kap_get_disclosure_detail", {"disclosure_index": 9999}))

    assert result["disclosureIndex"] == 9999
    assert result["stockCode"] == "AKBNK"


def test_kap_batch_tool_returns_results_and_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        "services.tools.batch_get_disclosure_details",
        lambda disclosure_indexes, max_workers: (
            [
                DisclosureDetail.model_validate(
                    {
                        "disclosureIndex": disclosure_indexes[0],
                        "stockCode": "AKBNK",
                        "companyTitle": "AKBANK",
                        "disclosureClass": "FR",
                        "publishDate": "19.02.2026 10:00:00",
                    }
                )
                ,
                None,
            ],
            {disclosure_indexes[-1]: "boom"},
        ),
    )

    result = asyncio.run(
        execute_tool(
            "kap_batch_disclosure_details",
            {"disclosure_indexes": [1001, 1002], "max_workers": 3},
        )
    )

    assert result["results"][0]["disclosureIndex"] == 1001
    assert result["results"][1] is None
    assert result["errors"]["1002"] == "boom"