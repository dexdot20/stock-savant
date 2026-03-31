"""
api/routers/kap.py için entegrasyon testleri.
TestClient, IP kısıtlaması olmadan izole bir test uygulamasına bağlanır.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.kap import router
from services.kap import KapLookupError
from services.kap.models import DisclosureDetail, DisclosureItem

# İzole test uygulaması — ana API'nin IP kısıtlaması olmadan
_app = FastAPI()
_app.include_router(router)

client = TestClient(_app)


# --------------------------------------------------------------------------- #
# Yardımcı
# --------------------------------------------------------------------------- #

def _sample_item() -> DisclosureItem:
    return DisclosureItem.model_validate({
        "disclosureIndex": 9999,
        "stockCode": "AKBNK",
        "companyTitle": "AKBANK T.A.Ş.",
        "disclosureClass": "FR",
        "publishDate": "19.02.2026 10:00:00",
        "relatedStocks": ["AKBNK"],
    })


def _sample_detail() -> DisclosureDetail:
    return DisclosureDetail.model_validate({
        "disclosureIndex": 9999,
        "mkkMemberOid": "oid-001",
        "stockCode": "AKBNK",
        "companyTitle": "AKBANK T.A.Ş.",
        "disclosureClass": "FR",
        "publishDate": "19.02.2026 10:00:00",
        "isLate": False,
        "isChanged": False,
        "isBlocked": False,
    })


# --------------------------------------------------------------------------- #
# POST /kap/disclosures/list
# --------------------------------------------------------------------------- #

class TestDisclosureList:
    TARGET = "/kap/disclosures/list"

    def test_returns_list_of_items(self):
        with patch("api.routers.kap.search_disclosures", return_value=[_sample_item()]):
            resp = client.post(self.TARGET, json={"stock_codes": ["AKBNK"]})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0].get("disclosureIndex") == 9999

    def test_returns_empty_list(self):
        with patch("api.routers.kap.search_disclosures", return_value=[]):
            resp = client.post(self.TARGET, json={"stock_codes": ["AKBNK"]})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_missing_stock_codes_returns_422(self):
        resp = client.post(self.TARGET, json={})
        assert resp.status_code == 422

    def test_kap_lookup_error_returns_422_with_suggestions(self):
        error_details = {
            "message": "Bazı hisse kodları bulunamadı.",
            "errors": [{"stock_code": "AKBK", "suggestions": ["AKBNK"]}],
        }
        exc = KapLookupError(error_details)
        with patch("api.routers.kap.search_disclosures", side_effect=exc):
            resp = client.post(self.TARGET, json={"stock_codes": ["AKBK"]})
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail["errors"][0]["stock_code"] == "AKBK"
        assert "AKBNK" in detail["errors"][0]["suggestions"]

    def test_optional_filters_forwarded(self):
        with patch("api.routers.kap.search_disclosures", return_value=[]) as mock_svc:
            resp = client.post(
                self.TARGET,
                json={
                    "stock_codes": ["AKBNK"],
                    "category": "FR",
                    "from_date": "2025-01-01",
                    "to_date": "2025-12-31",
                },
            )
        assert resp.status_code == 200
        _, call_kwargs = mock_svc.call_args
        assert call_kwargs.get("category") == "FR"
        assert call_kwargs.get("from_date") == "2025-01-01"
        assert call_kwargs.get("to_date") == "2025-12-31"

    def test_days_param_forwarded_to_service(self):
        with patch("api.routers.kap.search_disclosures", return_value=[]) as mock_svc:
            resp = client.post(
                self.TARGET,
                json={"stock_codes": ["AKBNK"], "days": 7, "from_date": "2020-01-01"},
            )
        assert resp.status_code == 200
        _, call_kwargs = mock_svc.call_args
        assert call_kwargs.get("days") == 7

    def test_http_status_error_returns_502(self):
        import httpx
        mock_response = MagicMock()
        mock_response.status_code = 503
        exc = httpx.HTTPStatusError("err", request=MagicMock(), response=mock_response)
        with patch("api.routers.kap.search_disclosures", side_effect=exc):
            resp = client.post(self.TARGET, json={"stock_codes": ["AKBNK"]})
        assert resp.status_code == 502

    def test_request_error_returns_503(self):
        import httpx
        with patch("api.routers.kap.search_disclosures", side_effect=httpx.ConnectError("hata")):
            resp = client.post(self.TARGET, json={"stock_codes": ["AKBNK"]})
        assert resp.status_code == 503


# --------------------------------------------------------------------------- #
# GET /kap/disclosures/{index}/detail
# --------------------------------------------------------------------------- #

class TestDisclosureDetail:
    def _url(self, index: int) -> str:
        return f"/kap/disclosures/{index}/detail"

    def test_returns_detail(self):
        with patch("api.routers.kap.get_disclosure_detail", return_value=_sample_detail()):
            resp = client.get(self._url(9999))
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("disclosureIndex") == 9999
        assert data.get("stockCode") == "AKBNK"

    def test_none_result_returns_422(self):
        with patch("api.routers.kap.get_disclosure_detail", return_value=None):
            resp = client.get(self._url(9999))
        assert resp.status_code == 422

    def test_http_status_error_returns_502(self):
        import httpx
        mock_response = MagicMock()
        mock_response.status_code = 404
        exc = httpx.HTTPStatusError("err", request=MagicMock(), response=mock_response)
        with patch("api.routers.kap.get_disclosure_detail", side_effect=exc):
            resp = client.get(self._url(9999))
        assert resp.status_code == 502

    def test_request_error_returns_503(self):
        import httpx
        with patch("api.routers.kap.get_disclosure_detail", side_effect=httpx.ConnectError("hata")):
            resp = client.get(self._url(9999))
        assert resp.status_code == 503

    def test_non_integer_index_returns_422(self):
        resp = client.get("/kap/disclosures/abc/detail")
        assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# POST /kap/disclosures/batch-details
# --------------------------------------------------------------------------- #

class TestBatchDetails:
    TARGET = "/kap/disclosures/batch-details"

    def test_returns_results_for_all_indexes(self):
        detail = _sample_detail()
        with patch(
            "api.routers.kap.batch_get_disclosure_details", return_value=([detail, detail], {})
        ):
            resp = client.post(self.TARGET, json={"disclosure_indexes": [9999, 8888]})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        assert data["errors"] == {}

    def test_errors_tracked_per_index(self):
        with patch(
            "api.routers.kap.batch_get_disclosure_details",
            return_value=([None], {1111: "simüle hata"}),
        ):
            resp = client.post(self.TARGET, json={"disclosure_indexes": [1111]})
        assert resp.status_code == 200
        data = resp.json()
        assert "1111" in data["errors"]

    def test_max_workers_zero_returns_422(self):
        resp = client.post(self.TARGET, json={"disclosure_indexes": [1], "max_workers": 0})
        assert resp.status_code == 422

    def test_empty_indexes_returns_empty(self):
        with patch(
            "api.routers.kap.batch_get_disclosure_details", return_value=([], {})
        ):
            resp = client.post(self.TARGET, json={"disclosure_indexes": []})
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []
        assert data["errors"] == {}


# --------------------------------------------------------------------------- #
# GET /kap/proxies/status
# --------------------------------------------------------------------------- #

class TestProxyStatus:
    TARGET = "/kap/proxies/status"

    def test_returns_proxy_status_structure(self):
        mock_mgr = MagicMock()
        mock_mgr.status.return_value = [{"url": "http://p1:80", "active": True, "error_count": 0}]
        with patch("api.routers.kap.get_proxy_manager", return_value=mock_mgr):
            resp = client.get(self.TARGET)
        assert resp.status_code == 200
        data = resp.json()
        assert "proxies" in data
        assert "total" in data
        assert "active" in data

    def test_active_less_or_equal_total(self):
        mock_mgr = MagicMock()
        mock_mgr.status.return_value = [
            {"url": "http://p1:80", "active": True, "error_count": 0},
            {"url": "http://p2:80", "active": False, "error_count": 3},
        ]
        with patch("api.routers.kap.get_proxy_manager", return_value=mock_mgr):
            data = client.get(self.TARGET).json()
        assert data["active"] <= data["total"]
        assert data["total"] == 2
        assert data["active"] == 1
