"""Integration-style tests for the non-finance API routers."""

from __future__ import annotations

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import alerts, favorites, history, portfolio, profile

_app = FastAPI()
_app.include_router(favorites.router)
_app.include_router(portfolio.router)
_app.include_router(alerts.router)
_app.include_router(profile.router)
_app.include_router(history.router)

client = TestClient(_app)


class TestFavoritesRouter:
    def test_lists_favorites(self):
        with patch(
            "api.routers.favorites.load_favorites",
            return_value=["THYAO.IS", "aapl", "thyAo.is", ""],
        ):
            resp = client.get("/favorites/")

        assert resp.status_code == 200
        data = resp.json()
        assert data["favorites"] == ["AAPL", "THYAO.IS"]
        assert data["count"] == 2

    def test_adds_favorite(self):
        with patch("api.routers.favorites.add_favorite", return_value=True), patch(
            "api.routers.favorites.load_favorites",
            return_value=["AKBNK"],
        ):
            resp = client.post("/favorites/akbnk")

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["symbol"] == "AKBNK"
        assert data["favorites"] == ["AKBNK"]

    def test_remove_missing_favorite_returns_404(self):
        with patch("api.routers.favorites.remove_favorite", return_value=False):
            resp = client.delete("/favorites/akbnk")

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Favorite not found."


class TestPortfolioRouter:
    def test_lists_positions(self):
        with patch(
            "api.routers.portfolio.load_portfolio",
            return_value=[
                {"symbol": "THYAO.IS", "quantity": 5, "average_cost": 280.5},
                {"symbol": "akbnk", "quantity": 10, "average_cost": 42.0},
            ],
        ):
            resp = client.get("/portfolio/positions")

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert data["positions"][0]["symbol"] == "AKBNK"
        assert data["positions"][1]["symbol"] == "THYAO.IS"

    def test_saves_position(self):
        with patch("api.routers.portfolio.get_config", return_value={}), patch(
            "api.routers.portfolio.add_position_with_feedback",
            return_value=(True, "Position saved."),
        ), patch(
            "api.routers.portfolio.load_portfolio",
            return_value=[{"symbol": "AKBNK", "quantity": 10.0, "average_cost": 42.0}],
        ):
            resp = client.post(
                "/portfolio/positions",
                json={"symbol": "AKBNK", "quantity": 10, "average_cost": 42},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["position"]["symbol"] == "AKBNK"

    def test_guardrail_failure_returns_422(self):
        with patch("api.routers.portfolio.get_config", return_value={}), patch(
            "api.routers.portfolio.add_position_with_feedback",
            return_value=(False, "Guardrail hit."),
        ):
            resp = client.post(
                "/portfolio/positions",
                json={"symbol": "AKBNK", "quantity": 10, "average_cost": 42},
            )

        assert resp.status_code == 422
        assert resp.json()["detail"] == "Guardrail hit."

    def test_returns_risk_snapshot(self):
        snapshot = {
            "positions": [
                {
                    "symbol": "AKBNK",
                    "quantity": 10.0,
                    "average_cost": 42.0,
                    "current_price": 44.0,
                    "cost_value": 420.0,
                    "market_value": 440.0,
                    "pnl": 20.0,
                    "pnl_pct": 4.76,
                    "price_unavailable": False,
                    "weight_pct": 100.0,
                    "sector": "Financial Services",
                    "industry": "Banks",
                    "confidence_pct": 80.0,
                    "confidence_level": "high",
                    "data_quality": "high",
                }
            ],
            "summary": {
                "position_count": 1,
                "total_market_value": 440.0,
                "diversification_score": 0.0,
                "average_confidence": 80.0,
                "estimated_portfolio_volatility": 12.5,
                "risk_score": 35.0,
            },
            "sector_exposure": [
                {"sector": "Financial Services", "weight_pct": 100.0, "market_value": 440.0}
            ],
            "breaches": [{"message": "Single-name concentration is elevated.", "severity": "high"}],
            "correlation_alerts": [],
        }

        with patch("api.routers.portfolio.get_config", return_value={}), patch(
            "api.routers.portfolio.portfolio_risk_snapshot",
            return_value=snapshot,
        ):
            resp = client.get("/portfolio/risk")

        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["position_count"] == 1
        assert data["positions"][0]["symbol"] == "AKBNK"


class TestAlertsRouter:
    def test_lists_alerts(self):
        with patch(
            "api.routers.alerts.list_alerts",
            return_value={
                "count": 1,
                "alerts": [
                    {
                        "id": "alert-1",
                        "type": "price",
                        "symbol": "AKBNK",
                        "target_price": 50.0,
                        "direction": "above",
                        "note": "watch",
                        "triggered": False,
                        "status": "active",
                    }
                ],
            },
        ):
            resp = client.get("/alerts/?symbol=akbnk")

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["alerts"][0]["symbol"] == "AKBNK"

    def test_creates_price_alert(self):
        with patch(
            "api.routers.alerts.create_price_alert",
            return_value={
                "status": "set",
                "alert": {
                    "id": "alert-1",
                    "type": "price",
                    "symbol": "AKBNK",
                    "target_price": 50.0,
                    "direction": "above",
                    "note": "watch",
                    "triggered": False,
                    "status": "active",
                },
            },
        ):
            resp = client.post(
                "/alerts/price",
                json={
                    "symbol": "AKBNK",
                    "target_price": 50,
                    "direction": "above",
                    "note": "watch",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["message"] == "Price alert created."
        assert data["alert"]["symbol"] == "AKBNK"

    def test_returns_alert_center(self):
        payload = {
            "triggered_events": [{"message": "Triggered.", "severity": "medium"}],
            "price_events": [{"message": "Triggered.", "severity": "medium"}],
            "risk_events": [],
            "kap_events": [],
            "saved_alerts": {"count": 1, "alerts": []},
        }

        with patch("api.routers.alerts.get_config", return_value={}), patch(
            "api.routers.alerts.evaluate_alert_center",
            return_value=payload,
        ):
            resp = client.get("/alerts/center")

        assert resp.status_code == 200
        assert resp.json()["saved_alerts"]["count"] == 1


class TestProfileRouter:
    def test_gets_profile(self):
        profile_payload = {
            "profile_name": "Default",
            "risk_tolerance": "medium",
            "investment_horizon": "long-term",
            "market_focus": "BIST",
            "preferred_sectors": ["Banks"],
            "avoided_sectors": [],
            "max_single_position_pct": 25.0,
            "alert_sensitivity": "medium",
            "active_playbook": "balanced",
        }

        with patch(
            "api.routers.profile.load_investor_profile",
            return_value=profile_payload,
        ), patch(
            "api.routers.profile.get_playbook_summary",
            return_value="Balanced summary",
        ):
            resp = client.get("/profile/")

        assert resp.status_code == 200
        data = resp.json()
        assert data["profile_name"] == "Default"
        assert data["playbook_summary"] == "Balanced summary"

    def test_updates_profile(self):
        current_profile = {
            "profile_name": "Default",
            "risk_tolerance": "medium",
            "investment_horizon": "long-term",
            "market_focus": "BIST",
            "preferred_sectors": [],
            "avoided_sectors": [],
            "max_single_position_pct": 25.0,
            "alert_sensitivity": "medium",
            "active_playbook": "balanced",
        }
        updated_profile = {**current_profile, "risk_tolerance": "high"}

        with patch(
            "api.routers.profile.load_investor_profile",
            return_value=current_profile,
        ), patch(
            "api.routers.profile.save_investor_profile",
            return_value=updated_profile,
        ) as save_mock, patch(
            "api.routers.profile.get_playbook_choices",
            return_value=["balanced", "growth"],
        ), patch(
            "api.routers.profile.get_playbook_summary",
            return_value="Balanced summary",
        ):
            resp = client.put("/profile/", json={"risk_tolerance": "high"})

        assert resp.status_code == 200
        assert save_mock.call_args.args[0]["risk_tolerance"] == "high"
        assert resp.json()["risk_tolerance"] == "high"

    def test_rejects_unknown_playbook(self):
        with patch(
            "api.routers.profile.load_investor_profile",
            return_value={"active_playbook": "balanced"},
        ), patch(
            "api.routers.profile.get_playbook_choices",
            return_value=["balanced"],
        ):
            resp = client.put("/profile/", json={"active_playbook": "unknown"})

        assert resp.status_code == 422
        assert resp.json()["detail"] == "Unknown playbook."


class TestHistoryRouter:
    def test_lists_history_entries(self):
        entries = [
            {
                "id": 1,
                "symbol": "AKBNK",
                "type": "stock",
                "timestamp": "2026-03-31T10:00:00",
                "file_path": "/tmp/1_AKBNK_20260331.json",
            }
        ]

        with patch("api.routers.history.list_history", return_value=entries):
            resp = client.get("/history/?symbol=akbnk&limit=5")

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["entries"][0]["symbol"] == "AKBNK"

    def test_returns_history_entry(self):
        entry = {
            "id": 1,
            "symbol": "AKBNK",
            "type": "stock",
            "data": {"final_analysis_output": "Sample output"},
            "timestamp": "2026-03-31T10:00:00",
        }

        with patch("api.routers.history.show_history_entry", return_value=entry):
            resp = client.get("/history/1")

        assert resp.status_code == 200
        assert resp.json()["data"]["final_analysis_output"] == "Sample output"

    def test_returns_404_for_missing_history_entry(self):
        with patch("api.routers.history.show_history_entry", return_value=None):
            resp = client.get("/history/99")

        assert resp.status_code == 404
        assert resp.json()["detail"] == "History entry not found."