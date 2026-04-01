import unittest
from unittest.mock import patch

import pandas as pd

from services.finance.market_data import MarketDataService
from services.finance.market_data.yfinance_adapter.helpers import YFinanceHelperMixin
from services.finance.market_data.yfinance_adapter.fetch import YFinanceFetchMixin
from services.tools import YFinanceSectorTool


class YFinanceScreenNormalizationTests(unittest.TestCase):
    def test_screen_tickers_normalizes_dict_payload(self) -> None:
        service = MarketDataService({})

        with patch(
            "services.finance.market_data.yfinance_adapter.discovery.yf.screen",
            return_value={
                "quotes": [
                    {"symbol": "EREGL.IS", "marketCap": "123", "regularMarketPrice": "28.84"},
                    {"symbol": "TUPRS.IS", "marketCap": 456, "regularMarketPrice": 250.1},
                ]
            },
        ):
            rows = service.screen_tickers(
                query_type="EQUITY",
                filters={
                    "conditions": [
                        ("eq", "region", "tr"),
                        ("is-in", "exchange", "IST"),
                    ]
                },
                limit=1,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "EREGL.IS")
        self.assertEqual(rows[0]["marketCap"], 123.0)
        self.assertEqual(rows[0]["regularMarketPrice"], 28.84)


class YFinanceHistorySummaryTests(unittest.TestCase):
    def test_summarize_history_adds_relative_low_metrics(self) -> None:
        history = pd.DataFrame(
            {"Close": [10.0, 14.0, 13.0, 11.0]},
            index=pd.date_range("2026-01-01", periods=4, freq="D"),
        )

        summary = YFinanceHelperMixin._summarize_history(history)

        self.assertEqual(summary["periodHigh"], 14.0)
        self.assertEqual(summary["periodLow"], 10.0)
        self.assertEqual(summary["firstClose"], 10.0)
        self.assertEqual(summary["lastClose"], 11.0)
        self.assertAlmostEqual(summary["periodReturnPct"], 10.0)
        self.assertAlmostEqual(summary["rangePositionPct"], 25.0)
        self.assertAlmostEqual(summary["distanceFromLowPct"], 10.0)
        self.assertAlmostEqual(summary["distanceFromHighPct"], (14.0 - 11.0) / 14.0 * 100)
        self.assertTrue(summary["isNearPeriodLow"])

    def test_summarize_history_adds_technical_indicators(self) -> None:
        closes = [100.0 + index * 0.8 for index in range(40)]
        highs = [value + 1.5 for value in closes]
        lows = [value - 1.5 for value in closes]
        history = pd.DataFrame(
            {"Close": closes, "High": highs, "Low": lows},
            index=pd.date_range("2026-01-01", periods=40, freq="D"),
        )

        summary = YFinanceHelperMixin._summarize_history(history)

        technicals = summary["technicalIndicators"]
        self.assertIn("rsi", technicals)
        self.assertIn("macd", technicals)
        self.assertIn("macdSignal", technicals)
        self.assertIn("supportLevel", technicals)
        self.assertIn("resistanceLevel", technicals)
        self.assertGreaterEqual(
            technicals["resistanceLevel"], technicals["supportLevel"]
        )


class FakeFinanceService:
    def screen_equities_by_market(self, *, region, exchange, sector=None, limit=25):
        return {
            "region": region,
            "exchange": exchange,
            "sector": sector,
            "limit": limit,
            "count": 1,
            "quotes": [{"symbol": "EREGL.IS", "longName": "Eregli Demir ve Celik Fabrikalari T.A.S."}],
        }

    def get_index_data(self, symbol):
        return {"symbol": symbol, "regularMarketPrice": 13201.72}

    def get_sector_info(self, sector_key):
        return {"key": sector_key}

    def get_industry_info(self, industry_key):
        return {"key": industry_key}


class YFinanceSectorToolFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_exchange_level_request_uses_market_screen(self) -> None:
        tool = YFinanceSectorTool()
        tool.finance_service = FakeFinanceService()

        result = await tool.execute(exchange="BIST", sector="all", max_tickers=15)

        self.assertEqual(result["mode"], "market_screen")
        self.assertEqual(result["market_scope"]["exchange"], "IST")
        self.assertEqual(result["market_scope"]["region"], "tr")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["quotes"][0]["symbol"], "EREGL.IS")
        self.assertEqual(result["index_context"]["symbol"], "XU100.IS")


class _BrokenFastInfo:
    def __dir__(self):
        return ["currency", "last_price", "previous_close"]

    def __getattr__(self, name):
        if name == "currency":
            raise KeyError("currency")
        if name == "last_price":
            return 100.0
        if name == "previous_close":
            return 99.0
        raise AttributeError(name)


class _BrokenTicker:
    ticker = "FAKE.IS"

    @property
    def fast_info(self):
        return _BrokenFastInfo()


class YFinanceFastInfoRegressionTests(unittest.TestCase):
    def test_safe_get_fast_info_skips_currency_key_error(self) -> None:
        class _Harness(YFinanceFetchMixin):
            def __init__(self):
                self.logger = type(
                    "L",
                    (),
                    {
                        "debug": lambda *_, **__: None,
                        "warning": lambda *_, **__: None,
                        "error": lambda *_, **__: None,
                    },
                )()

        harness = _Harness()
        data = harness._safe_get_fast_info(_BrokenTicker())
        self.assertEqual(data["last_price"], 100.0)
        self.assertEqual(data["previous_close"], 99.0)
        self.assertNotIn("currency", data)


if __name__ == "__main__":
    unittest.main()