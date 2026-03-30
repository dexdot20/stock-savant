"""
Data Validation Service.

This module is responsible for data validation and consistency checking.
Following the single responsibility principle, it handles only data validation operations.
"""

import math
from datetime import date, datetime
from typing import Dict, List, Optional, Any

from core import get_standard_logger
from domain.utils import safe_float, get_currency_symbol
from config import NA_VALUE


class DataValidationService:
    """Dedicated service class for data validation."""

    def __init__(self):
        """Initialize the data validation service."""
        self.logger = get_standard_logger(__name__)

        # Minimum required fields (critical identity and classification info)
        self._required_fields = {"symbol", "longName", "sector", "industry"}

        # Kritik finansal metrikler
        self._critical_financial_fields = {
            "revenue",
            "netIncome",
            "marketCap",
            "trailingPE",
            "priceToBook",
            "regularMarketPrice",
            "operatingIncome",
        }

        # Numerik alanlar
        self._numeric_fields = {
            "regularMarketPrice",
            "marketCap",
            "trailingPE",
            "priceToBook",
            "priceToSales",
            "returnOnEquity",
            "returnOnAssets",
            "debtToEquity",
            "currentRatio",
            "quickRatio",
            "revenue",
            "netIncome",
            "grossProfit",
            "operatingIncome",
            "ebitda",
            "eps",
            "freeCashFlow",
            "enterpriseValue",
            "beta",
            "dividendYield",
            "fiftyTwoWeekHigh",
            "fiftyTwoWeekLow",
            "volume",
            "previousClose",
            "altmanZScore",
            "piotroskiScore",
        }

        # Pozitif olması gereken alanlar
        self._positive_fields = {
            "regularMarketPrice",
            "marketCap",
            "volume",
            "fiftyTwoWeekHigh",
            "fiftyTwoWeekLow",
        }

        # Reasonable lower/upper bounds for numeric fields (filters abnormal values)
        self._numeric_bounds = {
            "dividendYield": (0.0, 20.0),
            "regularMarketChangePercent": (-100.0, 100.0),
            "returnOnEquity": (-3.0, 3.0),
            "returnOnAssets": (-5.0, 5.0),
            "profitMargins": (-2.0, 2.0),
            "operatingMargins": (-2.0, 2.0),
            "grossMargins": (-2.0, 2.0),
            "beta": (-10.0, 10.0),
            "debtToEquity": (0.0, 1000.0),
            "trailingPE": (0.0, 1000.0),
            "currentRatio": (0.0, None),
            "quickRatio": (0.0, None),
        }

        # Expected direction for date fields (future/past)
        self._date_validation_rules = {
            "lastDividendDate": "past_or_today",
            "lastEarningsDate": "past_or_today",
            "nextDividendDate": "future",
            "nextEarningsDate": "future",
        }

    def _is_valid_numeric(self, value: Any, allow_negative: bool = True) -> bool:
        """Check if numeric value is valid."""
        if value is None or value == NA_VALUE:
            return False

        try:
            num_val = float(value)
            if not allow_negative and num_val < 0:
                return False

            # NaN ve inf kontrolü
            if math.isnan(num_val) or math.isinf(num_val):
                return False

            return True
        except (ValueError, TypeError):
            return False

    def _is_valid_string(self, value: Any) -> bool:
        """String değerin geçerli olup olmadığını kontrol eder."""
        if value is None:
            return False

        if isinstance(value, str):
            return len(value.strip()) > 0 and value.strip() not in [
                "",
                NA_VALUE,
                "null",
                "None",
            ]

        return False

    def _apply_numeric_bounds(self, data: Dict[str, Any]) -> None:
        """Numerik alanları makul sınırlar içinde tutar."""
        for field, (min_val, max_val) in self._numeric_bounds.items():
            if field not in data:
                continue

            value = data[field]
            if value in (None, NA_VALUE):
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                self.logger.debug(
                    "%s sayısal olarak parse edilemedi, N/A atandı", field
                )
                data[field] = NA_VALUE
                continue

            if min_val is not None and numeric_value < min_val:
                self.logger.debug(
                    "%s değeri alt sınırın altında (%s), N/A yapıldı",
                    field,
                    numeric_value,
                )
                data[field] = NA_VALUE
                continue

            if max_val is not None and numeric_value > max_val:
                self.logger.debug(
                    "%s değeri üst sınırın üzerinde (%s), N/A yapıldı",
                    field,
                    numeric_value,
                )
                data[field] = NA_VALUE

    def _normalize_date_fields(self, data: Dict[str, Any]) -> None:
        """Tarih alanlarının kronolojik tutarlılığını sağlar."""
        today = date.today()

        for field, expectation in self._date_validation_rules.items():
            raw_value = data.get(field)
            if raw_value in (None, NA_VALUE, ""):
                continue

            parsed = self._parse_date(raw_value)
            if parsed is None:
                self.logger.debug(
                    "%s tarihi parse edilemedi (%s), N/A yapıldı", field, raw_value
                )
                data[field] = NA_VALUE
                continue

            if expectation == "future" and parsed <= today:
                self.logger.debug("%s tarihi geçmişte (%s), N/A yapıldı", field, parsed)
                data[field] = NA_VALUE
            elif expectation == "past_or_today" and parsed > today:
                self.logger.debug(
                    "%s tarihi gelecekte (%s), N/A yapıldı", field, parsed
                )
                data[field] = NA_VALUE

    @staticmethod
    def _parse_date(value: Any) -> Optional[date]:
        """String veya datetime değerini date nesnesine çevirir."""
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value).date()
            except ValueError:
                return None
        return None

    def validate_required_fields(self, data: Dict[str, Any]) -> List[str]:
        """
        Gerekli alanlarin varligini kontrol eder.

        Args:
            data: Kontrol edilecek veri sozlugu

        Returns:
            Eksik olan alanlarin listesi
        """
        missing_fields = []

        for field in self._required_fields:
            if not self._is_valid_string(data.get(field)):
                missing_fields.append(field)

        return missing_fields

    def validate_numeric_consistency(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Numerik değerlerin tutarlılığını kontrol eder.

        Args:
            data: Kontrol edilecek veri sözlüğü

        Returns:
            Tutarsızlık bulunan alanlar ve açıklamaları
        """
        inconsistencies = {}

        # Pozitif olması gereken alanları kontrol et
        for field in self._positive_fields:
            value = safe_float(data.get(field))
            if value is not None and value <= 0:
                inconsistencies[field] = f"Değer pozitif olmalı, mevcut: {value}"

        # Sınır kontrollerini tek merkezden yap
        for field, (min_val, max_val) in self._numeric_bounds.items():
            value = safe_float(data.get(field))
            if value is None:
                continue

            if min_val is not None and value < min_val:
                inconsistencies[field] = (
                    f"{field} alt sınırın altında ({value}), beklenen >= {min_val}"
                )
            elif max_val is not None and value > max_val:
                inconsistencies[field] = (
                    f"{field} üst sınırın üzerinde ({value}), beklenen <= {max_val}"
                )

        return inconsistencies

    def validate_financial_relationships(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Finansal metrikler arası ilişkilerin tutarlılığını kontrol eder.

        Args:
            data: Kontrol edilecek veri sözlüğü

        Returns:
            İlişki tutarsızlıkları ve açıklamaları
        """
        relationship_issues = {}

        try:
            # Market Cap ve Price ilişkisi (temel tutarlılık)
            market_cap = safe_float(data.get("marketCap"))
            price = safe_float(data.get("regularMarketPrice"))

            if market_cap and price:
                # Çok küçük şirketler için uyarı
                if market_cap < 1000000:  # 1M USD altı
                    curr_sym = get_currency_symbol(data.get("currency"))
                    relationship_issues["marketCap"] = (
                        f"Piyasa değeri çok düşük: {curr_sym}{market_cap:,.0f}"
                    )

            # ROE ve ROA ilişkisi
            roe = safe_float(data.get("returnOnEquity"))
            roa = safe_float(data.get("returnOnAssets"))

            if roe and roa:
                # ROE genellikle ROA'dan yüksek olmalı (kaldıraç etkisi)
                if abs(roe) > 0.01 and abs(roa) > 0.01 and roe < roa:
                    relationship_issues["returnOnEquity"] = (
                        f"ROE ({roe:.2%}) genellikle ROA ({roa:.2%})'dan yüksek olmalı"
                    )

            # Current Ratio ve Quick Ratio ilişkisi
            current_ratio = safe_float(data.get("currentRatio"))
            quick_ratio = safe_float(data.get("quickRatio"))

            if current_ratio and quick_ratio:
                if quick_ratio > current_ratio:
                    relationship_issues["quickRatio"] = (
                        f"Asit test oranı ({quick_ratio:.2f}) cari orandan ({current_ratio:.2f}) yüksek olamaz"
                    )

            # Revenue ve Net Income tutarlılığı
            revenue = safe_float(data.get("revenue"))
            net_income = safe_float(data.get("netIncome"))

            if revenue and net_income:
                if abs(net_income) > abs(revenue):
                    relationship_issues["netIncome"] = (
                        f"Net gelir ({net_income:,.0f}) hasılatı ({revenue:,.0f}) geçiyor"
                    )

            # 52-week range kontrolü
            high_52w = safe_float(data.get("fiftyTwoWeekHigh"))
            low_52w = safe_float(data.get("fiftyTwoWeekLow"))
            current_price = safe_float(data.get("regularMarketPrice"))

            if high_52w and low_52w and current_price:
                if low_52w > high_52w:
                    relationship_issues["fiftyTwoWeekLow"] = (
                        f"52-hafta düşük ({low_52w}) yüksekten ({high_52w}) büyük olamaz"
                    )
                elif current_price < low_52w or current_price > high_52w:
                    relationship_issues["regularMarketPrice"] = (
                        f"Güncel fiyat ({current_price}) 52-hafta aralığı dışında"
                    )

        except Exception as e:
            self.logger.debug("İlişkisel kontrol hatası: %s", e)

        return relationship_issues

    def validate_financial_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finansal veri tutarlılığını kontrol eder ve düzeltir.

        Args:
            data: Kontrol edilecek veri sözlüğü

        Returns:
            Doğrulanmış ve düzeltilmiş veri sözlüğü
        """
        validated_data = data.copy()

        # Numerik alanları normalize et
        for field in self._numeric_fields:
            if field in validated_data:
                normalized_value = safe_float(validated_data[field], NA_VALUE)
                validated_data[field] = normalized_value

        # String alanları temizle
        string_fields = ["longName", "sector", "industry", "symbol", "country", "ceo"]
        for field in string_fields:
            if field in validated_data and isinstance(validated_data[field], str):
                validated_data[field] = validated_data[field].strip()
                if not validated_data[field]:
                    validated_data[field] = NA_VALUE

        # Numerik ve tarih alanlarını makul sınırlar içinde normalize et
        self._apply_numeric_bounds(validated_data)
        self._normalize_date_fields(validated_data)

        return validated_data

    def has_minimum_required_data(self, data: Dict[str, Any]) -> bool:
        """
        Minimum gerekli veri alanlarının varlığını kontrol eder.

        Args:
            data: Kontrol edilecek veri sözlüğü

        Returns:
            Minimum veri gereksinimlerinin karşılanıp karşılanmadığı
        """
        quote_type_raw = data.get("quoteType")
        quote_type = (
            str(quote_type_raw).strip().upper()
            if quote_type_raw not in (None, NA_VALUE)
            else ""
        )
        is_index_like = quote_type in {"INDEX", "INDICE", "INDICES"}

        if is_index_like:
            index_required = ["symbol", "longName"]
            missing_index_fields = [
                field
                for field in index_required
                if not self._is_valid_string(data.get(field))
            ]
            if missing_index_fields:
                self.logger.debug(
                    "Endeks için eksik temel alanlar: %s", missing_index_fields
                )
                return False

            has_price_like = any(
                self._is_valid_numeric(data.get(field), allow_negative=False)
                for field in (
                    "regularMarketPrice",
                    "previousClose",
                    "fiftyTwoWeekHigh",
                    "fiftyTwoWeekLow",
                )
            )
            if not has_price_like:
                self.logger.debug("Endeks için fiyat benzeri alan bulunamadı")
                return False
            return True

        missing_fields = self.validate_required_fields(data)

        # Temel alanlar eksikse false
        if missing_fields:
            self.logger.debug("Eksik temel alanlar: %s", missing_fields)
            return False

        # Opsiyonel ama kritik: regularMarketPrice kontrolü
        if not self._is_valid_numeric(
            data.get("regularMarketPrice"), allow_negative=False
        ):
            self.logger.debug("Güncel fiyat eksik veya geçersiz (devam ediliyor)")

        # En az bir finansal metrik olmalı
        financial_metrics_present = 0
        for field in self._critical_financial_fields:
            if field in data and self._is_valid_numeric(data[field]):
                financial_metrics_present += 1

        if financial_metrics_present < 2:
            self.logger.debug(
                "Yetersiz finansal metrik sayısı: %d", financial_metrics_present
            )
            return False

        return True

    def get_data_quality_score(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Veri kalitesi hakkında detaylı skorlama yapar.

        Args:
            data: Analiz edilecek veri sözlüğü

        Returns:
            Veri kalitesi raporu
        """
        total_fields = len(self._numeric_fields) + len(self._required_fields)
        present_fields = 0
        valid_numeric_fields = 0

        # Alan varlığı kontrolü
        for field in self._required_fields:
            if field in data and data[field] not in [None, NA_VALUE, ""]:
                present_fields += 1

        for field in self._numeric_fields:
            if field in data and self._is_valid_numeric(data[field]):
                valid_numeric_fields += 1

        # Tutarlılık kontrolleri
        numeric_issues = self.validate_numeric_consistency(data)
        relationship_issues = self.validate_financial_relationships(data)

        # Genel kalite skoru hesaplama
        field_completeness = present_fields / len(self._required_fields)
        numeric_validity = valid_numeric_fields / len(self._numeric_fields)
        consistency_penalty = (len(numeric_issues) + len(relationship_issues)) * 0.1

        overall_score = max(
            0,
            (field_completeness * 0.4 + numeric_validity * 0.6 - consistency_penalty)
            * 100,
        )

        return {
            "overall_score": round(overall_score, 2),
            "field_completeness": round(field_completeness * 100, 2),
            "numeric_validity": round(numeric_validity * 100, 2),
            "total_fields_expected": total_fields,
            "present_fields": present_fields,
            "valid_numeric_fields": valid_numeric_fields,
            "numeric_inconsistencies": len(numeric_issues),
            "relationship_issues": len(relationship_issues),
            "has_minimum_data": self.has_minimum_required_data(data),
            "inconsistency_details": {**numeric_issues, **relationship_issues},
        }
