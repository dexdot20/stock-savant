"""
Domain Utilities - Struct

Bu paket, sirket verilerinin islenmesi ve AI ciktilarinin hazirlanmasi
icin gerekli temel modelleri ve yardimci fonksiyonlari icerir.

Alt moduller:
- models.py: Veri modelleri (DataQuality ve dataclass tanimlari)
- quality.py: Data quality and confidence score calculations
- data_processors.py: Ana isleme ve birlestirme fonksiyonlari
- utils.py: Tip donusumleri ve format yardimcilari
"""

# Modelleri import et
from .models import (
    DataQuality,
    ProcessedFinancialMetrics,
    ProcessedCompanyProfile,
    ProcessedComparativeAnalysis,
    ProcessedCompanyData,
)

# Import processing functions
from .data_processors import (
    process_and_enrich_company_data,
)

# Import confidence functions
from .confidence import (
    auto_complete_company_data,
    calculate_company_confidence,
    calculate_news_confidence,
)

__all__ = [
    # Veri modelleri
    "DataQuality",
    "ProcessedFinancialMetrics",
    "ProcessedCompanyProfile",
    "ProcessedComparativeAnalysis",
    "ProcessedCompanyData",
    # Ana fonksiyonlar
    "process_and_enrich_company_data",
    # Confidence
    "auto_complete_company_data",
    "calculate_company_confidence",
    "calculate_news_confidence",
]
