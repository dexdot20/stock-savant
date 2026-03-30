"""Composed YFinance adapter mixin."""

from __future__ import annotations

from .analyst import YFinanceAnalystMixin
from .dividends import YFinanceDividendMixin
from .earnings import YFinanceEarningsMixin
from .fetch import YFinanceFetchMixin
from .helpers import YFinanceHelperMixin
from .statements import YFinanceStatementsMixin
from .data_accessors import (
    YFinanceNewsMixin,
    YFinanceSustainabilityMixin,
    YFinanceFundsMixin,
)
from .discovery import (
    YFinanceMarketMixin,
    YFinanceSearchMixin,
    YFinanceSectorIndustryMixin,
    YFinanceOptionsMixin,
)
from .ownership import YFinanceHoldersMixin, YFinanceInsiderMixin


class YFinanceAdapterMixin(
    YFinanceHelperMixin,
    YFinanceStatementsMixin,
    YFinanceDividendMixin,
    YFinanceEarningsMixin,
    YFinanceAnalystMixin,
    YFinanceNewsMixin,
    YFinanceSustainabilityMixin,
    YFinanceInsiderMixin,
    YFinanceHoldersMixin,
    YFinanceFetchMixin,
    YFinanceMarketMixin,
    YFinanceSearchMixin,
    YFinanceSectorIndustryMixin,
    YFinanceOptionsMixin,
    YFinanceFundsMixin,
):
    """Aggregates mixins that interact with the YFinance data source."""

    # Intentional: methods are provided by composed mixins.
    pass
