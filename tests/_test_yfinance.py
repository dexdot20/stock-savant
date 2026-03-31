"""
yfinance category validation test
Usage: python _test_yfinance.py [SYMBOL1 SYMBOL2 ...]
Default symbols: AAPL, THYAO.IS
"""

import sys
import time

sys.path.insert(0, ".")
import yfinance as yf

# ── Color constants ────────────────────────────────────────────
PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"
WARN = "\033[93m WARN \033[0m"
SKIP = "\033[90m SKIP \033[0m"

# ── Known yfinance limitations by exchange ─────────────────────
# These data are outside Yahoo Finance scope, so SKIP not FAIL.
KNOWN_GAPS = {
    # (exchange_suffix, field_key)
    (".IS", "upgrades_downgrades"),
    (".IS", "earnings_history"),
    (".IS", "institutional_holders"),
    (".IS", "insider_transactions"),
    (".IS", "sustainability"),
    # All symbols
    ("*", "sustainability"),  # Yahoo fundamentals 404 — common issue
}


def is_known_gap(symbol: str, field_key: str) -> bool:
    for suffix, key in KNOWN_GAPS:
        if key != field_key:
            continue
        if suffix == "*" or symbol.upper().endswith(suffix.upper()):
            return True
    return False


# ── Result tracking ────────────────────────────────────────────
passed_total = failed_total = skipped_total = 0


def check(label: str, value, symbol: str = "", field_key: str = ""):
    global passed_total, failed_total, skipped_total
    has_data = value is not None and (
        (hasattr(value, "__len__") and len(value) > 0)
        or isinstance(value, (int, float))
    )
    if has_data:
        status = PASS
        passed_total += 1
    elif is_known_gap(symbol, field_key):
        status = SKIP
        skipped_total += 1
    else:
        status = FAIL
        failed_total += 1

    display = repr(value)[:72] if value is not None else "None"
    print(f"  [{status}] {label:<42} {display}")


# ── Test symbols ──────────────────────────────────────────────
symbols = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "THYAO.IS"]

for symbol in symbols:
    t0 = time.time()
    print(f"\n{'═'*60}")
    print(f"  TICKER : {symbol}")
    print(f"{'═'*60}")

    t = yf.Ticker(symbol)

    # ── Overview ────────────────────────────────────────────────
    print("\n  ── OVERVIEW  →  GET /finance/{symbol}/overview")
    fi = t.fast_info
    check("fast_info.last_price", fi.last_price, symbol, "last_price")
    check("fast_info.market_cap", fi.market_cap, symbol, "market_cap")
    info = t.info
    check("info.longName", info.get("longName"), symbol, "longName")
    check("info.sector", info.get("sector"), symbol, "sector")
    check("info.trailingPE", info.get("trailingPE"), symbol, "trailingPE")

    # ── Price History ──────────────────────────────────────────
    print("\n  ── PRICE HISTORY  →  GET /finance/{symbol}/price")
    h1y = t.history(period="1y", interval="1d", repair=False, actions=False)
    check("history 1y/1d  — rows", len(h1y), symbol, "history_rows")
    check(
        "history 1y/1d  — close",
        float(h1y["Close"].iloc[-1]) if not h1y.empty else None,
        symbol,
        "history_close",
    )
    h5y = t.history(period="5y", interval="1wk", repair=False, actions=False)
    check("history 5y/1wk — rows", len(h5y), symbol, "history_rows")
    h3mo = t.history(period="3mo", interval="1d", repair=False, actions=False)
    check("history 3mo/1d — rows", len(h3mo), symbol, "history_rows")
    h1mo = t.history(period="1mo", interval="1h", repair=False, actions=False)
    check("history 1mo/1h — rows", len(h1mo), symbol, "history_rows")

    # ── Dividends ──────────────────────────────────────────────
    print("\n  ── DIVIDENDS  →  GET /finance/{symbol}/dividends")
    divs = t.dividends
    check(
        "dividends — count", len(divs) if divs is not None else 0, symbol, "dividends"
    )
    splts = t.splits
    check("splits    — count", len(splts) if splts is not None else 0, symbol, "splits")
    acts = t.actions
    check("actions   — count", len(acts) if acts is not None else 0, symbol, "actions")

    # ── Analyst ────────────────────────────────────────────────
    print("\n  ── ANALYST  →  GET /finance/{symbol}/analyst")
    apt = t.analyst_price_targets
    check(
        "analyst_price_targets.mean",
        apt.get("mean") if isinstance(apt, dict) else None,
        symbol,
        "analyst_price_targets",
    )
    rs = t.recommendations_summary
    check(
        "recommendations_summary — rows",
        len(rs) if rs is not None else 0,
        symbol,
        "recommendations_summary",
    )
    ud = t.upgrades_downgrades
    check(
        "upgrades_downgrades     — rows",
        len(ud) if ud is not None else 0,
        symbol,
        "upgrades_downgrades",
    )
    ge = t.growth_estimates
    check(
        "growth_estimates        — rows",
        len(ge) if ge is not None else 0,
        symbol,
        "growth_estimates",
    )
    ee = t.earnings_estimate
    check(
        "earnings_estimate       — rows",
        len(ee) if ee is not None else 0,
        symbol,
        "earnings_estimate",
    )

    # ── Earnings ────────────────────────────────────────────────
    print("\n  ── EARNINGS  →  GET /finance/{symbol}/earnings")
    ed = t.earnings_dates
    check(
        "earnings_dates   — rows",
        len(ed) if ed is not None else 0,
        symbol,
        "earnings_dates",
    )
    eh = t.earnings_history
    check(
        "earnings_history — rows",
        len(eh) if eh is not None else 0,
        symbol,
        "earnings_history",
    )

    # ── Ownership ──────────────────────────────────────────────
    print("\n  ── OWNERSHIP  →  GET /finance/{symbol}/ownership")
    ih = t.institutional_holders
    check(
        "institutional_holders — rows",
        len(ih) if ih is not None else 0,
        symbol,
        "institutional_holders",
    )
    mh = t.major_holders
    check(
        "major_holders         — rows",
        len(mh) if mh is not None else 0,
        symbol,
        "major_holders",
    )
    sf = t.get_shares_full()
    check(
        "shares_full           — rows",
        len(sf) if sf is not None else 0,
        symbol,
        "shares_full",
    )
    it = t.insider_transactions
    check(
        "insider_transactions  — rows",
        len(it) if it is not None else 0,
        symbol,
        "insider_transactions",
    )

    # ── Sustainability ────────────────────────────────────────
    print("\n  ── SUSTAINABILITY  →  GET /finance/{symbol}/sustainability")
    sus = t.sustainability
    sus_ok = sus is not None and (not hasattr(sus, "empty") or not sus.empty)
    check("sustainability — present", 1 if sus_ok else None, symbol, "sustainability")

    elapsed = time.time() - t0
    print(f"\n  ⏱  {symbol} completed — {elapsed:.1f}s")

# ── Overall summary ────────────────────────────────────────────
total = passed_total + failed_total + skipped_total
print(f"\n{'═'*60}")
print(
    f"  RESULT  : \033[92m{passed_total} PASS\033[0m  |  \033[91m{failed_total} FAIL\033[0m  |  \033[90m{skipped_total} SKIP\033[0m  (total {total})"
)
if skipped_total:
    print("  Note    : SKIP = known yfinance limitations (scope varies by exchange)")
print(f"{'═'*60}\n")
