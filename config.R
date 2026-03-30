# =========================================================================
# config.R — Shared universe definition and factor metadata
# =========================================================================
# Source this file from both downloadFactors.R and prepareForBacktest.R
# to keep ticker-to-country mapping in one place.
# =========================================================================

# ---- Country equity index universe (MSCI single-country indices) --------
# These are the tickers used for quintile-sort factor backtesting.
# Excludes the EM benchmark (MIMUEMRN), regional composites (MXLA),
# and all bond indices.

country_tickers <- c(
  "M1CNA Index"  = "China",
  "MXIN Index"   = "India",
  "MXKR Index"   = "South Korea",
  "TAMSCI Index"  = "Taiwan",
  "MXID Index"   = "Indonesia",
  "MXMY Index"   = "Malaysia",
  "MXTH Index"   = "Thailand",
  "MXPH Index"   = "Philippines",
  "MXBR Index"   = "Brazil",
  "MXMX Index"   = "Mexico",
  "MXZA Index"   = "South Africa",
  "MXTR Index"   = "Turkey",
  "MXGR Index"   = "Greece",
  "MXPL Index"   = "Poland",
  "MXSA Index"   = "Saudi Arabia"
)

# ---- Factor definitions --------------------------------------------------
# Each factor: Bloomberg field name, human-readable label, direction
#   higher_is_better = FALSE → low values are attractive (value metrics)
#   higher_is_better = TRUE  → high values are attractive (quality/momentum)
#
# FUNDAMENTAL FACTORS (downloaded directly from Bloomberg):
#   1. Trailing PE         — classic value
#   2. Forward PE          — consensus value, potentially more timely
#   3. Price-to-Book       — deep value, strong EM evidence (Van der Hart 2003)
#   4. Dividend Yield      — value/income, also proxies shareholder discipline
#   5. Forward ROE         — quality/profitability
#   6. Price-to-Cash-Flow  — value, more manipulation-resistant than PE
#   7. Forward EV/EBITDA   — capital-structure-adjusted value
#
# DERIVED FACTORS (computed from price data or Bloomberg data):
#   8. 12-month Momentum   — very strong EM evidence (Rouwenhorst 1999)
#   9. 3-month Momentum    — shorter-term signal, tests persistence
#  10. Earnings Revision   — change in consensus EPS, sentiment/expectations

bloomberg_factors <- data.frame(
  field = c(
    "PE_RATIO",
    "BEST_PE_RATIO",
    "PX_TO_BOOK_RATIO",
    "EQY_DVD_YLD_IND",
    "BEST_ROE",
    "PX_TO_CASH_FLOW",
    "BEST_CUR_EV_TO_EBITDA",
    "BEST_EPS"               # used to compute earnings revision
  ),
  label = c(
    "Trailing PE",
    "Forward PE",
    "Price-to-Book",
    "Dividend Yield",
    "Forward ROE",
    "Price-to-Cash-Flow",
    "Forward EV-EBITDA",
    "Forward EPS"            # raw download; revision computed later
  ),
  stringsAsFactors = FALSE
)

# Derived factors metadata (for the preparation script)
derived_factors <- data.frame(
  name = c(
    "Mom_12M",
    "Mom_3M",
    "Earnings_Revision_3M"
  ),
  label = c(
    "12M Momentum",
    "3M Momentum",
    "Earnings Revision (3M)"
  ),
  higher_is_better = c(TRUE, TRUE, TRUE),
  stringsAsFactors = FALSE
)

# Combined factor metadata for the Python backtest config
# (used by prepareForBacktest.R when writing factor CSVs)
all_factors_meta <- data.frame(
  factor_id = c(
    "trailing_pe", "forward_pe", "price_to_book", "dividend_yield",
    "forward_roe", "price_to_cf", "forward_ev_ebitda",
    "mom_12m", "mom_3m", "earnings_revision_3m"
  ),
  label = c(
    "Trailing PE", "Forward PE", "Price-to-Book", "Dividend Yield",
    "Forward ROE", "Price-to-Cash-Flow", "Forward EV/EBITDA",
    "12M Momentum", "3M Momentum", "Earnings Revision (3M)"
  ),
  higher_is_better = c(
    FALSE, FALSE, FALSE, TRUE,
    TRUE, FALSE, FALSE,
    TRUE, TRUE, TRUE
  ),
  stringsAsFactors = FALSE
)

cat("config.R loaded:", length(country_tickers), "countries,",
    nrow(all_factors_meta), "factors defined.\n")
