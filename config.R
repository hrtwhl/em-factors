# =========================================================================
# config.R — Universe definition and comprehensive factor metadata
# =========================================================================
# Source this from all R scripts to keep everything in one place.
# =========================================================================

# =========================================================================
# SECTION 1: Country universe
# =========================================================================

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

# Country names (convenience vector)
country_names <- unname(country_tickers)


# =========================================================================
# SECTION 2: Index-level fundamental factors (from MSCI index tickers)
# =========================================================================
# These are Bloomberg fields you can pull directly from the MSCI index
# tickers above.  Downloaded by downloadFactors.R.

bloomberg_factors <- data.frame(
  field = c(
    # --- VALUE ---
    "PE_RATIO",                  #  1. Trailing PE
    "BEST_PE_RATIO",             #  2. Forward (consensus) PE
    "PX_TO_BOOK_RATIO",          #  3. Price-to-Book
    "EQY_DVD_YLD_IND",           #  4. Trailing Dividend Yield
    "PX_TO_CASH_FLOW",           #  5. Price-to-Cash-Flow
    "BEST_CUR_EV_TO_EBITDA",     #  6. Forward EV/EBITDA
    "PX_TO_SALES_RATIO",         #  7. Price-to-Sales
    "PX_TO_FREE_CASH_FLOW",      #  8. Price-to-Free-Cash-Flow
    "BEST_DIV_YLD",              #  9. Forward Dividend Yield

    # --- QUALITY / PROFITABILITY ---
    "BEST_ROE",                  # 10. Forward ROE
    "RETURN_COM_EQY",            # 11. Trailing ROE
    "RETURN_ON_ASSET",           # 12. Trailing ROA
    "PROF_MARGIN",               # 13. Profit Margin
    "BEST_CUR_EV_TO_BEST_EBITDA",# 14. EV/EBITDA (alt. field, fallback)
    "BEST_NET_DEBT_TO_EBITDA",   # 15. Net Debt / EBITDA (leverage)

    # --- GROWTH ---
    "BEST_EPS_NXT_YR",           # 16. Forward EPS (next FY) — for revision
    "BEST_EPS",                  # 17. Forward EPS (current FY) — for revision
    "BEST_SALES",                # 18. Forward Sales — for sales revision
    "TRAIL_12M_EPS"              # 19. Trailing 12M EPS — for trailing revision
  ),
  label = c(
    "Trailing PE",
    "Forward PE",
    "Price-to-Book",
    "Dividend Yield",
    "Price-to-Cash-Flow",
    "Forward EV/EBITDA",
    "Price-to-Sales",
    "Price-to-Free-Cash-Flow",
    "Forward Dividend Yield",

    "Forward ROE",
    "Trailing ROE",
    "Trailing ROA",
    "Profit Margin",
    "EV/EBITDA (alt)",
    "Net Debt to EBITDA",

    "Forward EPS (next FY)",
    "Forward EPS (current FY)",
    "Forward Sales",
    "Trailing 12M EPS"
  ),
  stringsAsFactors = FALSE
)


# =========================================================================
# SECTION 3: Macro factor tickers (per-country, separate from indices)
# =========================================================================
# These require a SEPARATE download script (downloadMacro.R) because
# the tickers are country-specific, not the MSCI index tickers.
#
# NOTE: Some tickers may not work or may have short histories.
# The download script handles failures gracefully — verify the coverage
# summary it prints before relying on the data.

# --- FX spot rates (vs USD) -----------------------------------------------
# Used to compute: currency momentum, currency carry (with rate diffs)
# Greece uses EUR, so FX signal is less meaningful for Greece specifically.
fx_tickers <- c(
  "China"        = "USDCNY Curncy",
  "India"        = "USDINR Curncy",
  "South Korea"  = "USDKRW Curncy",
  "Taiwan"       = "USDTWD Curncy",
  "Indonesia"    = "USDIDR Curncy",
  "Malaysia"     = "USDMYR Curncy",
  "Thailand"     = "USDTHB Curncy",
  "Philippines"  = "USDPHP Curncy",
  "Brazil"       = "USDBRL Curncy",
  "Mexico"       = "USDMXN Curncy",
  "South Africa" = "USDZAR Curncy",
  "Turkey"       = "USDTRY Curncy",
  "Greece"       = "EURUSD Curncy",   # inverted — handle in prep script
  "Poland"       = "USDPLN Curncy",
  "Saudi Arabia" = "USDSAR Curncy"    # pegged, will show no signal
)

# --- Sovereign CDS 5Y (USD, mid spread in bps) ----------------------------
# Used for: sovereign risk signal (low CDS = less risky)
# These are Markit CDS tickers.  Some may need adjustment in your
# Bloomberg terminal.  Run the download and check the coverage output.
cds_tickers <- c(
  "China"        = "CCHN1U5 CBGN Curncy",
  "India"        = "CIND1U5 CBGN Curncy",
  "South Korea"  = "CKOR1U5 CBGN Curncy",
  "Taiwan"       = "CTAI1U5 CBGN Curncy",  # may be unavailable
  "Indonesia"    = "CIDN1U5 CBGN Curncy",
  "Malaysia"     = "CMAL1U5 CBGN Curncy",
  "Thailand"     = "CTHA1U5 CBGN Curncy",
  "Philippines"  = "CPHI1U5 CBGN Curncy",
  "Brazil"       = "CBRA1U5 CBGN Curncy",
  "Mexico"       = "CMEX1U5 CBGN Curncy",
  "South Africa" = "CSAF1U5 CBGN Curncy",
  "Turkey"       = "CTUR1U5 CBGN Curncy",
  "Greece"       = "CGRK1U5 CBGN Curncy",  # post-restructuring only
  "Poland"       = "CPOL1U5 CBGN Curncy",
  "Saudi Arabia" = "CSAU1U5 CBGN Curncy"
)

# --- 10Y Government Bond Yields -------------------------------------------
# Used for: carry (yield level), real yield (yield - inflation),
#           yield change momentum
# These are Bloomberg generic benchmark tickers.
govt10y_tickers <- c(
  "China"        = "GCNY10YR Index",
  "India"        = "GIND10YR Index",
  "South Korea"  = "GVSK10YR Index",
  "Taiwan"       = "GTWN10Y Index",
  "Indonesia"    = "GIDN10YR Index",
  "Malaysia"     = "GMYR10Y Index",
  "Thailand"     = "GTHB10YR Index",
  "Philippines"  = "GPHI10YR Index",
  "Brazil"       = "GEBR10Y Index",
  "Mexico"       = "GMXN10YR Index",
  "South Africa" = "GSAB10YR Index",
  "Turkey"       = "GTUR10YR Index",
  "Greece"       = "GGGB10YR Index",
  "Poland"       = "GPOL10YR Index",
  "Saudi Arabia" = "GSAR10Y Index"
)

# --- Policy / Short-term rates (central bank or 3M equivalent) ------------
# Used for: carry (short rate differential), monetary policy stance
policy_rate_tickers <- c(
  "China"        = "CHLR12M Index",   # 1Y Loan Prime Rate
  "India"        = "INRPYLDP Index",  # RBI repo rate
  "South Korea"  = "KORP7DR Index",   # BOK base rate
  "Taiwan"       = "TADISC Index",    # CBC discount rate
  "Indonesia"    = "IDBIRATE Index",  # BI rate
  "Malaysia"     = "BNMROVER Index",  # BNM OPR
  "Thailand"     = "BTRR1DAY Index",  # BOT repo rate
  "Philippines"  = "PPCBKRRP Index",  # BSP overnight RRP
  "Brazil"       = "BZSTSETA Index",  # SELIC target
  "Mexico"       = "MXONBR Index",    # Banxico overnight rate
  "South Africa" = "SARBREP Index",   # SARB repo rate
  "Turkey"       = "TURSEFT Index",   # CBRT policy rate
  "Greece"       = "EURR002W Index",  # ECB refi rate (shared with Eurozone)
  "Poland"       = "NBPRATE Index",   # NBP reference rate
  "Saudi Arabia" = "SARRREP Index"    # SAMA repo rate
)

# --- CPI / Inflation (YoY %) -----------------------------------------------
# Used for: real yield computation, inflation momentum
cpi_tickers <- c(
  "China"        = "CNCPIYOY Index",
  "India"        = "INFUTOTY Index",
  "South Korea"  = "KOCPIYOY Index",
  "Taiwan"       = "TWCPIYOY Index",
  "Indonesia"    = "IDCPIY Index",
  "Malaysia"     = "MACPIYOY Index",
  "Thailand"     = "THCPIYOY Index",
  "Philippines"  = "PHC2II Index",
  "Brazil"       = "BZPIIPCY Index",
  "Mexico"       = "MXCPYOY Index",
  "South Africa" = "SACPIYOY Index",
  "Turkey"       = "TUCPIY Index",
  "Greece"       = "GKCPNEUY Index",
  "Poland"       = "POCPIYOY Index",
  "Saudi Arabia" = "SACPIYOY Index"   # may clash with South Africa ticker
  # If Bloomberg returns an error, try "SACPI Index" or search manually
)

# --- Current Account (% of GDP) -------------------------------------------
# Used for: external vulnerability signal
# These are quarterly/annual — sparse but important macro signal.
# Bloomberg ECFC (economics) tickers.
current_account_tickers <- c(
  "China"        = "CNFRBAL% Index",
  "India"        = "INBPCAB% Index",
  "South Korea"  = "KOBPCA% Index",
  "Taiwan"       = "TWBPCAG% Index",
  "Indonesia"    = "IDBPCAG% Index",
  "Malaysia"     = "MYBPCAG% Index",
  "Thailand"     = "THBPCAG% Index",
  "Philippines"  = "PHBPCAG% Index",
  "Brazil"       = "BZBPCAG% Index",
  "Mexico"       = "MXBPCAG% Index",
  "South Africa" = "SABPCAG% Index",
  "Turkey"       = "TUBPCAG% Index",
  "Greece"       = "GRBPCAG% Index",
  "Poland"       = "POBPCAG% Index",
  "Saudi Arabia" = "SABPCAG% Index"  # may conflict with South Africa
)


# =========================================================================
# SECTION 4: Complete factor registry
# =========================================================================
# Master list of all factors the Python framework will test.
# factor_id must match the CSV filename: factor_{factor_id}.csv

all_factors_meta <- data.frame(
  factor_id = c(
    # Index fundamentals — VALUE
    "trailing_pe", "forward_pe", "price_to_book", "dividend_yield",
    "price_to_cf", "forward_ev_ebitda", "price_to_sales",
    "price_to_fcf", "forward_div_yield",

    # Index fundamentals — QUALITY
    "forward_roe", "trailing_roe", "trailing_roa", "profit_margin",
    "net_debt_to_ebitda",

    # Price-derived
    "mom_12m", "mom_6m", "mom_3m", "mom_1m",
    "vol_12m", "risk_adj_mom_12m",
    "max_dd_12m", "beta_to_em",

    # Earnings / sentiment
    "earnings_revision_3m", "sales_revision_3m",

    # Macro
    "cds_5y", "cds_3m_change",
    "bond_yield_10y", "real_yield_10y",
    "yield_curve", "carry",
    "fx_mom_3m", "fx_mom_12m",
    "cpi_yoy", "cpi_3m_change",
    "current_account"
  ),
  label = c(
    "Trailing PE", "Forward PE", "Price-to-Book", "Dividend Yield",
    "Price-to-Cash-Flow", "Forward EV/EBITDA", "Price-to-Sales",
    "Price-to-Free-Cash-Flow", "Forward Dividend Yield",

    "Forward ROE", "Trailing ROE", "Trailing ROA", "Profit Margin",
    "Net Debt / EBITDA",

    "12M Momentum", "6M Momentum", "3M Momentum", "1M Momentum (Reversal)",
    "12M Realized Volatility", "Risk-Adjusted Momentum (12M)",
    "Max Drawdown (12M)", "Beta to EM Index",

    "Earnings Revision (3M)", "Sales Revision (3M)",

    "Sovereign CDS 5Y", "CDS 3M Change",
    "10Y Govt Bond Yield", "Real 10Y Yield",
    "Yield Curve (10Y - Policy)", "Carry (Policy Rate)",
    "FX Momentum 3M", "FX Momentum 12M",
    "CPI YoY", "CPI 3M Change",
    "Current Account (% GDP)"
  ),
  higher_is_better = c(
    # Value: low is attractive
    FALSE, FALSE, FALSE, TRUE,   # PE, PE, PB, DivYld (high yield = good)
    FALSE, FALSE, FALSE,          # PCF, EV/EBITDA, PS
    FALSE, TRUE,                   # PFCF, Fwd DivYld (high = good)

    # Quality: high is attractive (except leverage)
    TRUE, TRUE, TRUE, TRUE,       # ROE, ROE, ROA, Margin
    FALSE,                         # Leverage: low is better

    # Price-derived
    TRUE, TRUE, TRUE, FALSE,      # Momentum: high = good; 1M = reversal
    FALSE, TRUE,                   # Low vol = good; high risk-adj mom = good
    FALSE, FALSE,                  # Less drawdown = good; low beta = defensive

    # Sentiment
    TRUE, TRUE,                    # Positive revision = good

    # Macro
    FALSE, FALSE,                  # Low CDS = good; CDS tightening = good
    TRUE, TRUE,                    # High yield = carry; high real yield = carry
    TRUE, TRUE,                    # Steep curve = good; high carry = good
    TRUE, TRUE,                    # FX appreciation = good (we quote USDXXX, so inverted)
    FALSE, FALSE,                  # Low inflation = quality; falling inflation = good
    TRUE                           # Strong current account = good
  ),
  category = c(
    rep("Value", 9),
    rep("Quality", 5),
    rep("Momentum/Risk", 8),
    rep("Sentiment", 2),
    rep("Macro", 11)
  ),
  stringsAsFactors = FALSE
)

cat("config.R loaded:", length(country_tickers), "countries,",
    nrow(all_factors_meta), "factors defined.\n")
cat("  Categories:", paste(unique(all_factors_meta$category), collapse=", "), "\n")
