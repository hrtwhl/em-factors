# =========================================================================
# prepareForBacktest.R — Convert all raw data into Python framework format
# =========================================================================
# Inputs:
#   1. EM_Indices_EUR.csv    — daily prices from getIndexData.R
#   2. raw_factor_data.csv   — fundamental data from downloadFactors.R
#   3. raw_macro_data.csv    — macro data from downloadMacro.R
#
# Outputs (in ./backtest_data/ folder):
#   - returns.csv              — monthly returns (shared across all tests)
#   - factor_<id>.csv          — one per factor
#   - factor_registry.csv      — maps factor_id → filename, direction, label
#
# Output CSVs match the Python framework format:
#   returns.csv:   date, country, return
#   factor_*.csv:  date, country, factor_value
# =========================================================================

library(dplyr)
library(tidyr)
library(zoo)

source("config.R")

# ---- Configuration -------------------------------------------------------
price_csv  <- "EM_Indices_EUR.csv"
factor_csv <- "raw_factor_data.csv"
macro_csv  <- "raw_macro_data.csv"
output_dir <- "backtest_data"

dir.create(output_dir, showWarnings = FALSE)

# Helper: save a factor CSV and print status
save_factor <- function(df, factor_id, label) {
  if (nrow(df) == 0) {
    cat(sprintf("  SKIPPED %-40s (no data)\n", label))
    return(FALSE)
  }

  # Standardise columns
  df <- df %>%
    select(date, country, factor_value) %>%
    filter(!is.na(factor_value) & is.finite(factor_value)) %>%
    arrange(date, country)

  path <- file.path(output_dir, paste0("factor_", factor_id, ".csv"))
  write.csv(df, path, row.names = FALSE)
  n_ctry <- n_distinct(df$country)
  cat(sprintf("  SAVED   %-40s (%5d rows, %2d countries)\n",
              label, nrow(df), n_ctry))
  return(TRUE)
}


# =========================================================================
# PART 1: Monthly returns from daily prices
# =========================================================================
cat("=== PART 1: Monthly returns ===\n")

prices_raw <- read.csv(price_csv, stringsAsFactors = FALSE)
colnames(prices_raw)[1] <- "date"
prices_raw$date <- as.Date(prices_raw$date)

# Map Bloomberg column names to country names
equity_tickers <- names(country_tickers)
available_cols <- colnames(prices_raw)

ticker_to_col <- sapply(equity_tickers, function(tkr) {
  if (tkr %in% available_cols) return(tkr)
  tkr_dot <- gsub(" ", ".", tkr)
  if (tkr_dot %in% available_cols) return(tkr_dot)
  return(NA_character_)
})

found     <- equity_tickers[!is.na(ticker_to_col)]
found_cols <- ticker_to_col[!is.na(ticker_to_col)]
missing   <- equity_tickers[is.na(ticker_to_col)]

if (length(missing) > 0) {
  cat("WARNING: Tickers not found in price CSV:\n")
  cat("  ", paste(missing, collapse = ", "), "\n")
}
cat("Found", length(found), "of", length(equity_tickers), "tickers in price file.\n")

# Pivot to long
price_long <- prices_raw %>%
  select(date, all_of(unname(found_cols))) %>%
  pivot_longer(cols = -date, names_to = "col_name", values_to = "price") %>%
  filter(!is.na(price))

col_to_country <- setNames(country_tickers[found], unname(found_cols))
price_long$country <- col_to_country[price_long$col_name]

# Month-end prices
price_long$year_month <- format(price_long$date, "%Y-%m")
monthly_prices <- price_long %>%
  group_by(country, year_month) %>%
  filter(date == max(date)) %>%
  ungroup() %>%
  select(date, country, price) %>%
  arrange(country, date)

# Monthly returns
monthly_returns <- monthly_prices %>%
  group_by(country) %>%
  mutate(ret = price / lag(price) - 1) %>%
  ungroup() %>%
  filter(!is.na(ret)) %>%
  select(date, country, ret) %>%
  rename("return" = ret)

write.csv(monthly_returns, file.path(output_dir, "returns.csv"), row.names = FALSE)
cat("Saved: returns.csv (", nrow(monthly_returns), "rows )\n\n")


# =========================================================================
# PART 2: Index-level fundamental factors
# =========================================================================
cat("=== PART 2: Index fundamental factors ===\n")

factor_raw <- read.csv(factor_csv, stringsAsFactors = FALSE)
factor_raw$date <- as.Date(factor_raw$date)

# Map Bloomberg fields → factor IDs (only the direct pass-through ones)
field_to_id <- c(
  "PE_RATIO"               = "trailing_pe",
  "BEST_PE_RATIO"          = "forward_pe",
  "PX_TO_BOOK_RATIO"       = "price_to_book",
  "EQY_DVD_YLD_IND"        = "dividend_yield",
  "PX_TO_CASH_FLOW"        = "price_to_cf",
  "BEST_CUR_EV_TO_EBITDA"  = "forward_ev_ebitda",
  "PX_TO_SALES_RATIO"      = "price_to_sales",
  "PX_TO_FREE_CASH_FLOW"   = "price_to_fcf",
  "BEST_DIV_YLD"           = "forward_div_yield",
  "BEST_ROE"               = "forward_roe",
  "RETURN_COM_EQY"         = "trailing_roe",
  "RETURN_ON_ASSET"        = "trailing_roa",
  "PROF_MARGIN"            = "profit_margin",
  "BEST_NET_DEBT_TO_EBITDA"= "net_debt_to_ebitda"
)

available_fields <- unique(factor_raw$field)

for (fld in names(field_to_id)) {
  fid   <- field_to_id[[fld]]
  label <- all_factors_meta$label[all_factors_meta$factor_id == fid]
  if (length(label) == 0) label <- fid

  if (!(fld %in% available_fields)) {
    cat(sprintf("  SKIPPED %-40s (field %s not in download)\n", label, fld))
    next
  }

  df <- factor_raw %>%
    filter(field == fld) %>%
    select(date, country, value) %>%
    rename(factor_value = value)

  save_factor(df, fid, label)
}


# =========================================================================
# PART 3: Price-derived factors
# =========================================================================
cat("\n=== PART 3: Price-derived factors ===\n")

# Enrich monthly_prices with lagged prices for various windows
mp <- monthly_prices %>%
  arrange(country, date) %>%
  group_by(country) %>%
  mutate(
    price_lag1  = lag(price, 1),
    price_lag3  = lag(price, 3),
    price_lag6  = lag(price, 6),
    price_lag12 = lag(price, 12),
    ret         = price / lag(price) - 1
  ) %>%
  ungroup()

# 3a. Momentum factors
# 12M: t-12 to t-1 (skip most recent month — standard)
save_factor(
  mp %>% filter(!is.na(price_lag1) & !is.na(price_lag12)) %>%
    mutate(factor_value = price_lag1 / price_lag12 - 1) %>%
    select(date, country, factor_value),
  "mom_12m", "12M Momentum"
)

# 6M: t-6 to t
save_factor(
  mp %>% filter(!is.na(price_lag6)) %>%
    mutate(factor_value = price / price_lag6 - 1) %>%
    select(date, country, factor_value),
  "mom_6m", "6M Momentum"
)

# 3M: t-3 to t
save_factor(
  mp %>% filter(!is.na(price_lag3)) %>%
    mutate(factor_value = price / price_lag3 - 1) %>%
    select(date, country, factor_value),
  "mom_3m", "3M Momentum"
)

# 1M: most recent month return (reversal signal)
save_factor(
  mp %>% filter(!is.na(ret)) %>%
    mutate(factor_value = ret) %>%
    select(date, country, factor_value),
  "mom_1m", "1M Momentum (Reversal)"
)

# 3b. Realised Volatility (trailing 12M, from daily returns)
cat("Computing realised volatility from daily data...\n")

daily_returns <- price_long %>%
  arrange(country, date) %>%
  group_by(country) %>%
  mutate(daily_ret = price / lag(price) - 1) %>%
  ungroup() %>%
  filter(!is.na(daily_ret))

# For each country × month-end, compute trailing 252-day annualised vol
vol_data <- daily_returns %>%
  mutate(year_month = format(date, "%Y-%m")) %>%
  group_by(country, year_month) %>%
  summarise(
    date      = max(date),
    n_days    = n(),
    vol_daily = sd(daily_ret, na.rm = TRUE),
    .groups   = "drop"
  ) %>%
  filter(n_days >= 15) %>%
  mutate(factor_value = vol_daily * sqrt(252))  # annualise

# But this is just 1-month vol. For 12M trailing vol, we need rolling.
# Use the monthly return series instead (simpler, avoids daily complexity).
vol_12m <- mp %>%
  filter(!is.na(ret)) %>%
  arrange(country, date) %>%
  group_by(country) %>%
  mutate(
    factor_value = rollapplyr(ret, width = 12, FUN = sd, fill = NA) * sqrt(12)
  ) %>%
  ungroup() %>%
  filter(!is.na(factor_value)) %>%
  select(date, country, factor_value)

save_factor(vol_12m, "vol_12m", "12M Realized Volatility")

# 3c. Risk-adjusted Momentum: 12M momentum / 12M vol
mom12 <- mp %>%
  filter(!is.na(price_lag1) & !is.na(price_lag12)) %>%
  mutate(mom = price_lag1 / price_lag12 - 1) %>%
  select(date, country, mom)

risk_adj <- mom12 %>%
  inner_join(vol_12m %>% rename(vol = factor_value), by = c("date", "country")) %>%
  mutate(factor_value = mom / vol) %>%
  filter(is.finite(factor_value)) %>%
  select(date, country, factor_value)

save_factor(risk_adj, "risk_adj_mom_12m", "Risk-Adjusted Momentum (12M)")

# 3d. Max Drawdown (trailing 12M)
cat("Computing 12M max drawdown...\n")

max_dd_data <- price_long %>%
  arrange(country, date) %>%
  group_by(country) %>%
  mutate(
    # Rolling 252-day max drawdown
    year_month = format(date, "%Y-%m")
  ) %>%
  ungroup()

# Compute per month: max drawdown over trailing 252 trading days
dd_monthly <- max_dd_data %>%
  group_by(country) %>%
  arrange(date) %>%
  mutate(
    running_max = cummax(price),
    drawdown    = price / running_max - 1
  ) %>%
  mutate(year_month = format(date, "%Y-%m")) %>%
  group_by(country, year_month) %>%
  summarise(
    date     = max(date),
    max_dd   = min(drawdown),   # most negative
    .groups  = "drop"
  ) %>%
  ungroup()

# For trailing 12M, take the worst monthly max_dd over 12 months
dd_12m <- dd_monthly %>%
  arrange(country, date) %>%
  group_by(country) %>%
  mutate(
    factor_value = rollapplyr(max_dd, width = 12, FUN = min, fill = NA)
  ) %>%
  ungroup() %>%
  filter(!is.na(factor_value)) %>%
  select(date, country, factor_value)

save_factor(dd_12m, "max_dd_12m", "Max Drawdown (12M)")

# 3e. Beta to EM Index (MIMUEMRN)
cat("Computing beta to EM index...\n")

# Check if EM benchmark is in the price file
em_benchmark_col <- NULL
for (candidate in c("MIMUEMRN Index", "MIMUEMRN.Index")) {
  if (candidate %in% available_cols) {
    em_benchmark_col <- candidate
    break
  }
}

if (!is.null(em_benchmark_col)) {
  em_prices <- prices_raw %>%
    select(date, all_of(em_benchmark_col)) %>%
    rename(em_price = 2) %>%
    filter(!is.na(em_price)) %>%
    mutate(year_month = format(date, "%Y-%m")) %>%
    group_by(year_month) %>%
    filter(date == max(date)) %>%
    ungroup() %>%
    arrange(date) %>%
    mutate(em_ret = em_price / lag(em_price) - 1) %>%
    filter(!is.na(em_ret)) %>%
    select(date, year_month, em_ret)

  # Join with country monthly returns
  country_monthly <- monthly_returns %>%
    mutate(year_month = format(date, "%Y-%m"))

  # Rolling 12-month beta
  beta_data <- country_monthly %>%
    inner_join(em_prices %>% select(year_month, em_ret),
               by = "year_month") %>%
    arrange(country, date) %>%
    group_by(country) %>%
    mutate(
      factor_value = rollapplyr(
        cbind(return, em_ret), width = 12,
        FUN = function(x) {
          if (nrow(x) < 12) return(NA)
          cov(x[,1], x[,2]) / var(x[,2])
        },
        by.column = FALSE,
        fill = NA
      )
    ) %>%
    ungroup() %>%
    filter(!is.na(factor_value)) %>%
    select(date, country, factor_value)

  save_factor(beta_data, "beta_to_em", "Beta to EM Index")
} else {
  cat("  SKIPPED Beta to EM — benchmark MIMUEMRN not found in price file.\n")
}


# =========================================================================
# PART 4: Earnings & sales revision factors
# =========================================================================
cat("\n=== PART 4: Earnings/Sales revision factors ===\n")

# Earnings revision: 3-month % change in forward EPS
for (eps_field in c("BEST_EPS", "BEST_EPS_NXT_YR")) {
  fid   <- if (eps_field == "BEST_EPS") "earnings_revision_3m" else NULL
  label <- if (eps_field == "BEST_EPS") "Earnings Revision (3M)" else NULL

  # Only process the primary one
  if (is.null(fid)) next

  if (!(eps_field %in% available_fields)) {
    cat(sprintf("  SKIPPED %-40s (field %s not in download)\n", label, eps_field))
    next
  }

  eps_data <- factor_raw %>%
    filter(field == eps_field) %>%
    select(date, country, value) %>%
    rename(eps = value) %>%
    arrange(country, date)

  revision <- eps_data %>%
    group_by(country) %>%
    mutate(
      eps_lag3  = lag(eps, 3),
      factor_value = (eps - eps_lag3) / abs(eps_lag3)
    ) %>%
    ungroup() %>%
    filter(!is.na(factor_value) & is.finite(factor_value)) %>%
    select(date, country, factor_value)

  save_factor(revision, fid, label)
}

# Sales revision: 3-month % change in forward sales
if ("BEST_SALES" %in% available_fields) {
  sales_data <- factor_raw %>%
    filter(field == "BEST_SALES") %>%
    select(date, country, value) %>%
    rename(sales = value) %>%
    arrange(country, date)

  sales_rev <- sales_data %>%
    group_by(country) %>%
    mutate(
      sales_lag3 = lag(sales, 3),
      factor_value = (sales - sales_lag3) / abs(sales_lag3)
    ) %>%
    ungroup() %>%
    filter(!is.na(factor_value) & is.finite(factor_value)) %>%
    select(date, country, factor_value)

  save_factor(sales_rev, "sales_revision_3m", "Sales Revision (3M)")
} else {
  cat("  SKIPPED Sales Revision (3M) — BEST_SALES not in download.\n")
}


# =========================================================================
# PART 5: Macro factors
# =========================================================================
cat("\n=== PART 5: Macro factors ===\n")

if (!file.exists(macro_csv)) {
  cat("WARNING: raw_macro_data.csv not found. Skipping all macro factors.\n")
  cat("Run downloadMacro.R first to generate this file.\n")
} else {

  macro_raw <- read.csv(macro_csv, stringsAsFactors = FALSE)
  macro_raw$date <- as.Date(macro_raw$date)

  # Helper: extract a series, resample to month-end
  get_monthly_macro <- function(series_name) {
    sub <- macro_raw %>%
      filter(series == series_name) %>%
      mutate(year_month = format(date, "%Y-%m")) %>%
      group_by(country, year_month) %>%
      filter(date == max(date)) %>%
      ungroup() %>%
      select(date, country, value) %>%
      arrange(country, date)
    return(sub)
  }

  # ----- 5a. CDS level -----
  cds_monthly <- get_monthly_macro("cds_5y")
  if (nrow(cds_monthly) > 0) {
    save_factor(
      cds_monthly %>% rename(factor_value = value),
      "cds_5y", "Sovereign CDS 5Y"
    )

    # CDS 3-month change (tightening = improvement = good)
    cds_change <- cds_monthly %>%
      arrange(country, date) %>%
      group_by(country) %>%
      mutate(factor_value = value - lag(value, 3)) %>%  # positive = widening = bad
      ungroup() %>%
      filter(!is.na(factor_value)) %>%
      select(date, country, factor_value)

    save_factor(cds_change, "cds_3m_change", "CDS 3M Change")
  }

  # ----- 5b. Bond yields -----
  yield_monthly <- get_monthly_macro("govt_10y_yield")
  if (nrow(yield_monthly) > 0) {
    save_factor(
      yield_monthly %>% rename(factor_value = value),
      "bond_yield_10y", "10Y Govt Bond Yield"
    )
  }

  # ----- 5c. Policy rate / Carry -----
  policy_monthly <- get_monthly_macro("policy_rate")
  if (nrow(policy_monthly) > 0) {
    save_factor(
      policy_monthly %>% rename(factor_value = value),
      "carry", "Carry (Policy Rate)"
    )
  }

  # ----- 5d. Yield curve (10Y - policy rate) -----
  if (nrow(yield_monthly) > 0 && nrow(policy_monthly) > 0) {
    ycurve <- yield_monthly %>%
      rename(yield_10y = value) %>%
      inner_join(
        policy_monthly %>% rename(policy = value) %>% select(country, date, policy),
        by = c("date", "country")
      ) %>%
      # Match by year_month if exact date join fails
      mutate(factor_value = yield_10y - policy) %>%
      select(date, country, factor_value)

    # If exact date join gives too few rows, try year_month join
    if (nrow(ycurve) < 50) {
      cat("  (Yield curve: retrying with year-month matching...)\n")
      ym <- yield_monthly %>%
        mutate(ym = format(date, "%Y-%m")) %>%
        rename(yield_10y = value)
      pm <- policy_monthly %>%
        mutate(ym = format(date, "%Y-%m")) %>%
        rename(policy = value)
      ycurve <- ym %>%
        inner_join(pm %>% select(country, ym, policy), by = c("country", "ym")) %>%
        mutate(factor_value = yield_10y - policy) %>%
        select(date, country, factor_value)
    }

    save_factor(ycurve, "yield_curve", "Yield Curve (10Y - Policy)")
  }

  # ----- 5e. Real yield (10Y nominal - CPI) -----
  cpi_monthly <- get_monthly_macro("cpi_yoy")
  if (nrow(yield_monthly) > 0 && nrow(cpi_monthly) > 0) {
    # Join on year_month since CPI and yield dates may not align exactly
    ym_yield <- yield_monthly %>%
      mutate(ym = format(date, "%Y-%m")) %>%
      rename(nominal = value)
    ym_cpi <- cpi_monthly %>%
      mutate(ym = format(date, "%Y-%m")) %>%
      rename(cpi = value)

    real_yield <- ym_yield %>%
      inner_join(ym_cpi %>% select(country, ym, cpi), by = c("country", "ym")) %>%
      mutate(factor_value = nominal - cpi) %>%
      select(date, country, factor_value)

    save_factor(real_yield, "real_yield_10y", "Real 10Y Yield")
  }

  # ----- 5f. CPI level and change -----
  if (nrow(cpi_monthly) > 0) {
    save_factor(
      cpi_monthly %>% rename(factor_value = value),
      "cpi_yoy", "CPI YoY"
    )

    cpi_change <- cpi_monthly %>%
      arrange(country, date) %>%
      group_by(country) %>%
      mutate(factor_value = value - lag(value, 3)) %>%
      ungroup() %>%
      filter(!is.na(factor_value)) %>%
      select(date, country, factor_value)

    save_factor(cpi_change, "cpi_3m_change", "CPI 3M Change")
  }

  # ----- 5g. FX momentum -----
  fx_monthly <- get_monthly_macro("fx_spot")
  if (nrow(fx_monthly) > 0) {
    # Convention: tickers are USDXXX, so a rising value = local depreciation.
    # For Greece (EURUSD): rising = EUR appreciation = opposite sign.
    # We want: positive factor_value = local currency appreciated.
    # So for USDXXX: factor = -(fx / lag(fx) - 1)  [negative of USD strength]
    # For EURUSD (Greece): factor = +(fx / lag(fx) - 1)

    fx_mom <- fx_monthly %>%
      arrange(country, date) %>%
      group_by(country) %>%
      mutate(
        fx_ret_3m  = value / lag(value, 3) - 1,
        fx_ret_12m = value / lag(value, 12) - 1
      ) %>%
      ungroup()

    # Flip sign: for USDXXX, negative return means local appreciation
    # Greece (EURUSD) needs opposite treatment
    fx_mom <- fx_mom %>%
      mutate(
        fx_mom_3m  = ifelse(country == "Greece", fx_ret_3m, -fx_ret_3m),
        fx_mom_12m = ifelse(country == "Greece", fx_ret_12m, -fx_ret_12m)
      )

    save_factor(
      fx_mom %>% filter(!is.na(fx_mom_3m)) %>%
        mutate(factor_value = fx_mom_3m) %>%
        select(date, country, factor_value),
      "fx_mom_3m", "FX Momentum 3M"
    )

    save_factor(
      fx_mom %>% filter(!is.na(fx_mom_12m)) %>%
        mutate(factor_value = fx_mom_12m) %>%
        select(date, country, factor_value),
      "fx_mom_12m", "FX Momentum 12M"
    )
  }

  # ----- 5h. Current account -----
  ca_monthly <- get_monthly_macro("current_account_pct_gdp")
  if (nrow(ca_monthly) > 0) {
    # Current account data is sparse (quarterly/annual).
    # Forward-fill to monthly for the backtest.
    ca_filled <- ca_monthly %>%
      arrange(country, date) %>%
      # Create a complete monthly grid per country
      group_by(country) %>%
      mutate(ym = format(date, "%Y-%m")) %>%
      ungroup()

    save_factor(
      ca_filled %>% rename(factor_value = value),
      "current_account", "Current Account (% GDP)"
    )
  }
}


# =========================================================================
# PART 6: Write the factor registry (only for factors that were saved)
# =========================================================================
cat("\n=== PART 6: Factor registry ===\n")

# Check which factor files actually exist
existing_files <- list.files(output_dir, pattern = "^factor_.*\\.csv$")
existing_ids   <- gsub("^factor_|\\.csv$", "", existing_files)

registry <- all_factors_meta %>%
  filter(factor_id %in% existing_ids) %>%
  mutate(
    filename     = paste0("factor_", factor_id, ".csv"),
    returns_file = "returns.csv"
  )

write.csv(registry, file.path(output_dir, "factor_registry.csv"), row.names = FALSE)

cat("Factor registry:", nrow(registry), "of", nrow(all_factors_meta),
    "factors available.\n\n")

cat("--- Available factors by category ---\n")
for (cat_name in unique(registry$category)) {
  sub <- registry %>% filter(category == cat_name)
  cat(sprintf("  %s (%d):\n", cat_name, nrow(sub)))
  for (i in 1:nrow(sub)) {
    dir <- if (sub$higher_is_better[i]) "HIGH=good" else "LOW=good"
    cat(sprintf("    %-40s  (%s)\n", sub$label[i], dir))
  }
}

cat("\n--- Missing factors (not generated) ---\n")
missing_ids <- setdiff(all_factors_meta$factor_id, existing_ids)
if (length(missing_ids) > 0) {
  for (mid in missing_ids) {
    lbl <- all_factors_meta$label[all_factors_meta$factor_id == mid]
    cat(sprintf("  %-40s  (factor_%s.csv)\n", lbl, mid))
  }
} else {
  cat("  None — all factors generated successfully!\n")
}

cat("\n=========================================================\n")
cat("All done! Your backtest_data/ folder is ready.\n")
cat("Copy the entire backtest_data/ folder to your laptop,\n")
cat("then run: python run_all_backtests.py\n")
cat("=========================================================\n")
