# =========================================================================
# prepareForBacktest.R — Convert raw data into Python framework format
# =========================================================================
# Inputs:
#   1. EM_Indices_EUR.csv   — daily prices from getIndexData.R
#   2. raw_factor_data.csv  — monthly factor data from downloadFactors.R
#
# Outputs (in ./backtest_data/ folder):
#   - returns.csv              — monthly returns (shared across all tests)
#   - factor_trailing_pe.csv   — one per factor
#   - factor_forward_pe.csv
#   - factor_price_to_book.csv
#   - ... etc. (10 factor files total)
#   - factor_registry.csv      — maps factor_id → filename, label, direction
#
# The output CSVs have exactly the columns the Python framework expects:
#   returns.csv:  date, country, return
#   factor_*.csv: date, country, factor_value
# =========================================================================

library(dplyr)
library(tidyr)
library(zoo)     # for rollmean / na.locf if needed

source("config.R")

# ---- Configuration -------------------------------------------------------
price_csv        <- "EM_Indices_EUR.csv"     # from getIndexData.R
factor_csv       <- "raw_factor_data.csv"    # from downloadFactors.R
output_dir       <- "backtest_data"

dir.create(output_dir, showWarnings = FALSE)

# =========================================================================
# PART 1: Compute monthly returns from daily price data
# =========================================================================

cat("=== PART 1: Monthly returns ===\n")

prices_raw <- read.csv(price_csv, stringsAsFactors = FALSE)

# The price CSV has columns: Dates, <ticker1>, <ticker2>, ...
# Rename "Dates" to "date"
colnames(prices_raw)[1] <- "date"
prices_raw$date <- as.Date(prices_raw$date)

# Keep only the country equity tickers
equity_tickers <- names(country_tickers)

# Check which tickers are actually present in the price file
# Bloomberg column names may have dots or spaces — normalise
available_cols <- colnames(prices_raw)
cat("Available columns in price file:", paste(head(available_cols, 5), collapse=", "), "...\n")

# Bloomberg tickers in the CSV may appear with dots instead of spaces
# e.g. "M1CNA.Index" or "M1CNA Index". Try both.
ticker_to_col <- sapply(equity_tickers, function(tkr) {
  # Try exact match
  if (tkr %in% available_cols) return(tkr)
  # Try with dot
  tkr_dot <- gsub(" ", ".", tkr)
  if (tkr_dot %in% available_cols) return(tkr_dot)
  return(NA_character_)
})

found_tickers   <- equity_tickers[!is.na(ticker_to_col)]
found_cols      <- ticker_to_col[!is.na(ticker_to_col)]
missing_tickers <- equity_tickers[is.na(ticker_to_col)]

if (length(missing_tickers) > 0) {
  cat("WARNING: These tickers not found in price CSV:\n")
  cat("  ", paste(missing_tickers, collapse = ", "), "\n")
}
cat("Found", length(found_tickers), "of", length(equity_tickers), "tickers.\n")

# Extract and pivot to long format
price_long <- prices_raw %>%
  select(date, all_of(unname(found_cols))) %>%
  pivot_longer(
    cols      = -date,
    names_to  = "col_name",
    values_to = "price"
  ) %>%
  filter(!is.na(price))

# Map column names back to country names
col_to_country <- setNames(
  country_tickers[found_tickers],
  unname(found_cols)
)
price_long$country <- col_to_country[price_long$col_name]

# Compute month-end prices: take the last available price in each month
price_long$year_month <- format(price_long$date, "%Y-%m")
monthly_prices <- price_long %>%
  group_by(country, year_month) %>%
  filter(date == max(date)) %>%
  ungroup() %>%
  select(date, country, price) %>%
  arrange(country, date)

# Compute simple monthly returns
monthly_returns <- monthly_prices %>%
  group_by(country) %>%
  mutate(
    ret = price / lag(price) - 1
  ) %>%
  ungroup() %>%
  filter(!is.na(ret)) %>%
  select(date, country, ret) %>%
  rename("return" = ret)

# Save returns CSV
returns_path <- file.path(output_dir, "returns.csv")
write.csv(monthly_returns, returns_path, row.names = FALSE)
cat("Saved:", returns_path, "(", nrow(monthly_returns), "rows )\n\n")

# =========================================================================
# PART 2: Prepare fundamental factor CSVs
# =========================================================================

cat("=== PART 2: Fundamental factors ===\n")

factor_raw <- read.csv(factor_csv, stringsAsFactors = FALSE)
factor_raw$date <- as.Date(factor_raw$date)

# Map Bloomberg field names to factor IDs
field_to_id <- c(
  "PE_RATIO"               = "trailing_pe",
  "BEST_PE_RATIO"          = "forward_pe",
  "PX_TO_BOOK_RATIO"       = "price_to_book",
  "EQY_DVD_YLD_IND"        = "dividend_yield",
  "BEST_ROE"               = "forward_roe",
  "PX_TO_CASH_FLOW"        = "price_to_cf",
  "BEST_CUR_EV_TO_EBITDA"  = "forward_ev_ebitda"
)

for (fld in names(field_to_id)) {
  fid   <- field_to_id[[fld]]
  label <- all_factors_meta$label[all_factors_meta$factor_id == fid]

  factor_subset <- factor_raw %>%
    filter(field == fld) %>%
    select(date, country, value) %>%
    rename(factor_value = value) %>%
    filter(!is.na(factor_value)) %>%
    arrange(date, country)

  if (nrow(factor_subset) == 0) {
    cat("  SKIPPED (no data):", label, "\n")
    next
  }

  out_path <- file.path(output_dir, paste0("factor_", fid, ".csv"))
  write.csv(factor_subset, out_path, row.names = FALSE)
  cat("  Saved:", out_path, " (", label, ",", nrow(factor_subset), "rows)\n")
}

# =========================================================================
# PART 3: Compute derived factors
# =========================================================================

cat("\n=== PART 3: Derived factors ===\n")

# --- 3a. Momentum factors (from monthly prices) --------------------------
# 12M Momentum: trailing 12-month return, skipping the most recent month
#               (standard in the literature to avoid short-term reversal)
# 3M Momentum:  trailing 3-month return

cat("Computing momentum factors...\n")

momentum_data <- monthly_prices %>%
  arrange(country, date) %>%
  group_by(country) %>%
  mutate(
    # Price 1 month ago
    price_lag1  = lag(price, 1),
    # Price 3 months ago
    price_lag3  = lag(price, 3),
    # Price 12 months ago
    price_lag12 = lag(price, 12),

    # 12M momentum: return from t-12 to t-1 (skip most recent month)
    mom_12m = price_lag1 / price_lag12 - 1,

    # 3M momentum: return from t-3 to t (include most recent month)
    mom_3m  = price / price_lag3 - 1
  ) %>%
  ungroup()

# Export 12M Momentum
mom12_out <- momentum_data %>%
  filter(!is.na(mom_12m)) %>%
  select(date, country, mom_12m) %>%
  rename(factor_value = mom_12m)

write.csv(mom12_out,
          file.path(output_dir, "factor_mom_12m.csv"),
          row.names = FALSE)
cat("  Saved: factor_mom_12m.csv (", nrow(mom12_out), "rows)\n")

# Export 3M Momentum
mom3_out <- momentum_data %>%
  filter(!is.na(mom_3m)) %>%
  select(date, country, mom_3m) %>%
  rename(factor_value = mom_3m)

write.csv(mom3_out,
          file.path(output_dir, "factor_mom_3m.csv"),
          row.names = FALSE)
cat("  Saved: factor_mom_3m.csv (", nrow(mom3_out), "rows)\n")


# --- 3b. Earnings Revision (3-month change in Forward EPS) ----------------
# Measures how analyst consensus has shifted — a strong sentiment signal.
# Positive revision = analysts upgrading earnings expectations.

cat("Computing earnings revision...\n")

eps_data <- factor_raw %>%
  filter(field == "BEST_EPS") %>%
  select(date, country, value) %>%
  rename(eps = value) %>%
  arrange(country, date)

if (nrow(eps_data) > 0) {
  earnings_rev <- eps_data %>%
    group_by(country) %>%
    mutate(
      eps_lag3 = lag(eps, 3),
      # Percentage change in consensus EPS over 3 months
      revision = (eps - eps_lag3) / abs(eps_lag3)
    ) %>%
    ungroup() %>%
    filter(!is.na(revision) & is.finite(revision)) %>%
    select(date, country, revision) %>%
    rename(factor_value = revision)

  write.csv(earnings_rev,
            file.path(output_dir, "factor_earnings_revision_3m.csv"),
            row.names = FALSE)
  cat("  Saved: factor_earnings_revision_3m.csv (", nrow(earnings_rev), "rows)\n")
} else {
  cat("  SKIPPED earnings revision (no BEST_EPS data)\n")
}


# =========================================================================
# PART 4: Write the factor registry
# =========================================================================

cat("\n=== PART 4: Factor registry ===\n")

registry <- all_factors_meta %>%
  mutate(
    filename = paste0("factor_", factor_id, ".csv"),
    returns_file = "returns.csv"
  )

registry_path <- file.path(output_dir, "factor_registry.csv")
write.csv(registry, registry_path, row.names = FALSE)
cat("Saved:", registry_path, "\n")

cat("\n--- Factor Registry ---\n")
print(registry[, c("factor_id", "label", "higher_is_better", "filename")])

cat("\n=========================================================\n")
cat("All done! Your backtest_data/ folder now contains:\n")
cat("  - returns.csv (shared monthly returns)\n")
cat("  - factor_*.csv (one per factor)\n")
cat("  - factor_registry.csv (maps factors to files & settings)\n")
cat("\nTo run a backtest in Python:\n")
cat("  from factor_backtest import FactorBacktest, BacktestConfig\n")
cat("  cfg = BacktestConfig(n_buckets=5, higher_is_better=False,\n")
cat("                       factor_name='Trailing PE')\n")
cat("  bt = FactorBacktest(cfg)\n")
cat("  result = bt.run(\n")
cat("      factor_csv='backtest_data/factor_trailing_pe.csv',\n")
cat("      returns_csv='backtest_data/returns.csv'\n")
cat("  )\n")
cat("  result.print_summary()\n")
cat("  result.plot_all()\n")
cat("=========================================================\n")
