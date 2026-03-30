# =========================================================================
# downloadMacro.R — Download macro factor data from Bloomberg
# =========================================================================
# Run this on your university Bloomberg machine, AFTER downloadFactors.R.
#
# This downloads country-level macro data (CDS, yields, FX, inflation,
# policy rates, current account) using DIFFERENT Bloomberg tickers per
# country (not the MSCI index tickers).
#
# IMPORTANT: Some tickers may not work on your terminal. The script
# handles failures gracefully. Check the coverage summary at the end
# and manually verify any tickers that fail.
#
# Output:
#   raw_macro_data.csv — long-format: date, country, series, value
# =========================================================================

library(Rblpapi)
library(dplyr)

source("config.R")

# ---- Bloomberg connection ------------------------------------------------
con <- blpConnect()

# ---- Parameters ----------------------------------------------------------
start_date <- as.Date("2005-01-01")
end_date   <- Sys.Date()

opt_daily   <- c("periodicitySelection" = "DAILY")
opt_monthly <- c("periodicitySelection" = "MONTHLY")

# ---- Helper function: download a named vector of tickers -----------------
download_series <- function(ticker_vec, series_name, periodicity = "DAILY") {
  cat(sprintf("\n--- Downloading: %s (%d tickers, %s) ---\n",
              series_name, length(ticker_vec), periodicity))

  opt <- if (periodicity == "DAILY") opt_daily else opt_monthly
  results <- list()

  for (country in names(ticker_vec)) {
    tkr <- ticker_vec[country]
    cat(sprintf("  %-15s %-30s ... ", country, tkr))

    tryCatch({
      df <- bdh(
        securities = tkr,
        fields     = "PX_LAST",
        start.date = start_date,
        end.date   = end_date,
        options    = opt
      )

      if (is.null(df) || nrow(df) == 0) {
        cat("NO DATA\n")
        next
      }

      colnames(df) <- c("date", "value")
      df$country <- country
      df$series  <- series_name
      results[[country]] <- df
      cat("OK (", nrow(df), "rows )\n")

    }, error = function(e) {
      cat("FAILED:", conditionMessage(e), "\n")
    })
  }

  if (length(results) == 0) return(data.frame())
  return(bind_rows(results))
}

# ---- Download all macro series -------------------------------------------

all_macro <- list()

# 1. FX spot rates
all_macro[["fx"]] <- download_series(
  fx_tickers, "fx_spot", "DAILY"
)

# 2. Sovereign CDS 5Y
all_macro[["cds"]] <- download_series(
  cds_tickers, "cds_5y", "DAILY"
)

# 3. 10Y Government Bond Yields
all_macro[["govt10y"]] <- download_series(
  govt10y_tickers, "govt_10y_yield", "DAILY"
)

# 4. Policy / short-term rates
all_macro[["policy"]] <- download_series(
  policy_rate_tickers, "policy_rate", "DAILY"
)

# 5. CPI (YoY)
all_macro[["cpi"]] <- download_series(
  cpi_tickers, "cpi_yoy", "MONTHLY"
)

# 6. Current Account (% GDP)
all_macro[["ca"]] <- download_series(
  current_account_tickers, "current_account_pct_gdp", "MONTHLY"
)

# ---- Combine and save ----------------------------------------------------
macro_df <- bind_rows(all_macro)

if (nrow(macro_df) == 0) {
  stop("No macro data returned at all. Check Bloomberg connection and tickers.")
}

macro_df$date <- as.Date(macro_df$date)

macro_df <- macro_df %>%
  arrange(series, country, date) %>%
  select(date, country, series, value)

write.csv(macro_df, "raw_macro_data.csv", row.names = FALSE)

# ---- Coverage summary ----------------------------------------------------
cat("\n\n=== Macro Data Coverage Summary ===\n")
cat("Total rows:", nrow(macro_df), "\n\n")

coverage <- macro_df %>%
  group_by(series) %>%
  summarise(
    countries = n_distinct(country),
    earliest  = min(date),
    latest    = max(date),
    total_obs = n(),
    .groups   = "drop"
  )

print(as.data.frame(coverage))

# Per-series, per-country coverage
cat("\n--- Detailed coverage per series × country ---\n")
detail <- macro_df %>%
  group_by(series, country) %>%
  summarise(
    from = min(date),
    to   = max(date),
    obs  = n(),
    .groups = "drop"
  )

for (s in unique(detail$series)) {
  cat(sprintf("\n  %s:\n", s))
  sub <- detail %>% filter(series == s)
  for (i in 1:nrow(sub)) {
    cat(sprintf("    %-15s  %s to %s  (%d obs)\n",
                sub$country[i], sub$from[i], sub$to[i], sub$obs[i]))
  }
}

# ---- Which countries are completely missing per series? -------------------
cat("\n--- Missing countries per series ---\n")
for (s in unique(coverage$series)) {
  present  <- unique(macro_df$country[macro_df$series == s])
  missing  <- setdiff(country_names, present)
  if (length(missing) > 0) {
    cat(sprintf("  %s: MISSING %s\n", s, paste(missing, collapse = ", ")))
  }
}

cat("\nSaved: raw_macro_data.csv\n")

# ---- Disconnect ----------------------------------------------------------
blpDisconnect(con)
cat("Done.\n")
