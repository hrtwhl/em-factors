# =========================================================================
# downloadFactors.R — Download fundamental factor data from Bloomberg
# =========================================================================
# Prerequisites:
#   - Bloomberg Terminal running and logged in
#   - Rblpapi, dplyr, tidyr installed
#   - config.R in the same directory
#
# Output:
#   raw_factor_data.csv — long-format: date, ticker, field, value
#   One row per ticker × field × month-end date.
#
# Run this ONCE (or periodically to update), then run prepareForBacktest.R
# to produce the CSVs the Python framework expects.
# =========================================================================

library(Rblpapi)
library(dplyr)
library(tidyr)

# ---- Load universe and factor definitions --------------------------------
source("config.R")

# ---- Bloomberg connection ------------------------------------------------
con <- blpConnect()

# ---- Parameters ----------------------------------------------------------
start_date <- as.Date("2005-01-01")
end_date   <- Sys.Date()

# Tickers to query (the equity country indices only)
tickers <- names(country_tickers)

# Fields to download
fields <- bloomberg_factors$field

cat("Downloading", length(fields), "fields for", length(tickers),
    "country indices...\n")
cat("Fields:", paste(fields, collapse = ", "), "\n")
cat("Date range:", as.character(start_date), "to", as.character(end_date), "\n\n")

# ---- Download ------------------------------------------------------------
# Bloomberg bdh() returns a named list of data frames when given multiple
# securities. We request MONTHLY periodicity so we get month-end snapshots.

opt <- c("periodicitySelection" = "MONTHLY")

# We download each field separately to handle cases where some fields
# are unavailable for certain indices (e.g. BEST_ fields may have
# shorter histories for frontier-ish markets like Saudi Arabia).

all_results <- list()

for (fld in fields) {
  cat("  Downloading:", fld, "... ")

  tryCatch({
    raw <- bdh(
      securities = tickers,
      fields     = fld,
      start.date = start_date,
      end.date   = end_date,
      options    = opt
    )

    # raw is a named list: one data.frame per ticker
    # Each data.frame has columns: date, <field_name>
    for (tkr in names(raw)) {
      df <- raw[[tkr]]
      if (is.null(df) || nrow(df) == 0) next

      colnames(df) <- c("date", "value")
      df$ticker <- tkr
      df$field  <- fld
      all_results[[paste(tkr, fld, sep = "|")]] <- df
    }
    cat("OK\n")

  }, error = function(e) {
    cat("FAILED:", conditionMessage(e), "\n")
  })
}

# ---- Combine into a single long-format data frame ------------------------
result_df <- bind_rows(all_results)

# Add country name
result_df$country <- country_tickers[result_df$ticker]

# Ensure date is Date type
result_df$date <- as.Date(result_df$date)

# Sort
result_df <- result_df %>%
  arrange(date, country, field) %>%
  select(date, country, ticker, field, value)

cat("\nTotal rows:", nrow(result_df), "\n")
cat("Date range:", as.character(min(result_df$date)), "to",
    as.character(max(result_df$date)), "\n")
cat("Countries with data:", length(unique(result_df$country)), "\n")

# ---- Quick coverage summary ----------------------------------------------
coverage <- result_df %>%
  group_by(field, country) %>%
  summarise(
    first_date = min(date),
    last_date  = max(date),
    n_obs      = n(),
    n_na       = sum(is.na(value)),
    .groups    = "drop"
  )

cat("\n--- Coverage summary (first/last date per field × country) ---\n")
print(as.data.frame(
  coverage %>%
    group_by(field) %>%
    summarise(
      countries  = n(),
      earliest   = min(first_date),
      latest     = max(last_date),
      median_obs = median(n_obs),
      .groups    = "drop"
    )
))

# ---- Save ----------------------------------------------------------------
write.csv(result_df, "raw_factor_data.csv", row.names = FALSE)
cat("\nSaved: raw_factor_data.csv\n")

# ---- Disconnect ----------------------------------------------------------
blpDisconnect(con)
cat("Done.\n")
