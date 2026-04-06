# =========================================================================
# downloadFactors.R — Download index-level fundamental data from Bloomberg
# =========================================================================
# Run this on your university Bloomberg machine.
#
# Prerequisites:
#   - Bloomberg Terminal running and logged in
#   - install.packages(c("Rblpapi", "dplyr", "tidyr"))
#   - config.R in the same directory
#
# Output:
#   raw_factor_data.csv — long-format: date, country, ticker, field, value
#
# NOTE: Some fields may not be available for all indices. The script
# handles failures gracefully and prints a coverage summary at the end.
# Check which fields returned no data — you may need to try alternative
# Bloomberg field names for your terminal version.
# =========================================================================

library(Rblpapi)
library(dplyr)
library(tidyr)

source("config.R")

# ---- Bloomberg connection ------------------------------------------------
con <- blpConnect()

# ---- Parameters ----------------------------------------------------------
start_date <- as.Date("2005-01-01")
end_date   <- Sys.Date()

tickers <- names(country_tickers)
fields  <- bloomberg_factors$field

cat("=== Index Fundamental Factor Download ===\n")
cat("Tickers:", length(tickers), "\n")
cat("Fields:", length(fields), "\n")
cat("Date range:", as.character(start_date), "to", as.character(end_date), "\n")
cat("Fields to download:\n")
for (i in seq_along(fields)) {
  cat(sprintf("  %2d. %-35s (%s)\n", i, bloomberg_factors$label[i], fields[i]))
}
cat("\n")

# ---- Download (field by field) -------------------------------------------
opt <- c("periodicitySelection" = "MONTHLY")

all_results <- list()
failed_fields <- character(0)

for (fld_idx in seq_along(fields)) {
  fld   <- fields[fld_idx]
  label <- bloomberg_factors$label[fld_idx]
  cat(sprintf("  [%2d/%d] %-35s ... ", fld_idx, length(fields), label))

  tryCatch({
    raw <- bdh(
      securities = tickers,
      fields     = fld,
      start.date = start_date,
      end.date   = end_date,
      options    = opt
    )

    n_rows <- 0
    for (tkr in names(raw)) {
      df <- raw[[tkr]]
      if (is.null(df) || nrow(df) == 0) next

      colnames(df) <- c("date", "value")
      df$ticker <- tkr
      df$field  <- fld
      all_results[[paste(tkr, fld, sep = "|")]] <- df
      n_rows <- n_rows + nrow(df)
    }
    cat("OK (", n_rows, "rows )\n")

  }, error = function(e) {
    cat("FAILED:", conditionMessage(e), "\n")
    failed_fields <<- c(failed_fields, fld)
  })
}

# ---- Combine and save ----------------------------------------------------
if (length(all_results) == 0) {
  stop("No data returned at all. Check Bloomberg connection.")
}

result_df <- bind_rows(all_results)
result_df$country <- country_tickers[result_df$ticker]
result_df$date    <- as.Date(result_df$date)

result_df <- result_df %>%
  arrange(date, country, field) %>%
  select(date, country, ticker, field, value)

write.csv(result_df, "raw_factor_data.csv", row.names = FALSE)

# ---- Coverage summary ----------------------------------------------------
cat("\n=== Coverage Summary ===\n")
cat("Total rows:", nrow(result_df), "\n")
cat("Date range:", as.character(min(result_df$date)), "to",
    as.character(max(result_df$date)), "\n\n")

coverage <- result_df %>%
  group_by(field) %>%
  summarise(
    countries  = n_distinct(country),
    earliest   = min(date),
    latest     = max(date),
    total_obs  = n(),
    n_na       = sum(is.na(value)),
    .groups    = "drop"
  )

print(as.data.frame(coverage))

if (length(failed_fields) > 0) {
  cat("\nFIELDS THAT FAILED (not available for these index tickers):\n")
  cat("  ", paste(failed_fields, collapse = ", "), "\n")
  cat("  These are normal — not all fields exist for all indices.\n")
  cat("  The preparation script will skip missing factors.\n")
}

cat("\nSaved: raw_factor_data.csv\n")

# ---- Disconnect ----------------------------------------------------------
blpDisconnect(con)
cat("Done.\n")
