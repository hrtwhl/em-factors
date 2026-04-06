# -------------------------------------------------------------------------
# Bloomberg Index Downloader (EUR Conversion)
# -------------------------------------------------------------------------

# 1. Load necessary libraries
# If you don't have Rblpapi, install it: install.packages("Rblpapi")
library(Rblpapi)
library(dplyr) # For easier data manipulation

# 2. Connect to Bloomberg
# Note: A Bloomberg Terminal must be running and logged in on this machine.
con <- blpConnect()

# 3. Define the Tickers directly
# We use the correct Bloomberg format (Space instead of Dot) immediately.
tickers <- c(
  "MIMUEMRN Index", # IS3N GY Equity - iShares Core MSCI Emerging Markets IMI UCITS ETF (Acc)
  "M1CNA Index", # ICGA GY Equity – iShares MSCI China UCITS ETF USD (Acc)
  "MXIN Index", # QDV5 GY Equity – iShares MSCI India UCITS ETF USD (Acc)
  "MXKR Index", # CEBJ GR Equity – iShares MSCI Korea UCITS ETF (Acc)
  "TAMSCI Index", # XMTW GY Equity – Xtrackers MSCI Taiwan UCITS ETF
  "MXID Index", # H4ZT GY Equity – HSBC MSCI Indonesia UCITS ETF USD
  "MXMY Index", # XCS3 GT Equity – Xtrackers MSCI Malaysia UCITS ETF
  "MXTH Index", # XCS4 GY Equity – Xtrackers MSCI Thailand UCITS ETF
  "MXPH Index", # XPQP GY Equity – Xtrackers MSCI Philippines UCITS ETF
  "MXBR Index", # XMBR GY Equity – Xtrackers MSCI Brazil UCITS ETF
  "MXMX Index", # D5BI GY Equity – Xtrackers MSCI Mexico UCITS ETF
  "MXLA Index", # XMLA GT Equity - Xtraclers MSCI EM Latin America ESG Swap UCITS ETF
  "MXZA Index", # IBC4 GY Equity - iShares MSCI South Africa UCITS ETF
  "MXTR Index", # LTUR GY Equity - Amundi MSCI Turkey UCITS ETF
  "MXGR Index", # LYXGRE GY Equity - Amundi MSCi Greece UCITS ETF
  "MXPL Index", # IBCJ GY Equity - iShares MSCI Poland UCITS ETF
  "MXSA Index", # IUSS GY Equity - iShares MSCI Saudi Arabia Capped UCITS ETF
  "JPEICORE Index", # IUS7 GY Equity - iShares J.P. Morgan USD Emerging Markets Bond UCITS ETF (Dist)
  "GBIE1001 Index", # IUSP GY Equity - iShares J.P. Morgan EM Local Government Bond UCITS ETF
  #"GBIE1001 Index", # ISOQ GY Equity - iShares J.P. Morgan EM Local Government Bond UCITS ETF
  "BCEX6T Index",   # IBCA GY Equity - iShares Euro Government Bond 1-3yr UCITS ETF (Dist)
  "BCEX4T Index",   # IBCM GY Equity - iShares Euro Government Bond 7-10yr UCITS ETF EUR (Dist)
  #"IDCOT1 Index",  # IUSU GY Equity - iShares $ Treasury Bond 1-3yr UCITS ETF
  "IDCOT7 Index",   # IUSM GY Equity - iShares USD Treasury Bond 7-10yr UCITS ETF (Dist)
  "BEIG1T Index",   # IBCI GY Equity - iShares Euro Inflation Linked Government Bond UCITS ETF
  "LECPTREU Index", # EUN5 GY Equity - iShares Core Euro Corporate Bond UCITS ETF (Dist)
  "IBOXXMJA Index"  # EUNW GY Equity - iShares EUR High Yield Corporate Bond UCITS ETF EUR (Dist)
)

print(paste("Downloading", length(tickers), "indices in EUR..."))

# 4. Define Download Parameters
start_date <- as.Date("2000-01-01") 
end_date   <- Sys.Date()
field      <- "PX_LAST"             

# 5. Fetch Data with Currency Conversion
# CORRECT USAGE:
# "currency" is an OPTION, not an override. 
# We remove the 'overrides' argument entirely to prevent API conflicts.
opt <- c("currency" = "EUR", "periodicitySelection" = "DAILY")

data_list <- bdh(
  securities = tickers, 
  fields = field, 
  start.date = start_date, 
  end.date = end_date,
  options = opt
)

# 6. Post-Processing: Convert List to Single Data Frame
# Get the actual names returned by Bloomberg (safer than assuming order)
returned_tickers <- names(data_list)

if (length(returned_tickers) == 0) {
  stop("No data was returned from Bloomberg. Check connection.")
}

# Initialize with the first dataframe
final_df <- data_list[[1]]
colnames(final_df) <- c("Dates", returned_tickers[1]) 

# Loop through the rest and merge by Date
if (length(data_list) > 1) {
  for (i in 2:length(data_list)) {
    temp_df <- data_list[[i]]
    
    # Only process if we actually got data for this ticker
    if (!is.null(temp_df) && nrow(temp_df) > 0) {
      # Rename columns: keep Date, use the ticker name
      colnames(temp_df) <- c("Dates", returned_tickers[i])
      
      # Merge (full_join ensures we don't lose days if calendars differ)
      final_df <- merge(final_df, temp_df, by = "Dates", all = TRUE)
    }
  }
}

# Optional: Save to CSV
write.csv(final_df, "EM_Indices_EUR.csv", row.names = FALSE)

print("Download complete.")

summary(final_df)


tail(final_df, 2)
