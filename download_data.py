import pandas as pd
import yfinance as yf
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    csv_file = "ind_nifty500list.csv"
    output_file = "nifty500_ohlcv.parquet"
    
    if not os.path.exists(csv_file):
        logging.error(f"File not found: {csv_file}")
        return

    # Read the Nifty 500 list
    df = pd.read_csv(csv_file)
    
    # Append .NS to match Yahoo Finance NSE ticker format
    symbols = [f"{sym}.NS" for sym in df['Symbol'].dropna().unique()]
    logging.info(f"Loaded {len(symbols)} symbols from {csv_file}")
    
    # Bulk download data (using for robust ML training depth)
    logging.info("Downloading historical OHLCV data...")
    data = yf.download(symbols, period="max", group_by='ticker', threads=True)
    
    # Parse the MultiIndex dataframe returned by yfinance
    all_data = []
    for sym in symbols:
        if sym in data:
            df_sym = data[sym].copy()
            # Drop rows where all OHLCV values are NaN
            df_sym.dropna(how='all', subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            if not df_sym.empty:
                df_sym['Ticker'] = sym
                all_data.append(df_sym)
                
    if not all_data:
        logging.error("No data successfully downloaded.")
        return

    # Combine and format final DataFrame
    final_df = pd.concat(all_data)
    final_df.reset_index(inplace=True)
    
    # Ensure Date column is correctly named and formatted
    if 'Date' in final_df.columns:
        final_df['Date'] = pd.to_datetime(final_df['Date'])
    
    # Save to Parquet format (optimized for ML reading)
    final_df.to_parquet(output_file, index=False, engine='pyarrow')
    logging.info(f"Successfully saved consolidated data to {output_file}")

if __name__ == "__main__":
    main()
