
import yfinance as yf
import pandas as pd

def load_data(tickers, start, end):
    try:
        print(f"Loading data for: {tickers} from {start} to {end}")
        
        # This may not work in all versions; if it fails, comment it out
        data = yf.download(
            tickers,
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        # Check if we got anything useful
        if data.empty:
            raise ValueError("❌ No data returned from yfinance.")

        print("✅ Data load successful.")
        return data

    except Exception as e:
        print(f"💥 Error during data loading: {e}")
        raise  # Let Streamlit show the exception in debug
