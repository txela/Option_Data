import yfinance as yf
import pandas as pd

def fetch_option_data(ticker_symbol):
    try:
        # Fetch the ticker
        stock = yf.Ticker(ticker_symbol)

        # Get the available expiry dates
        expiry_dates = stock.options
        if not expiry_dates:
            print("❌ No options data available for this ticker.")
            return

        print(f"Available expiry dates for {ticker_symbol}:")
        for date in expiry_dates:
            print(f" - {date}")

        # Prompt the user to select an expiry date
        selected_date = input("Enter the expiry date (YYYY-MM-DD) to fetch options data: ")

        if selected_date not in expiry_dates:
            print("❌ Invalid expiry date selected.")
            return

        # Fetch option chain for the selected expiry date
        option_chain = stock.option_chain(selected_date)

        # Extract call and put data
        calls = option_chain.calls
        puts = option_chain.puts

        # Display the data
        print("\n📊 Calls Data:")
        print(calls.head())

        print("\n📊 Puts Data:")
        print(puts.head())

        # Save to Excel
        save_to_excel = input("Do you want to save the options data to an Excel file? (yes/no): ").strip().lower()
        if save_to_excel in ['yes', 'y']:
            file_name = f"{ticker_symbol}_options_{selected_date}.xlsx"
            with pd.ExcelWriter(file_name) as writer:
                calls.to_excel(writer, sheet_name="Calls")
                puts.to_excel(writer, sheet_name="Puts")
            print(f"✅ Options data saved to {file_name}")
        else:
            print("❌ Data not saved.")

    except Exception as e:
        print(f"❌ Error fetching options data: {e}")

# Run the script
if __name__ == "__main__":
    ticker = input("Enter the stock ticker symbol (e.g., AAPL, TSLA): ").upper()
    fetch_option_data(ticker)
