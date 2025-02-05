import streamlit as st
import pandas as pd
import yfinance as yf
import ccxt
import plotly.express as px

# Title
st.title("Trading Strategy & Risk Management App")

# Sidebar for market selection
market = st.sidebar.selectbox("Choose Market", ["Indian Market", "Crypto Market", "Forex Market"])

# Sidebar for strategy selection
strategy = st.sidebar.selectbox("Choose Strategy", ["Trend Following", "Mean Reversion", "Range Trading"])

# User Inputs
symbol = st.text_input("Enter Symbol", "TCS")
start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))

# Fetch Market Data Function
def fetch_data(market, symbol, start_date, end_date):
    try:
        if market == "Indian Market":
            data = yf.download(f"{symbol}.NS", start=start_date, end=end_date)
        
        elif market == "Crypto Market":
            exchange = ccxt.binance()
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', since=exchange.parse8601(str(start_date)))
            
            if not ohlcv:
                st.error("No crypto data available. Check the symbol.")
                return None
            
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)

        elif market == "Forex Market":
            data = yf.download(symbol, start=start_date, end=end_date)

        # Debugging: Print data columns
        if data is not None and not data.empty:
            st.write("Columns in the dataset:", list(data.columns))
        else:
            st.error("No data found. Please check the symbol or date range.")
            return None

        # Rename 'Close' to 'close' for consistency
        if "Close" in data.columns:
            data.rename(columns={'Close': 'close'}, inplace=True)

        return data

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Fetch data
data = fetch_data(market, symbol, start_date, end_date)

# Display data
if data is not None and not data.empty:
    st.write(f"Data for {symbol}")
    st.write(data)

    # Check if 'close' column exists before plotting
    if 'close' in data.columns:
        fig = px.line(data, x=data.index, y='close', title=f"{symbol} Price Chart")
        st.plotly_chart(fig)
    else:
        st.error("The 'close' column is missing in the retrieved data. Check symbol/market selection.")
else:
    st.error("Failed to retrieve valid data. Please check your inputs.")
