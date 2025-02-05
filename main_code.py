import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import plotly.express as px

st.title("Trading Strategy & Risk Management App")

# Sidebar Inputs
st.sidebar.header("Market Selection")
market = st.sidebar.selectbox("Choose Market", ["Indian Market", "Crypto Market", "Forex Market"])

st.sidebar.header("Trading Strategy")
strategy = st.sidebar.selectbox(
    "Choose Strategy", 
    ["Trend Following", "Mean Reversion", "Range Trading", "Scalping", "Market Making"]
)

st.sidebar.header("Risk Management")
stop_loss = st.sidebar.number_input("Stop-Loss (%)", min_value=0.1, max_value=10.0, value=2.0)
position_size = st.sidebar.number_input("Position Size (% of Capital)", min_value=1, max_value=100, value=10)
risk_reward_ratio = st.sidebar.number_input("Risk-Reward Ratio", min_value=1.0, max_value=5.0, value=2.0)

# Function to fetch data
def fetch_data(market, symbol, start_date, end_date):
    if market == "Indian Market":
        data = yf.download(f"{symbol}.NS", start=start_date, end=end_date)
    elif market == "Crypto Market":
        exchange = ccxt.binance()
        data = exchange.fetch_ohlcv(symbol, timeframe='1d', since=exchange.parse8601(start_date))
        data = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
    elif market == "Forex Market":
        data = yf.download(symbol, start=start_date, end=end_date)

    # Handle multi-index columns issue
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]

    return data

# Main App
st.header(f"{market} - {strategy} Strategy")

symbol = st.text_input("Enter Symbol", "TCS")
start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))

data = fetch_data(market, symbol, start_date, end_date)

if not data.empty:
    st.write(f"Data for {symbol}")

    # Flatten MultiIndex Columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col) for col in data.columns]

    st.write(data)  # Display the DataFrame

    # Use index-based column selection to avoid missing column errors
    fig = px.line(data, x=data.index, y=data.iloc[:, 0], title=f"{symbol} Price Chart")
    st.plotly_chart(fig)

    if strategy == "Trend Following":
        data["SMA"] = data.iloc[:, 0].rolling(window=20).mean()
        fig = px.line(data, x=data.index, y=[data.iloc[:, 0], "SMA"], title="Trend Following - Simple Moving Average")
        st.plotly_chart(fig)

    elif strategy == "Mean Reversion":
        data["SMA"] = data.iloc[:, 0].rolling(window=20).mean()
        data["Deviation"] = data.iloc[:, 0] - data["SMA"]
        fig = px.line(data, x=data.index, y="Deviation", title="Mean Reversion - Deviation from SMA")
        st.plotly_chart(fig)

    elif strategy == "Range Trading":
        data["Rolling High"] = data.iloc[:, 1].rolling(window=20).max()
        data["Rolling Low"] = data.iloc[:, 2].rolling(window=20).min()
        fig = px.line(data, x=data.index, y=[data.iloc[:, 0], "Rolling High", "Rolling Low"], title="Range Trading - Rolling Highs and Lows")
        st.plotly_chart(fig)

    # Risk Management Display
    st.header("Risk Management Metrics")
    st.write(f"Stop-Loss: {stop_loss}%")
    st.write(f"Position Size: {position_size}% of Capital")
    st.write(f"Risk-Reward Ratio: {risk_reward_ratio}")

else:
    st.error("No data found for the given symbol and market. Please check your inputs.")
