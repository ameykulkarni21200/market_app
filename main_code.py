import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import plotly.express as px

# Title of the app
st.title("Trading Strategy & Risk Management App")

# Sidebar for market selection
st.sidebar.header("Market Selection")
market = st.sidebar.selectbox("Choose Market", ["Indian Market", "Crypto Market", "Forex Market"])

# Sidebar for strategy selection
st.sidebar.header("Trading Strategy")
strategy = st.sidebar.selectbox(
    "Choose Strategy",
    ["Trend Following", "Mean Reversion", "Range Trading", "Scalping", "Market Making"]
)

# Sidebar for risk management
st.sidebar.header("Risk Management")
stop_loss = st.sidebar.number_input("Stop-Loss (%)", min_value=0.1, max_value=10.0, value=2.0)
position_size = st.sidebar.number_input("Position Size (% of Capital)", min_value=1, max_value=100, value=10)
risk_reward_ratio = st.sidebar.number_input("Risk-Reward Ratio", min_value=1.0, max_value=5.0, value=2.0)

# Function to fetch data based on market
def fetch_data(market, symbol, start_date, end_date):
    if market == "Indian Market":
        data = yf.download(f"{symbol}.NS", start=start_date, end=end_date)
    elif market == "Crypto Market":
        exchange = ccxt.binance()
        data = exchange.fetch_ohlcv(symbol, timeframe='1d', since=exchange.parse8601(start_date))
        data = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    elif market == "Forex Market":
        data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Main app
st.header(f"{market} - {strategy} Strategy")

# Input for symbol and date range
symbol = st.text_input("Enter Symbol (e.g., TCS for Indian Market, BTC/USDT for Crypto, EURUSD=X for Forex)", "TCS")
start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))

# Fetch data
data = fetch_data(market, symbol, start_date, end_date)

# Display data
if not data.empty:
    st.write(f"Data for {symbol}")
    st.write(data)

    # Plot data
    fig = px.line(data, x=data.index, y='close', title=f"{symbol} Price Chart")
    st.plotly_chart(fig)

    # Apply strategy
    if strategy == "Trend Following":
        data['SMA'] = data['close'].rolling(window=20).mean()
        fig = px.line(data, x=data.index, y=['close', 'SMA'], title="Trend Following - Simple Moving Average")
        st.plotly_chart(fig)

    elif strategy == "Mean Reversion":
        data['SMA'] = data['close'].rolling(window=20).mean()
        data['Deviation'] = data['close'] - data['SMA']
        fig = px.line(data, x=data.index, y=['Deviation'], title="Mean Reversion - Deviation from SMA")
        st.plotly_chart(fig)

    elif strategy == "Range Trading":
        data['Rolling High'] = data['high'].rolling(window=20).max()
        data['Rolling Low'] = data['low'].rolling(window=20).min()
        fig = px.line(data, x=data.index, y=['close', 'Rolling High', 'Rolling Low'], title="Range Trading - Rolling Highs and Lows")
        st.plotly_chart(fig)

    # Risk management calculations
    st.header("Risk Management Metrics")
    st.write(f"Stop-Loss: {stop_loss}%")
    st.write(f"Position Size: {position_size}% of Capital")
    st.write(f"Risk-Reward Ratio: {risk_reward_ratio}")

else:
    st.error("No data found for the given symbol and market. Please check your inputs.")
