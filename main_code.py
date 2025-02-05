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

# Function to fetch data
def fetch_data(market, symbol, start_date, end_date):
    try:
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
        
        if data.empty:
            st.error("No data found. Please check the symbol or date range.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to calculate Value-at-Risk (VaR)
def calculate_var(data, confidence=0.95):
    if data.shape[1] > 1:  # Check for multi-level columns
        close_column = data.iloc[:, data.columns.get_loc('close')]
        returns = close_column.pct_change().dropna()
        var = np.percentile(returns, (1 - confidence) * 100)
        return var
    return None

# Function to calculate Maximum Drawdown
def calculate_max_drawdown(data):
    if data.shape[1] > 1:  # Check for multi-level columns
        close_column = data.iloc[:, data.columns.get_loc('close')]
        cumulative_returns = (1 + close_column.pct_change()).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()
    return None

# Main app
st.header(f"{market} - {strategy} Strategy")

# Input for symbol and date range
symbol = st.text_input("Enter Symbol (e.g., TCS, BTC/USDT, EURUSD=X)", "TCS")
start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))

# Fetch data
data = fetch_data(market, symbol, start_date, end_date)

if data is not None and not data.empty:
    st.write(f"Data for {symbol}")
    st.write(data)
    
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    if data.shape[1] > 1:  # Handle multi-level columns
        close_column = data.iloc[:, data.columns.get_loc('close')]
        fig = px.line(data, x=data.index, y=close_column, title=f"{symbol} Price Chart")
        st.plotly_chart(fig)
    else:
        st.error("The 'close' column is missing. Please check the symbol or market selection.")
    
    # Strategy Implementation
    if strategy == "Trend Following" and data.shape[1] > 1:
        close_column = data.iloc[:, data.columns.get_loc('close')]
        data["SMA"] = close_column.rolling(window=20).mean()
        fig = px.line(data, x=data.index, y=["close", "SMA"], title="Trend Following - SMA")
        st.plotly_chart(fig)
    elif strategy == "Mean Reversion" and data.shape[1] > 1:
        close_column = data.iloc[:, data.columns.get_loc('close')]
        data["SMA"] = close_column.rolling(window=20).mean()
        data["Deviation"] = close_column - data["SMA"]
        fig = px.line(data, x=data.index, y=["Deviation"], title="Mean Reversion - Deviation from SMA")
        st.plotly_chart(fig)
    elif strategy == "Range Trading" and all(col in data.columns for col in ['high', 'low', 'close']):
        high_column = data.iloc[:, data.columns.get_loc('high')]
        low_column = data.iloc[:, data.columns.get_loc('low')]
        close_column = data.iloc[:, data.columns.get_loc('close')]
        data["Rolling High"] = high_column.rolling(window=20).max()
        data["Rolling Low"] = low_column.rolling(window=20).min()
        fig = px.line(data, x=data.index, y=["close", "Rolling High", "Rolling Low"], title="Range Trading")
        st.plotly_chart(fig)
    elif strategy == "Scalping" and data.shape[1] > 1:
        close_column = data.iloc[:, data.columns.get_loc('close')]
        data["EMA"] = close_column.ewm(span=9, adjust=False).mean()
        fig = px.line(data, x=data.index, y=["close", "EMA"], title="Scalping - EMA")
        st.plotly_chart(fig)
    elif strategy == "Market Making" and all(col in data.columns for col in ['open', 'close']):
        open_column = data.iloc[:, data.columns.get_loc('open')]
        close_column = data.iloc[:, data.columns.get_loc('close')]
        data["Spread"] = close_column - open_column
        fig = px.bar(data, x=data.index, y="Spread", title="Market Making - Spread")
        st.plotly_chart(fig)
    
    # Risk Management Metrics
    st.header("Risk Management Metrics")
    st.write(f"Stop-Loss: {stop_loss}%")
    st.write(f"Position Size: {position_size}% of Capital")
    st.write(f"Risk-Reward Ratio: {risk_reward_ratio}")
    
    var = calculate_var(data)
    drawdown = calculate_max_drawdown(data)
    
    st.write(f"Value-at-Risk (VaR 95%): {var:.5f}" if var else "VaR could not be calculated.")
    st.write(f"Maximum Drawdown: {drawdown:.5f}" if drawdown else "Max Drawdown could not be calculated.")
    
    # Backtesting Placeholder
    st.header("Backtesting (Coming Soon)")
    st.info("Backtesting functionality will be added in future updates.")
else:
    st.error("No valid data available. Please check your inputs.")
