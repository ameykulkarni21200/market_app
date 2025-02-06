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
confidence_interval = st.sidebar.number_input("VaR Confidence Interval (%)", min_value=90, max_value=99, value=95)

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

# Calculate Value at Risk (VaR) using Historical Simulation
def calculate_var(data, confidence_interval=95):
    # Calculate daily returns
    returns = data.pct_change().dropna()

    # Calculate VaR at the given confidence level
    var_percentile = np.percentile(returns, (100 - confidence_interval))
    var = -var_percentile * 100  # Convert to percentage
    return var

# Calculate Rolling Volatility
def calculate_volatility(data, window=20):
    returns = data.pct_change().dropna()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized volatility
    return volatility

# Calculate Sharpe Ratio
def calculate_sharpe_ratio(data, risk_free_rate=0.02):
    returns = data.pct_change().dropna()
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualized Sharpe Ratio
    return sharpe_ratio

# Main App
st.header(f"{market} - {strategy} Strategy")

symbol = st.text_input("Enter Symbol", "TCS")
start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))

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

    # Trend Following Strategy
    if strategy == "Trend Following":
        data["SMA"] = data.iloc[:, 0].rolling(window=20).mean()
        fig = px.line(data, x=data.index, y=[data.columns[0], "SMA"], title="Trend Following - Simple Moving Average")
        st.plotly_chart(fig)

    # Mean Reversion Strategy
    elif strategy == "Mean Reversion":
        data["SMA"] = data.iloc[:, 0].rolling(window=20).mean()
        data["Deviation"] = data.iloc[:, 0] - data["SMA"]
        fig = px.line(data, x=data.index, y="Deviation", title="Mean Reversion - Deviation from SMA")
        st.plotly_chart(fig)

    # Range Trading Strategy
    elif strategy == "Range Trading":
        high_column = data.iloc[:, 1]  # Access 'high' using index
        low_column = data.iloc[:, 2]  # Access 'low' using index
        data["Rolling High"] = high_column.rolling(window=20).max()
        data["Rolling Low"] = low_column.rolling(window=20).min()
        fig = px.line(data, x=data.index, y=[data.columns[0], "Rolling High", "Rolling Low"], title="Range Trading - Rolling Highs and Lows")
        st.plotly_chart(fig)

    # Scalping Strategy
    elif strategy == "Scalping":
        data["EMA"] = data.iloc[:, 0].ewm(span=9, adjust=False).mean()
        fig = px.line(data, x=data.index, y=[data.columns[0], "EMA"], title="Scalping - EMA")
        st.plotly_chart(fig)

    # Market Making Strategy
    elif strategy == "Market Making":
        open_column = data.iloc[:, 1]  # Access 'open' using index
        close_column = data.iloc[:, 0]  # Access 'close' using index
        data["Spread"] = close_column - open_column
        fig = px.bar(data, x=data.index, y="Spread", title="Market Making - Spread")
        st.plotly_chart(fig)

    # Risk Management Display
    st.header("Risk Management Metrics")
    st.write(f"Stop-Loss: {stop_loss}%")
    st.write(f"Position Size: {position_size}% of Capital")
    st.write(f"Risk-Reward Ratio: {risk_reward_ratio}")

    # Calculate Value at Risk (VaR)
    var = calculate_var(data.iloc[:, 0], confidence_interval)
    st.write(f"Value at Risk (VaR) at {confidence_interval}% confidence level: {var:.2f}%")

    # Calculate Rolling Volatility
    volatility = calculate_volatility(data.iloc[:, 0])
    fig = px.line(volatility, x=volatility.index, y=volatility, title="Rolling Volatility (Annualized)")
    st.plotly_chart(fig)

    # Calculate Sharpe Ratio
    sharpe_ratio = calculate_sharpe_ratio(data.iloc[:, 0])
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

else:
    st.error("No data found for the given symbol and market. Please check your inputs.")
