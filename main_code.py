import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

st.title("Trading Strategy, Risk Management & Market Prediction")

# Sidebar Inputs
st.sidebar.header("Market Selection")
market = st.sidebar.selectbox("Choose Market", ["Indian Market", "Crypto Market", "Forex Market"])

st.sidebar.header("Trading Strategy")
strategy = st.sidebar.selectbox("Choose Strategy", ["Trend Following", "Mean Reversion", "Range Trading", "Scalping", "Market Making"])

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

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]

    return data

# Risk Management Functions
def calculate_var(data, confidence_interval=95):
    returns = data.pct_change().dropna()
    var_percentile = np.percentile(returns, (100 - confidence_interval))
    return -var_percentile * 100

def calculate_cvar(data, confidence_interval=95):
    returns = data.pct_change().dropna()
    var = np.percentile(returns, (100 - confidence_interval))
    cvar = returns[returns <= var].mean() * -100
    return cvar

def calculate_max_drawdown(data):
    cumulative_returns = (1 + data.pct_change()).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min() * 100

def calculate_beta(data, benchmark):
    asset_returns = data.pct_change().dropna()
    benchmark_returns = benchmark.pct_change().dropna()
    covariance = np.cov(asset_returns, benchmark_returns)[0, 1]
    variance = np.var(benchmark_returns)
    beta = covariance / variance
    return beta

def calculate_sortino_ratio(data, risk_free_rate=0.02):
    returns = data.pct_change().dropna()
    excess_returns = returns - risk_free_rate / 252
    downside_deviation = returns[returns < 0].std() * np.sqrt(252)
    return excess_returns.mean() / downside_deviation

# ARIMA Forecasting
def arima_forecast(data, steps=10):
    model = ARIMA(data, order=(5,1,0))  # ARIMA(5,1,0) is a simple configuration
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Main App
st.header(f"{market} - {strategy} Strategy")

symbol = st.text_input("Enter Symbol", "TCS")
start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-02-05"))

data = fetch_data(market, symbol, start_date, end_date)

if not data.empty:
    st.write(f"Data for {symbol}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col) for col in data.columns]

    st.write(data)

    fig = px.line(data, x=data.index, y=data.iloc[:, 0], title=f"{symbol} Price Chart")
    st.plotly_chart(fig)

    # Risk Management Metrics
    st.header("Risk Management Metrics")
    var = calculate_var(data.iloc[:, 0], confidence_interval)
    cvar = calculate_cvar(data.iloc[:, 0], confidence_interval)
    max_drawdown = calculate_max_drawdown(data.iloc[:, 0])
    sortino_ratio = calculate_sortino_ratio(data.iloc[:, 0])

    st.write(f"Value at Risk (VaR): {var:.2f}%")
    st.write(f"Conditional VaR (Expected Shortfall): {cvar:.2f}%")
    st.write(f"Max Drawdown: {max_drawdown:.2f}%")
    st.write(f"Sortino Ratio: {sortino_ratio:.2f}")

    # Market Prediction
    st.header("Market Prediction")

    # ARIMA Forecast
    st.subheader("ARIMA Forecast (Next 10 Days)")
    forecast = arima_forecast(data.iloc[:, 0])
    forecast_dates = pd.date_range(start=data.index[-1], periods=11, freq='D')[1:]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Price': forecast})
    st.write(forecast_df)

    # Plot ARIMA Forecast
    fig, ax = plt.subplots()
    ax.plot(data.index, data.iloc[:, 0], label="Actual Price")
    ax.plot(forecast_dates, forecast, label="ARIMA Forecast", linestyle="dashed", color="red")
    ax.set_title("ARIMA Price Forecast")
    ax.legend()
    st.pyplot(fig)

else:
    st.error("No data found for the given symbol and market. Please check your inputs.")
