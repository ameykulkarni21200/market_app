import streamlit as st  
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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

def calculate_calmar_ratio(data, risk_free_rate=0.02):
    annual_return = data.pct_change().mean() * 252
    max_drawdown = calculate_max_drawdown(data)
    return annual_return / abs(max_drawdown)

def calculate_omega_ratio(data, threshold=0):
    returns = data.pct_change().dropna()
    gain = returns[returns > threshold].sum()
    loss = abs(returns[returns < threshold].sum())
    return gain / loss if loss != 0 else np.inf

def calculate_sortino_ratio(data, risk_free_rate=0.02):
    returns = data.pct_change().dropna()
    excess_returns = returns - risk_free_rate / 252
    downside_deviation = returns[returns < 0].std() * np.sqrt(252)
    return excess_returns.mean() / downside_deviation

def calculate_sharpe_ratio(data, risk_free_rate=0.02):
    returns = data.pct_change().dropna()
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualized Sharpe Ratio
    return sharpe_ratio

# ARIMA Forecasting
def arima_forecast(data, steps=10):
    model = ARIMA(data, order=(5,1,0))  # ARIMA(5,1,0) is a simple configuration
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Linear Regression Forecasting
def linear_regression_forecast(data, steps=10):
    data['Date'] = data.index
    data['Date'] = pd.to_datetime(data['Date']).map(pd.Timestamp.timestamp)
    X = data[['Date']]
    y = data['Close']
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model = LinearRegression()
    model.fit(X_train, y_train)
   
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=steps)
    future_dates_timestamp = future_dates.map(pd.Timestamp.timestamp).values.reshape(-1, 1)
    forecast = model.predict(future_dates_timestamp)
   
    return future_dates, forecast

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

    fig = px.line(data, x=data.index, y=data['Close'], title=f"{symbol} Price Chart")
    st.plotly_chart(fig)

    # Risk Management Metrics
    st.header("Risk Management Metrics")
    var = calculate_var(data['Close'], confidence_interval)
    cvar = calculate_cvar(data['Close'], confidence_interval)
    max_drawdown = calculate_max_drawdown(data['Close'])
    calmar_ratio = calculate_calmar_ratio(data['Close'])
    omega_ratio = calculate_omega_ratio(data['Close'])
    sortino_ratio = calculate_sortino_ratio(data['Close'])
    sharpe_ratio = calculate_sharpe_ratio(data['Close'])

    st.write(f"Value at Risk (VaR): {var:.2f}%")
    st.write(f"Conditional VaR (Expected Shortfall): {cvar:.2f}%")
    st.write(f"Max Drawdown: {max_drawdown:.2f}%")
    st.write(f"Calmar Ratio: {calmar_ratio:.2f}")
    st.write(f"Omega Ratio: {omega_ratio:.2f}")
    st.write(f"Sortino Ratio: {sortino_ratio:.2f}")
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # Market Prediction
    st.header("Market Prediction")

    # ARIMA Forecast
    st.subheader("ARIMA Forecast (Next 10 Days)")
    forecast_arima = arima_forecast(data['Close'], steps=10)
    forecast_dates_arima = pd.date_range(start=data.index[-1], periods=11, freq='D')[1:]
    forecast_df_arima = pd.DataFrame({'Date': forecast_dates_arima, 'Predicted Price': forecast_arima})
    st.write(forecast_df_arima)

    # Plot ARIMA Forecast
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label="Actual Price")
    ax.plot(forecast_dates_arima, forecast_arima, label="ARIMA Forecast", linestyle="dashed", color="red")
    ax.set_title("ARIMA Price Forecast")
    ax.legend()
    st.pyplot(fig)

    # Linear Regression Forecast
    st.subheader("Linear Regression Forecast (Next 10 Days)")
    future_dates_lr, forecast_lr = linear_regression_forecast(data, steps=10)
    forecast_df_lr = pd.DataFrame({'Date': future_dates_lr, 'Predicted Price': forecast_lr})
    st.write(forecast_df_lr)

    # Plot Linear Regression Forecast
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label="Actual Price")
    ax.plot(future_dates_lr, forecast_lr, label="Linear Regression Forecast", linestyle="dashed", color="green")
    ax.set_title("Linear Regression Price Forecast")
    ax.legend()
    st.pyplot(fig)

else:
    st.error("No data found for the given symbol and market. Please check your inputs.")

# Add some custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)
