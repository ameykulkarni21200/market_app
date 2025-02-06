import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import plotly.express as px
import requests
import json
import sqlite3
import hashlib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
from textblob import TextBlob
from tpot import TPOTRegressor
import websocket

# --- Streamlit UI ---
st.title("Trading Strategy, Risk Management & AI Market Prediction")

# Sidebar Inputs
st.sidebar.header("Market Selection")
market = st.sidebar.selectbox("Choose Market", ["Indian Market", "Crypto Market", "Forex Market"])

st.sidebar.header("Trading Strategy")
strategy = st.sidebar.selectbox("Choose Strategy", ["Trend Following", "Mean Reversion", "Scalping", "Market Making"])

st.sidebar.header("Risk Management")
stop_loss = st.sidebar.number_input("Stop-Loss (%)", min_value=0.1, max_value=10.0, value=2.0)
position_size = st.sidebar.number_input("Position Size (% of Capital)", min_value=1, max_value=100, value=10)
confidence_interval = st.sidebar.number_input("VaR Confidence Interval (%)", min_value=90, max_value=99, value=95)

# --- Data Fetching Function ---
def fetch_data(market, symbol, start_date, end_date):
    if market == "Indian Market":
        return yf.download(f"{symbol}.NS", start=start_date, end=end_date)
    elif market == "Crypto Market":
        exchange = ccxt.binance()
        data = exchange.fetch_ohlcv(symbol, timeframe='1d', since=exchange.parse8601(start_date))
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.set_index('timestamp')
    elif market == "Forex Market":
        return yf.download(symbol, start=start_date, end=end_date)

# --- AI Feature Importance (RandomForest) ---
def feature_importance(data):
    data = data.dropna()
    if data.shape[0] < 30: return "Insufficient data for training!"
    
    y = data.iloc[:, 3].shift(-1).dropna()  # 'Close' column by index (3)
    X = data.iloc[:-1, 1:]  # Drop 'Close' column and use other columns as features
    model = RandomForestRegressor()
    model.fit(X, y)
    
    return pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values(by="Importance", ascending=False)

# --- Sentiment Analysis ---
def get_sentiment_analysis(symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey=YOUR_NEWSAPI_KEY"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    sentiment_scores = [TextBlob(article["title"]).sentiment.polarity for article in articles]
    return np.mean(sentiment_scores) if sentiment_scores else "No news available"

# --- Portfolio Optimization ---
def optimize_portfolio(symbols):
    prices = {s: yf.download(s, period="1y")['Close'] for s in symbols}
    df = pd.DataFrame(prices).dropna()
    mu = mean_historical_return(df)
    S = CovarianceShrinkage(df).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    return weights

# --- AI AutoML Model Selection (TPOT) ---
def automl_prediction(data):
    data = data.dropna()
    if data.shape[0] < 30: return "Insufficient data!"
    
    y = data.iloc[:, 3].shift(-1).dropna()  # 'Close' column by index (3)
    X = data.iloc[:-1, 1:]  # Drop 'Close' column and use other columns as features
    automl = TPOTRegressor(generations=5, population_size=20, verbosity=0)
    automl.fit(X, y)
    return automl.predict(X.iloc[-1:])[0]

# --- Real-time Data Streaming (Crypto) ---
def on_message(ws, message):
    st.write(f"Live Data: {message}")

def on_open(ws):
    ws.send(json.dumps({"method": "SUBSCRIBE", "params": ["btcusdt@trade"], "id": 1}))

def start_streaming():
    ws = websocket.WebSocketApp("wss://stream.binance.com:9443/ws", on_open=on_open, on_message=on_message)
    ws.run_forever()

# --- Backtesting Trading Strategy ---
def backtest_strategy(data, strategy):
    if strategy == "Trend Following":
        data['Signal'] = np.where(data.iloc[:, 3] > data.iloc[:, 3].rolling(20).mean(), 1, -1)  # 'Close' column by index (3)
    elif strategy == "Mean Reversion":
        data['Signal'] = np.where(data.iloc[:, 3] < data.iloc[:, 3].rolling(20).mean(), 1, -1)  # 'Close' column by index (3)
    return data.iloc[:, [3, -1]]  # 'Close' and 'Signal' columns by index

# --- Risk Metrics Calculation ---
def calculate_var(data, confidence_interval=95):
    returns = data.pct_change().dropna()
    return -np.percentile(returns, 100 - confidence_interval) * 100

def calculate_max_drawdown(data):
    peak = data.cummax()
    drawdown = (data - peak) / peak
    return drawdown.min() * 100

# --- ARIMA Market Forecast ---
def arima_forecast(data, steps=10):
    model = ARIMA(data, order=(5,1,0)).fit()
    return model.forecast(steps=steps)

# --- Main Execution ---
st.header(f"{market} - {strategy} Strategy")
symbol = st.text_input("Enter Symbol", "TCS")
start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-02-05"))
data = fetch_data(market, symbol, start_date, end_date)

if not data.empty:
    st.write(f"Data for {symbol}")
    st.write(data)
    # Get the column name by index
    column_name = data.columns[3]  # Assuming 'Close' is at index 3
    fig = px.line(data, x=data.index, y=column_name, title=f"{symbol} Price Chart")

    st.plotly_chart(fig)

    # Risk Management
    st.header("Risk Management Metrics")
    var = calculate_var(data.iloc[:, 3], confidence_interval)  # 'Close' column by index (3)
    max_drawdown = calculate_max_drawdown(data.iloc[:, 3])  # 'Close' column by index (3)
    st.write(f"Value at Risk (VaR): {var:.2f}%")
    st.write(f"Max Drawdown: {max_drawdown:.2f}%")

    # AI Model Predictions
    st.header("AI Predictions")
    forecast = arima_forecast(data.iloc[:, 3])  # 'Close' column by index (3)
    st.write(f"ARIMA Forecast: {forecast}")

    # Feature Importance
    st.header("Feature Importance")
    st.write(feature_importance(data))

    # Sentiment Analysis
    st.header("Market Sentiment")
    sentiment = get_sentiment_analysis(symbol)
    st.write(f"Sentiment Score: {sentiment}")

    # Backtesting
    st.header("Backtesting Strategy")
    st.write(backtest_strategy(data, strategy))

    # Portfolio Optimization
    st.header("Portfolio Optimization")
    optimized_portfolio = optimize_portfolio(["AAPL", "GOOGL", "MSFT"])
    st.write(optimized_portfolio)

else:
    st.error("No data found. Check your inputs.")
