import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from fbprophet import Prophet
import requests

st.title("Trading Strategy, Risk Management & AI Predictions")

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

# AI Predictions - LSTM Model
def lstm_forecast(data, steps=10):
    data = data.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X_train, y_train = [], []
    for i in range(60, len(data_scaled) - steps):
        X_train.append(data_scaled[i-60:i])
        y_train.append(data_scaled[i:i+steps])

    X_train, y_train = np.array(X_train), np.array(y_train)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        LSTM(50),
        Dense(steps)
    ])
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

    X_test = data_scaled[-60:].reshape(1, 60, 1)
    prediction = model.predict(X_test)
    prediction = scaler.inverse_transform(prediction)[0]
    return prediction

# AI Predictions - Prophet Model
def prophet_forecast(data, steps=10):
    df = data.reset_index()
    df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(steps)

# Sentiment Analysis (Fetch Stock News)
def fetch_news_sentiment(symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey=YOUR_NEWSAPI_KEY"
    response = requests.get(url).json()
    if 'articles' in response:
        articles = response['articles'][:5]  # Get top 5 articles
        sentiments = []
        for article in articles:
            text = article['title'] + " " + article['description']
            sentiment_score = 1 if "positive" in text.lower() else -1 if "negative" in text.lower() else 0
            sentiments.append(sentiment_score)
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        return avg_sentiment, articles
    return 0, []

# Main App
st.header(f"{market} - {strategy} Strategy")

symbol = st.text_input("Enter Symbol", "TCS")
start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))

data = fetch_data(market, symbol, start_date, end_date)

if not data.empty:
    st.write(f"Data for {symbol}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col) for col in data.columns]

    st.write(data)

    fig = px.line(data, x=data.index, y=data.iloc[:, 0], title=f"{symbol} Price Chart")
    st.plotly_chart(fig)

    # Market Predictions
    st.header("AI Market Predictions")

    # ARIMA Forecast
    st.subheader("ARIMA Forecast (Next 10 Days)")
    arima_forecast = ARIMA(data.iloc[:, 0], order=(5,1,0)).fit().forecast(steps=10)
    st.write(pd.DataFrame({'Date': pd.date_range(start=data.index[-1], periods=11, freq='D')[1:], 'Predicted Price': arima_forecast}))

    # LSTM Forecast
    st.subheader("LSTM Forecast (Next 10 Days)")
    lstm_predictions = lstm_forecast(data.iloc[:, 0])
    st.write(pd.DataFrame({'Date': pd.date_range(start=data.index[-1], periods=11, freq='D')[1:], 'Predicted Price': lstm_predictions}))

    # Prophet Forecast
    st.subheader("Prophet Forecast (Next 10 Days)")
    prophet_pred = prophet_forecast(data.iloc[:, 0])
    st.write(prophet_pred)

    # Sentiment Analysis
    st.subheader("Sentiment Analysis on Stock News")
    sentiment_score, articles = fetch_news_sentiment(symbol)
    sentiment_label = "Positive 😊" if sentiment_score > 0 else "Negative 😞" if sentiment_score < 0 else "Neutral 😐"
    st.write(f"Overall Sentiment: {sentiment_label}")

    for article in articles:
        st.write(f"**{article['title']}** - {article['source']['name']}")
        st.write(article['url'])

else:
    st.error("No data found for the given symbol and market. Please check your inputs.")
