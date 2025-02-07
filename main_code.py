import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sqlite3
import hashlib
import json
import ta
from scipy import stats

# Page Configuration
st.set_page_config(page_title="Advanced Trading & Risk Management Platform", layout="wide")

# Database setup
def init_db():
    conn = sqlite3.connect('trading_finance.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    # Create portfolio table
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            symbol TEXT NOT NULL,
            quantity REAL NOT NULL,
            entry_price REAL NOT NULL,
            date TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create trading_journal table
    c.execute('''
        CREATE TABLE IF NOT EXISTS trading_journal (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            symbol TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            quantity REAL NOT NULL,
            strategy TEXT NOT NULL,
            notes TEXT,
            date TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

# Authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    conn = sqlite3.connect('trading_finance.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                 (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('trading_finance.db')
    c = conn.cursor()
    c.execute('SELECT id, password FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    if result and result[1] == hash_password(password):
        return result[0]
    return None

# Technical Analysis Functions
def calculate_technical_indicators(data):
    # Moving averages
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
    data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    
    # RSI
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['BB_Upper'] = bollinger.bollinger_hband()
    data['BB_Lower'] = bollinger.bollinger_lband()
    
    return data

# Advanced Risk Management Functions
def calculate_position_size(capital, risk_per_trade, stop_loss_pct):
    """Calculate recommended position size based on risk parameters"""
    max_loss_amount = capital * (risk_per_trade / 100)
    position_size = max_loss_amount / (stop_loss_pct / 100)
    return position_size

def calculate_kelly_criterion(win_rate, win_loss_ratio):
    """Calculate Kelly Criterion for optimal position sizing"""
    q = 1 - win_rate
    kelly = (win_rate * win_loss_ratio - q) / win_loss_ratio
    return max(0, kelly)  # Never recommend negative position sizes

def calculate_risk_metrics(data):
    returns = data['Close'].pct_change().dropna()
    
    metrics = {
        'Daily_Volatility': returns.std() * 100,
        'Annual_Volatility': returns.std() * np.sqrt(252) * 100,
        'Sharpe_Ratio': (returns.mean() / returns.std()) * np.sqrt(252),
        'Sortino_Ratio': returns.mean() / returns[returns < 0].std() * np.sqrt(252),
        'Max_Drawdown': ((data['Close'] / data['Close'].cummax() - 1).min() * 100),
        'VaR_95': np.percentile(returns, 5) * 100,
        'CVaR_95': returns[returns <= np.percentile(returns, 5)].mean() * 100
    }
    
    return metrics

# Machine Learning Prediction Function
def predict_prices(data, forecast_days=30):
    df = data.copy()
    
    # Feature engineering
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Prepare features and target
    df = df.dropna()
    features = ['Returns', 'Volatility', 'SMA_20', 'SMA_50']
    X = df[features]
    y = df['Close']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Generate future features
    last_data = df[features].tail(1)
    future_features = []
    
    for _ in range(forecast_days):
        next_features = last_data.copy()
        future_features.append(next_features.values[0])
        
        # Update features for next prediction
        returns = (model.predict(next_features)[0] / df['Close'].iloc[-1]) - 1
        next_features['Returns'] = returns
        next_features['Volatility'] = df['Volatility'].mean()
        next_features['SMA_20'] = df['SMA_20'].mean()
        next_features['SMA_50'] = df['SMA_50'].mean()
        last_data = next_features
    
    future_features = np.array(future_features)
    predictions = model.predict(future_features)
    
    return predictions

# Main App
def main():
    st.title("Advanced Trading & Risk Management Platform")
    
    # Authentication
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    if not st.session_state.user_id:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                
                if submit:
                    user_id = verify_user(username, password)
                    if user_id:
                        st.session_state.user_id = user_id
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        
        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Register")
                
                if submit:
                    if create_user(new_username, new_password):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Username already exists")
        return
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Market Selection
    market = st.sidebar.selectbox("Choose Market", ["Stocks", "Crypto", "Forex"])
    symbol = st.sidebar.text_input("Enter Symbol", "AAPL" if market == "Stocks" else "BTC/USDT")
    
    # Date Range
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    
    # Risk Management Parameters
    st.sidebar.subheader("Risk Management")
    capital = st.sidebar.number_input("Trading Capital ($)", min_value=100, value=10000)
    risk_per_trade = st.sidebar.number_input("Risk per Trade (%)", min_value=0.1, max_value=5.0, value=1.0)
    stop_loss = st.sidebar.number_input("Stop Loss (%)", min_value=0.1, max_value=10.0, value=2.0)
    
    # Fetch Data
    try:
        if market == "Stocks":
            data = yf.download(symbol, start=start_date, end=end_date)
        elif market == "Crypto":
            exchange = ccxt.binance()
            ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=int(start_date.timestamp() * 1000))
            data = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['Date'] = pd.to_datetime(data['Date'], unit='ms')
            data.set_index('Date', inplace=True)
        
        # Calculate Technical Indicators
        data = calculate_technical_indicators(data)
        
        # Main Content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Price Chart with Technical Indicators")
            fig = go.Figure()
            
            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="OHLC"
            ))
            
            # Add Moving Averages
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name="SMA 20"))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name="SMA 50"))
            
            # Add Bollinger Bands
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], name="BB Upper",
                                   line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], name="BB Lower",
                                   line=dict(dash='dash')))
            
            fig.update_layout(title=f"{symbol} Price Chart", xaxis_title="Date",
                            yaxis_title="Price", height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Risk Metrics")
            metrics = calculate_risk_metrics(data)
            
            for metric, value in metrics.items():
                st.metric(metric.replace('_', ' '), f"{value:.2f}%")
            
            # Position Sizing Calculator
            st.subheader("Position Sizing")
            position_size = calculate_position_size(capital, risk_per_trade, stop_loss)
            st.write(f"Recommended Position Size: ${position_size:.2f}")
            
            # Kelly Criterion
            win_rate = st.slider("Historical Win Rate (%)", 30, 70, 50)
            win_loss_ratio = st.slider("Win/Loss Ratio", 1.0, 3.0, 2.0)
            kelly_pct = calculate_kelly_criterion(win_rate/100, win_loss_ratio) * 100
            st.write(f"Kelly Criterion Suggested Allocation: {kelly_pct:.1f}%")
        
        # Price Prediction
        st.subheader("Price Prediction")
        predictions = predict_prices(data)
        
        pred_fig = go.Figure()
        pred_fig.add_trace(go.Scatter(x=data.index, y=data['Close'],
                                    name="Historical Price"))
        
        future_dates = pd.date_range(start=data.index[-1], periods=len(predictions)+1)[1:]
        pred_fig.add_trace(go.Scatter(x=future_dates, y=predictions,
                                    name="Predicted Price", line=dict(dash='dash')))
        
        pred_fig.update_layout(title="Price Prediction (30 Days)",
                             xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(pred_fig, use_container_width=True)
        
        # Trading Journal
        st.subheader("Trading Journal")
        with st.form("journal_entry"):
            cols = st.columns(4)
            entry_price = cols[0].number_input("Entry Price", min_value=0.0)
            exit_price = cols[1].number_input("Exit Price", min_value=0.0)
            quantity = cols[2].number_input("Quantity", min_value=0.0)
            strategy = cols[3].selectbox("Strategy", ["Trend Following", "Mean Reversion",
                                                    "Breakout", "Scalping"])
            notes = st.text_area("Trading Notes")
            submit_journal = st.form_submit_button("Add Trade")
            
            if submit_journal:
                conn = sqlite3.connect('trading_finance.db')
                c = conn.cursor()
                c.execute('''
                    INSERT INTO trading_journal
                    (user_id, symbol, entry_price, exit_price, quantity, strategy, notes, date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (st.session_state.user_id, symbol, entry_price, exit_price,
                     quantity, strategy, notes, datetime.now().strftime("%Y-%m-%d")))
                conn.commit()
                conn.close()
                st.success("Trade recorded successfully!")
        
        # Display Trading Journal
        conn = sqlite3.connect('trading_finance.db')
        journal_df = pd.read_sql_query('''
            SELECT * FROM trading_journal
            WHERE user_id = ?
            ORDER BY date DESC
        ''', conn, params=(st.session_state.user_id,))
        conn.close()
        
        if not journal_df.empty:
            st.dataframe(journal_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
    
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.user_id = None
        st.rerun()

if __name__ == "__main__":
    main()
