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



















import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
import hashlib
import json

# Database setup
def init_db():
    conn = sqlite3.connect('financee.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    # Create transactions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            title TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            type TEXT NOT NULL,
            date TEXT NOT NULL,
            is_recurring BOOLEAN DEFAULT 0,
            recurring_interval TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create budget_goals table
    c.execute('''
        CREATE TABLE IF NOT EXISTS budget_goals (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            category TEXT NOT NULL,
            amount REAL NOT NULL,
            period TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    conn = sqlite3.connect('financee.db')
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
    conn = sqlite3.connect('financee.db')
    c = conn.cursor()
    c.execute('SELECT id, password FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    if result and result[1] == hash_password(password):
        return result[0]
    return None

# Configure the page
st.set_page_config(
    page_title="Financee",
    page_icon="💰",
    layout="wide"
)

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'categories' not in st.session_state:
    st.session_state.categories = ['Food', 'Transport', 'Entertainment', 'Shopping', 'Bills', 'Other']

# Authentication UI
if not st.session_state.user_id:
    st.title("💰 Financee - Login")
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
                    st.error("Invalid username or password")
    
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

else:
    # Main title
    st.title("💰 Financee")
    
    # Sidebar
    with st.sidebar:
        if st.button("Logout"):
            st.session_state.user_id = None
            st.rerun()
            
        st.header("Add Transaction")
        
        # Transaction form
        with st.form("transaction_form"):
            title = st.text_input("Title")
            amount = st.number_input("Amount", min_value=0.0, step=0.01)
            category = st.selectbox("Category", st.session_state.categories)
            transaction_type = st.radio("Type", ["Expense", "Income"])
            date = st.date_input("Date")
            
            # Recurring transaction options
            is_recurring = st.checkbox("Recurring Transaction")
            recurring_interval = None
            if is_recurring:
                recurring_interval = st.selectbox(
                    "Interval",
                    ["Daily", "Weekly", "Monthly", "Yearly"]
                )
            
            submit = st.form_submit_button("Add Transaction")
            
            if submit and title and amount:
                conn = sqlite3.connect('financee.db')
                c = conn.cursor()
                c.execute('''
                    INSERT INTO transactions 
                    (user_id, title, amount, category, type, date, is_recurring, recurring_interval)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    st.session_state.user_id,
                    title,
                    amount,
                    category,
                    transaction_type.lower(),
                    date.strftime("%Y-%m-%d"),
                    1 if is_recurring else 0,
                    recurring_interval
                ))
                conn.commit()
                conn.close()
                st.success("Transaction added successfully!")
        
        # Budget Goals
        st.header("Budget Goals")
        with st.form("budget_form"):
            goal_category = st.selectbox("Category", st.session_state.categories, key="goal_category")
            goal_amount = st.number_input("Monthly Budget", min_value=0.0, step=10.0)
            submit_goal = st.form_submit_button("Set Budget Goal")
            
            if submit_goal:
                conn = sqlite3.connect('financee.db')
                c = conn.cursor()
                c.execute('''
                    INSERT OR REPLACE INTO budget_goals (user_id, category, amount, period)
                    VALUES (?, ?, ?, 'monthly')
                ''', (st.session_state.user_id, goal_category, goal_amount))
                conn.commit()
                conn.close()
                st.success("Budget goal set successfully!")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Recent Transactions")
        
        # Get transactions from database
        conn = sqlite3.connect('financee.db')
        transactions_df = pd.read_sql_query('''
            SELECT title, amount, category, type, date, is_recurring, recurring_interval
            FROM transactions
            WHERE user_id = ?
            ORDER BY date DESC
        ''', conn, params=(st.session_state.user_id,))
        conn.close()
        
        if not transactions_df.empty:
            st.dataframe(transactions_df, use_container_width=True, hide_index=True)
        else:
            st.info("No transactions yet. Add some using the sidebar!")

    with col2:
        st.subheader("Summary")
        
        if not transactions_df.empty:
            # Calculate total income and expenses
            total_income = transactions_df[transactions_df['type'] == 'income']['amount'].sum()
            total_expenses = transactions_df[transactions_df['type'] == 'expense']['amount'].sum()
            balance = total_income - total_expenses
            
            # Display metrics
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Income", f"${total_income:.2f}", "+")
            col_b.metric("Expenses", f"${total_expenses:.2f}", "-")
            col_c.metric("Balance", f"${balance:.2f}")
            
            # Expenses by category
            expenses_df = transactions_df[transactions_df['type'] == 'expense']
            if not expenses_df.empty:
                st.subheader("Expenses by Category")
                expenses_by_category = expenses_df.groupby('category')['amount'].sum()
                fig = px.pie(
                    values=expenses_by_category.values,
                    names=expenses_by_category.index,
                    title="Expense Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Budget Goals Progress
        st.subheader("Budget Goals Progress")
        conn = sqlite3.connect('financee.db')
        goals_df = pd.read_sql_query('''
            SELECT category, amount
            FROM budget_goals
            WHERE user_id = ?
        ''', conn, params=(st.session_state.user_id,))
        conn.close()
        
        if not goals_df.empty:
            current_month = datetime.now().strftime("%Y-%m")
            monthly_expenses = transactions_df[
                (transactions_df['type'] == 'expense') &
                (transactions_df['date'].str.startswith(current_month))
            ].groupby('category')['amount'].sum()
            
            for _, row in goals_df.iterrows():
                category = row['category']
                goal = row['amount']
                spent = monthly_expenses.get(category, 0)
                progress = (spent / goal) * 100 if goal > 0 else 0
                
                st.write(f"{category} Budget")
                st.progress(min(progress / 100, 1.0))
                st.write(f"Spent: ${spent:.2f} / ${goal:.2f}")

# Add some custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)
