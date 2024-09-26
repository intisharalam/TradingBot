from datetime import datetime
import numpy as np
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, StackingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import alpaca_trade_api as tradeapi
import backtrader as bt
import warnings

# Initialize warnings
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')

# Fetch historical data (OHLC and Volume)
ticker = "TSLA"
data = yf.download(ticker, period='1mo', interval='15m')

# Prepare the DataFrame
df = data.copy()
df['Open-Close'] = (df["Open"] - df['Close']).shift(-1)
df['Close-High'] = (df["Close"] - df['High']).shift(-1)
df['Close-Low'] = (df["Close"] - df['Low']).shift(-1)
df['High-Low'] = df['High'] - df['Low']
df['Mid'] = (df['High'] + df['Low']) / 2

# Calculate technical indicators
df['rsi'] = ta.RSI(df['Close'].values, timeperiod=14)
df['adx'] = ta.ADX(df['High'].values, df['Low'].values, df['Open'].values, timeperiod=50)
df['NATR'] = ta.NATR(df['High'], df['Low'], df['Close'], timeperiod=50)
df['pct_change5'] = df['Close'].pct_change(5)
df['pct_change'] = df['Close'].pct_change()
df['sma'] = ta.SMA(df['Close'], timeperiod=30)
df['corr'] = df['Close'].rolling(window=int(6.5*4)).corr(df['sma'])

df.dropna(inplace=True)

# Add these features to your feature set
X = df[['sma','pct_change','pct_change5','rsi', 'adx', 'corr', 'Volume', 'Open-Close', 'Close-Low', 'Close-High','High-Low','Mid']].copy()
y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)


# Split the data into training and testing sets
split_percentage = 0.4
split = int(split_percentage * len(df))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'SVM': SVC(probability=True),
    'Bagging SVM': BaggingClassifier(estimator=SVC(), n_estimators=50, random_state=42),
    'Stacked Model': StackingClassifier(
        estimators=[
            ('svm', SVC(probability=True)),
            ('rf', RandomForestClassifier(n_estimators=100)),
            ('lr', LogisticRegression())
        ],
        final_estimator=LogisticRegression()
    ),
    'Voting Classifier': VotingClassifier(
        estimators=[
            ('svm', SVC(probability=True)),
            ('rf', RandomForestClassifier(n_estimators=100)),
            ('lr', LogisticRegression())
        ],
        voting='soft'
    ),
}

# Train and evaluate models
accuracies = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[name] = accuracy
    print(f"{name} Test Accuracy: {accuracy * 100:.2f}%")

# Determine the best model
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
print(f"Best Model: {best_model_name} with Accuracy: {accuracies[best_model_name] * 100:.2f}%")



import logging
import alpaca_trade_api as tradeapi
import yfinance as yf
import numpy as np
import pandas as pd
import talib as ta
import time
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(filename='trading_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Alpaca API Keys
API_KEY = "PKSJIADPZFG55Y3EKDWR"
SECRET_KEY = "fh3pwRtnuzwBhTbOkG2XQEvrbkOhgWzIga75SBnH"
BASE_URL = "https://paper-api.alpaca.markets"

# Connect to Alpaca API
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# Fetch live data for 15-minute intervals
def fetch_live_data(ticker, period='5d', interval='15m'):
    try:
        data = yf.download(ticker, period=period, interval=interval)
        return data
    except Exception as e:
        logging.error(f"Error fetching live data for {ticker}: {e}")
        return None

# Fetch buying power
def get_buying_power():
    try:
        account = api.get_account()
        return float(account.buying_power)
    except tradeapi.rest.APIError as e:
        logging.error(f"API error fetching buying power: {e}")
        return 0  # Returning 0 in case of an error so no trades are made
    except Exception as e:
        logging.error(f"Unexpected error fetching buying power: {e}")
        return 0

# Get current position in stock
def get_current_position(ticker):
    try:
        position = api.get_position(ticker)
        return int(position.qty)
    except tradeapi.rest.APIError as e:
        logging.info(f"No position found for {ticker}. Returning 0. Error: {e}")
        return 0  # No position found
    except Exception as e:
        logging.error(f"Error fetching position for {ticker}: {e}")
        return 0

# Trade based on the signal
def trade(signal, ticker, max_investment=500):
    try:
        # Get buying power and latest price
        buying_power = get_buying_power()
        last_price = float(api.get_last_trade(ticker).price)
        
        # Calculate quantity to buy/sell
        qty = int(max_investment / last_price)
        current_position = get_current_position(ticker)
        
        # Buy signal
        if signal == 1 and qty > 0:
            if buying_power >= (qty * last_price):
                logging.info(f"Buying {qty} shares of {ticker} at ${last_price}")
                api.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
            else:
                logging.warning(f"Not enough buying power to buy {qty} shares of {ticker}. Available: ${buying_power:.2f}")
        
        # Sell signal
        elif signal == 0 and current_position > 0:
            logging.info(f"Selling {current_position} shares of {ticker}")
            api.submit_order(
                symbol=ticker,
                qty=current_position,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
        else:
            logging.info(f"No valid trade action for signal {signal} on {ticker}.")
    
    except tradeapi.rest.APIError as e:
        logging.error(f"Alpaca API error during trade execution for {ticker}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during trade for {ticker}: {e}")

# Predict and trade
def predict_and_trade(ticker):
    try:
        # Fetch live data
        df = fetch_live_data(ticker, period='5d', interval='15m')
        if df is None or df.empty:
            logging.error(f"No data available for {ticker}. Skipping trade.")
            return
        
        # Prepare features
        df['Open-Close'] = (df["Open"] - df['Close']).shift(-1)
        df['Close-High'] = (df["Close"] - df['High']).shift(-1)
        df['Close-Low'] = (df["Close"] - df['Low']).shift(-1)
        df['High-Low'] = df['High'] - df['Low']
        df['Mid'] = (df['High'] + df['Low']) / 2
        
        # Calculate technical indicators
        df['rsi'] = ta.RSI(df['Close'].values, timeperiod=14)
        df['adx'] = ta.ADX(df['High'].values, df['Low'].values, df['Open'].values, timeperiod=50)
        df['NATR'] = ta.NATR(df['High'], df['Low'], df['Close'], timeperiod=50)
        df['pct_change5'] = df['Close'].pct_change(5)
        df['pct_change'] = df['Close'].pct_change()
        df['sma'] = ta.SMA(df['Close'], timeperiod=30)
        df['corr'] = df['Close'].rolling(window=int(6.5*4)).corr(df['sma'])
        
        df.dropna(inplace=True)

        # Prepare the feature set for prediction
        X_live = df[['sma','pct_change','pct_change5','rsi', 'adx', 'corr', 'Volume', 'Open-Close', 'Close-Low', 'Close-High', 'High-Low','Mid']].copy()

        # Ensure the scaler and model are loaded
        if 'scaler' not in globals() or 'best_model' not in globals():
            logging.error("Scaler or model not found. Ensure the model is loaded correctly.")
            return
        
        # Scale the features
        X_live_scaled = scaler.transform(X_live)

        # Predict the signal
        predicted_signal = best_model.predict(X_live_scaled[-1].reshape(1, -1))[0]
        
        # Execute trade based on prediction
        trade(predicted_signal, ticker)

    except Exception as e:
        logging.error(f"Error during prediction or trade execution for {ticker}: {e}")

# Main strategy loop running every 15 minutes
def run_strategy():
    ticker = "TSLA"
    while True:
        try:
            predict_and_trade(ticker)
            time.sleep(15 * 60)  # Wait 15 minutes before checking again
        except Exception as e:
            logging.error(f"Critical error in strategy loop: {e}")

# Execute the strategy
run_strategy()
