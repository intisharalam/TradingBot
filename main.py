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
import warnings

# Initialize warnings
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')

# Fetch historical data (OHLC and Volume)
ticker = "TSLA"
data = yf.download(ticker, start='2014-01-01', end='2024-01-01', interval='1d')

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
split_percentage = 0.8
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




import alpaca_trade_api as tradeapi
import yfinance as yf
import numpy as np
import pandas as pd
import talib as ta
import time
from sklearn.preprocessing import StandardScaler

# Alpaca API Keys
API_KEY = "PKYE6V5P5MYF0BXJDUHX"
SECRET_KEY = "U2hIsIa9FNkg53dYlOS9QthS8ritojCRvarAEqW9"
BASE_URL = "https://paper-api.alpaca.markets"


# Connect to Alpaca API
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')


# Function to fetch live data
def fetch_live_data(ticker, period='5d', interval='1m'):
    data = yf.download(ticker, period=period, interval=interval)
    return data


# Function to fetch buying power
def get_buying_power():
    account = api.get_account()
    return float(account.buying_power)


# Function to make a trade
def trade(signal, ticker, max_investment=500):
    try:
        buying_power = get_buying_power()
        last_price = float(api.get_latest_trade(ticker).price)

        # Ensure the last price is valid
        if last_price <= 0:
            print(f"Invalid last price for {ticker}: {last_price}. Cannot place an order.")
            return

        # Calculate maximum quantity based on max investment
        qty = int(max_investment / last_price)  # Integer division to get whole shares

        if signal == 1:  # Buy signal
            # Check if there is enough buying power and qty is greater than 0
            if buying_power >= (qty * last_price) and qty > 0:
                print(f"Buying {qty} shares of {ticker} for ${last_price * qty:.2f}")
                api.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
            else:
                print(f"Not enough buying power to buy {qty} shares of {ticker}. Current buying power: ${buying_power:.2f}")

        elif signal == 0:  # Sell signal
            if qty > 0:
                print(f"Selling {qty} shares of {ticker}")
                api.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
            else:
                print(f"No shares to sell for {ticker}. Quantity: {qty}")

        else:
            print(f"Invalid signal for {ticker}.")
    
    except Exception as e:
        print(f"An error occurred while processing the trade for {ticker}: {e}")

# Predict and execute trade
def predict_and_trade(ticker):
    try:
        # Fetch live data
        df = fetch_live_data(ticker)
        
        # Prepare the DataFrame (same as you did before)
        df['Open-Close'] = (df["Open"] - df['Close']).shift(-1)
        df['Close-High'] = (df["Close"] - df['High']).shift(-1)
        df['Close-Low'] = (df["Close"] - df['Low']).shift(-1)
        df['High-Low'] = df['High'] - df['Low']
        df['Mid'] = (df['High'] + df['Low']) / 2
        
        # Calculate technical indicators (same as before)
        df['rsi'] = ta.RSI(df['Close'].values, timeperiod=14)
        df['adx'] = ta.ADX(df['High'].values, df['Low'].values, df['Open'].values, timeperiod=50)
        df['NATR'] = ta.NATR(df['High'], df['Low'], df['Close'], timeperiod=50)
        df['pct_change5'] = df['Close'].pct_change(5)
        df['pct_change'] = df['Close'].pct_change()
        df['sma'] = ta.SMA(df['Close'], timeperiod=30)
        df['corr'] = df['Close'].rolling(window=int(6.5*4)).corr(df['sma'])
        
        df.dropna(inplace=True)

        # Feature set
        X_live = df[['sma','pct_change','pct_change5','rsi', 'adx', 'corr', 'Volume', 'Open-Close', 'Close-Low', 'Close-High', 'High-Low','Mid']].copy()

        # Scale the features using the same scaler
        X_live_scaled = scaler.transform(X_live)

        # Use the trained model to predict
        predicted_signal = best_model.predict(X_live_scaled[-1].reshape(1, -1))[0]

        # Execute trade based on prediction
        trade(predicted_signal, ticker)

    except Exception as e:
        print(f"An error occurred during trading for {ticker}: {e}")

# Set the ticker symbol
ticker = "TSLA"

while True:
    predict_and_trade(ticker)
    time.sleep(60)  # Wait 1 minute before checking again
