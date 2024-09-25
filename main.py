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

# Predict and plot strategy returns using the best model
df_test = df[split:].copy()  # Test data
df_test['Predicted_Signal'] = best_model.predict(X_test_scaled)
df_test['Return'] = df_test['Close'].pct_change()
df_test['Strategy_Return'] = df_test['Return'] * df_test['Predicted_Signal'].shift(1)
df_test['Cum_Ret'] = df_test['Return'].cumsum()
df_test['Cum_Strategy'] = df_test['Strategy_Return'].cumsum()




import alpaca_trade_api as tradeapi
import yfinance as yf
import numpy as np
import pandas as pd
import talib as ta
import time
from sklearn.preprocessing import StandardScaler

# Alpaca API Keys
API_KEY = "PKBED76LKCYK6SP01NU8"
SECRET_KEY = "srZSLBgtsu0j5eKMH5KCWdm9QD0bu7w01xjHkRUH"
BASE_URL = "https://paper-api.alpaca.markets"


# Connect to Alpaca API
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')


# Function to fetch live data
def fetch_live_data(ticker, period='5d', interval='15m'):
    data = yf.download(ticker, period=period, interval=interval)
    return data


# Function to fetch buying power
def get_buying_power():
    account = api.get_account()
    return float(account.buying_power)


# Function to make a trade
def trade(signal, ticker, qty=1):
    buying_power = get_buying_power()
    last_price = float(api.get_latest_trade(ticker).price)

    if signal == 1 and (buying_power >= (qty * last_price)):  # Only buy if there's enough buying power
        print(f"Buying {ticker}")
        api.submit_order(
            symbol=ticker,
            qty=qty,  # Modify the quantity as needed
            side='buy',
            type='market',
            time_in_force='gtc'
        )
    elif signal == 0:
        print(f"Selling {ticker}")
        api.submit_order(
            symbol=ticker,
            qty=qty,
            side='sell',
            type='market',
            time_in_force='gtc'
        )
    else:
        print(f"Not enough buying power to buy {ticker}")



# Predict and execute trade
def predict_and_trade(ticker):
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

# Set the ticker symbol
ticker = "TSLA"

while True:
    predict_and_trade(ticker)
    time.sleep(60*15)  # Wait 15 minute before checking again
