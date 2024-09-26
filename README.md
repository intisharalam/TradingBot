# Stock Market Prediction and Strategy Backtesting with Machine Learning

This project is an implementation of a machine learning pipeline to predict stock market movements using historical data and technical indicators. The pipeline incorporates multiple machine learning models and backtests a trading strategy based on the best-performing model using the Backtrader library.

## Features

- **Data Collection:** Historical stock data (OHLC and volume) is fetched using the `yfinance` library.
- **Technical Indicators:** Several technical indicators are calculated using the `TA-Lib` library, including:
  - Simple Moving Average (SMA)
  - Relative Strength Index (RSI)
  - Average Directional Index (ADX)
  - Normalized Average True Range (NATR)
  - Percentage change over different time periods
  - Rolling correlation between SMA and price
- **Machine Learning Models:** The project uses the following models to predict whether the stock will close higher or lower in the next period:
  - Support Vector Machine (SVM)
  - Bagging SVM
  - Stacked Model (SVM, Random Forest, Logistic Regression)
  - Voting Classifier (SVM, Random Forest, Logistic Regression)
  
- **Model Selection:** The best-performing model is selected based on test accuracy.
- **Strategy Backtesting:** The strategy based on the best model's predictions is backtested using the Backtrader library.
  
## Example Results

### Plot 1: Cumulative Returns

The chart below shows the comparison between the cumulative market returns and the strategy returns based on the best-performing machine learning model.

<p align="center">
  <img src="https://github.com/intisharalam/TradingBot/blob/main/Cummulative_Returns.png" alt="Market vs Strategy Returns" width="600"/>
</p>

### Plot 2: Backtest Results

Below is a plot from the Backtrader backtest showing the performance of the strategy over the test period.

![Backtest Results](https://github.com/intisharalam/TradingBot/blob/main/Backtesting_Results.png)

## Installation

To run the project, you'll need to have Python 3.x installed along with the following libraries:

```bash
pip install yfinance talib numpy pandas scikit-learn matplotlib alpaca-trade-api backtrader
```

## Usage

1. **Download Historical Data:** The project downloads 1 month of 15-minute interval data for the Tesla (TSLA) stock by default. You can change the stock symbol or the period in the script.

2. **Run the Machine Learning Pipeline:**
   - The script calculates the technical indicators, trains multiple machine learning models, and evaluates their accuracy.
   - The best-performing model is selected based on test accuracy.

3. **Backtest the Trading Strategy:**
   - The strategy based on the predictions from the best model is backtested using the Backtrader library.
   - The results are plotted to show cumulative returns and the backtest results.

To run the script:

```bash
python main.py
```

Make sure to replace `stock_prediction.py` with the name of your script file.

## Project Structure

```
.
├── stock_prediction.py    # Main script with the machine learning and backtesting pipeline
├── README.md              # This README file
├── cumulative_returns_plot.png  # Result image for cumulative returns
├── backtest_results_plot.png    # Result image for backtest performance
```

## Requirements

- **Python 3.x**
- **Libraries:**
  - `yfinance` for fetching stock data.
  - `TA-Lib` for calculating technical indicators.
  - `numpy`, `pandas` for data manipulation.
  - `scikit-learn` for machine learning models.
  - `backtrader` for strategy backtesting.
  - `matplotlib` for plotting results.

## Backtesting

The backtest uses a simple strategy based on the predicted signal:

- **Buy:** If the model predicts the price will go up, we buy using 20% of the current cash balance.
- **Sell:** If the model predicts the price will go down, we sell the current position.

The strategy is run for the last 30 days of the test set, and the results are plotted using Backtrader.

## Contributing

Feel free to open issues or contribute to the project by submitting pull requests. All contributions are welcome!

## License

This project is open-source and available under the MIT License.
