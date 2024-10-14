import datetime
import alpaca_trade_api as tradeapi
import yaml
from UndervalueEstimator import UndervalueEstimator
import yfinance as yf
import os

# Load Alpaca API credentials from YAML configuration
with open(os.path.join("..", "alpaca_config.yaml"), "r") as file:
    config = yaml.safe_load(file)
api_key = config['alpaca']['api_key']
secret_key = config['alpaca']['secret_key']
base_url = config['alpaca']['base_url']

# Create an API object to interact with Alpaca's API
api = tradeapi.REST(
    api_key,
    secret_key,
    base_url,
    api_version='v2'
)

def get_stock_data(ticker):
    """
    Fetch historical data for a stock ticker using Alpaca.
    """
    try:
        # Fetching data for the last 6 months (using daily bars)
        end_date = datetime.datetime.now() - datetime.timedelta(days=1)
        start_date = end_date - datetime.timedelta(days=180)
        bars = api.get_bars(ticker, timeframe='1Day', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d')).df
        if bars.empty:
            raise ValueError("No data found for the ticker.")
        return bars
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def main():
    ticker = "AAPL"
    print(f"Fetching data for {ticker}...")
    stock_data = get_stock_data(ticker)

    if not stock_data.empty:
        print(stock_data.tail())
    else:
        print("Stock data retrieved successfully.")
        # undervalue_estimator = UndervalueEstimator(ticker, api, stock_data)
        # score = undervalue_estimator.evaluate_undervaluation()
        # print(stock_data)
        # snapshot = api.get_snapshot(ticker)
        # print(snapshot.latest_trade)
        # Get apple information from yf
        apple = yf.Ticker("AAPL")
        print(apple.quarterly_balance_sheet.astype(str))

if __name__ == '__main__':
    main()