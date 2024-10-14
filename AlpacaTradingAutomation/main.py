import datetime
import alpaca_trade_api as tradeapi
import yaml
import os
import time
import pandas as pd
import yfinance as yf

# Load Alpaca API credentials from YAML configuration
with open(os.path.join("alpaca_config.yaml"), "r") as file:
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

def get_rising_stocks(percent_increase, time_frame_minutes):
    # Calculate the start and end times
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(minutes=time_frame_minutes)

    # Convert time_frame_minutes to a valid yfinance interval
    if time_frame_minutes <= 60:
        interval = "1m"
    elif time_frame_minutes <= 60 * 24:
        interval = "1h"
    else:
        interval = "1d"

    # Get list of all tickers (this is a simple example, you might want to use a more comprehensive list)
    tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()

    rising_stocks = []

    for ticker in tickers:
        try:
            # Fetch data
            data = yf.Ticker(ticker).history(start=start_time, end=end_time, interval=interval)
            
            if not data.empty:
                start_price = data.iloc[0]['Close']
                end_price = data.iloc[-1]['Close']
                percent_change = (end_price - start_price) / start_price * 100

                if percent_change >= percent_increase:
                    rising_stocks.append({
                        'symbol': ticker,
                        'percent_change': percent_change
                    })
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")

    return rising_stocks

# Usage
rising_stocks = get_rising_stocks(percent_increase=3, time_frame_minutes=60)
for stock in rising_stocks:
    print(f"{stock['symbol']}: {stock['percent_change']:.2f}%")