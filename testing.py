import time
import yfinance as yf
from alpaca_trade_api import REST, TimeFrame
import os

# Set up Alpaca API credentials
os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'
os.environ['APCA_API_KEY_ID'] = 'PKTPDZKAR9IV1XT4R79R'
os.environ['APCA_API_SECRET_KEY'] = 'gNXVOt28TiC1paJMVV6XUVlgCBMtEVUJbnfgMnay'

# Initialize Alpaca API client
alpaca_api = REST(
    os.getenv('APCA_API_KEY_ID'),
    os.getenv('APCA_API_SECRET_KEY'),
    os.getenv('APCA_API_BASE_URL'),
    api_version='v2'
)

# Function to get stock data from yfinance
def get_yfinance_data(ticker):
    start_time = time.time()
    stock = yf.Ticker(ticker)
    data = stock.history(period='1d')  # Get 1 day of historical data
    end_time = time.time()
    return data, end_time - start_time

# Function to get stock data from Alpaca API using get_bars
def get_alpaca_data(ticker):
    start_time = time.time()
    barset = alpaca_api.get_bars(ticker, TimeFrame.Day, limit=1)  # Get 1 day of historical data
    end_time = time.time()
    return barset, end_time - start_time

# Test both APIs with a sample stock
ticker = 'AAPL'

# Get data from yfinance
yf_data, yf_duration = get_yfinance_data(ticker)
print(f"yfinance data retrieval took {yf_duration:.4f} seconds")

# Get data from Alpaca
alpaca_data, alpaca_duration = get_alpaca_data(ticker)
print(f"Alpaca data retrieval took {alpaca_duration:.4f} seconds")

# Compare results
if yf_duration < alpaca_duration:
    print("yfinance is faster")
else:
    print("Alpaca API is faster")

print("yfinance data:")
print(yf_data)

print("Alpaca data:")
print(alpaca_data)