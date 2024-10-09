import alpaca_trade_api as tradeapi
import os
import alpaca_config as AlpacaConfig

config = AlpacaConfig()
API_KEY, SECRET_KEY, BASE_URL = config.get_credentials()

# Create an API object to interact with Alpaca's API
api = tradeapi.REST(
    os.getenv('APCA_API_KEY_ID'),
    os.getenv('APCA_API_SECRET_KEY'),
    os.getenv('APCA_API_BASE_URL'),
    api_version='v2'
)

def main():
    # Account information to confirm we're connected properly
    account = api.get_account()
    print(f"Account status: {account.status}")
    print(f"Cash available: {account.cash}")

    # Calculate how much AAPL stock can be bought with $1
    aapl_quote = api.get_snapshot('AAPL')
    print(f"AAPL price: {aapl_quote.minute_bar}")

if __name__ == '__main__':
    main()
