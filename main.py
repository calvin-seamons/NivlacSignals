import alpaca_trade_api as tradeapi
import os

# Set your API credentials as environment variables for security
os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'
os.environ['APCA_API_KEY_ID'] = 'PKTPDZKAR9IV1XT4R79R'
os.environ['APCA_API_SECRET_KEY'] = 'gNXVOt28TiC1paJMVV6XUVlgCBMtEVUJbnfgMnay'

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
    aapl_quote = api.get_latest_trade('AAPL')
    price_per_share = aapl_quote.price
    quantity = round(1 / price_per_share, 4)  # Adjusting quantity for small purchases

    # Place a market order for $1 worth of Apple stock
    order = api.submit_order(
        symbol='AAPL',
        qty=quantity,
        side='buy',
        type='market',
        time_in_force='day'
    )

    print(f"Order submitted: {order}")

if __name__ == '__main__':
    main()
