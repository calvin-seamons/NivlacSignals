import alpaca_trade_api as tradeapi
import yaml

# Load Alpaca API credentials from the YAML configuration file
with open("alpaca_config.yaml", "r") as file:
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