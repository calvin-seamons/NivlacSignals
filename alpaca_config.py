class AlpacaConfig:
    def __init__(self):
        # Users should hardcode their Alpaca API credentials here
        self.api_key = "YOUR_API_KEY_HERE"
        self.secret_key = "YOUR_SECRET_KEY_HERE"
        self.base_url = "https://paper-api.alpaca.markets"

    def get_credentials(self):
        return self.api_key, self.secret_key, self.base_url
    