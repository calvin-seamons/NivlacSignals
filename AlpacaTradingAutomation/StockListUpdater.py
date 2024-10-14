import yfinance as yf
from tqdm import tqdm
import pandas as pd
from StockIndustryTracker import StockIndustryTracker
import requests
import io
from ftplib import FTP

class StockListUpdater:
    def __init__(self, tracker):
        self.tracker = tracker

    def get_sp500_tickers(self):
        # Get S&P 500 tickers from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(url)[0]
        return table['Symbol'].tolist()

    def get_nasdaq_tickers(self):
        ftp_host = "ftp.nasdaqtrader.com"
        ftp_path = "/SymbolDirectory/nasdaqlisted.txt"
        
        # Connect to the FTP server
        ftp = FTP(ftp_host)
        ftp.login()  # anonymous login
        
        # Download the file content
        content = []
        ftp.retrlines(f"RETR {ftp_path}", content.append)
        
        # Close the FTP connection
        ftp.quit()
        
        # Join the lines and create a DataFrame
        file_content = '\n'.join(content)
        df = pd.read_csv(io.StringIO(file_content), sep='|')
        
        # Filter out the header and footer rows
        df = df[df['Symbol'] != 'Symbol']
        df = df[df['Symbol'] != 'File Creation Time:']
        
        return df['Symbol'].tolist()

    def get_custom_tickers(self, filename):
        # Read custom list of tickers from a file
        with open(filename, 'r') as file:
            return [line.strip() for line in file]

    def update_from_list(self, tickers):
        results = self.tracker.update_stocks(tickers)
        return results

    def update_sp500(self):
        tickers = self.get_sp500_tickers()
        print(f"Updating {len(tickers)} S&P 500 stocks...")
        return self.update_from_list(tickers)

    def update_nasdaq(self):
        tickers = self.get_nasdaq_tickers()
        print(f"Updating {len(tickers)} NASDAQ stocks...")
        return self.update_from_list(tickers)

    def update_custom(self, filename):
        tickers = self.get_custom_tickers(filename)
        print(f"Updating {len(tickers)} stocks from custom list...")
        return self.update_from_list(tickers)

    def update_all(self):
        results = []
        results.extend(self.update_sp500())
        results.extend(self.update_nasdaq())
        return results

# Usage example
tracker = StockIndustryTracker()
updater = StockListUpdater(tracker)

# Update S&P 500 stocks
# sp500_results = updater.update_sp500()

# Update NASDAQ stocks
nasdaq_results = updater.update_nasdaq()

# Print the report
tracker.print_report()