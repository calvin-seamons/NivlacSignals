import yfinance as yf
from collections import defaultdict
import json
from tqdm import tqdm

class StockIndustryTracker:
    def __init__(self, filename='stock_industry_data.txt'):
        self.filename = filename
        self.industry_data = self.load_data()
        self.convert_to_dict_structure()

    def load_data(self):
        try:
            with open(self.filename, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return defaultdict(lambda: {'stocks': {}, 'total_pe': 0, 'count': 0})

    def convert_to_dict_structure(self):
        for industry, data in self.industry_data.items():
            if isinstance(data['stocks'], list):
                data['stocks'] = {stock: 0 for stock in data['stocks']}
                data['total_pe'] = 0
                data['count'] = len(data['stocks'])

    def save_data(self):
        with open(self.filename, 'w') as file:
            json.dump(self.industry_data, file, indent=2)

    def update_stock(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            industry = info.get('industry', 'Unknown')
            pe_ratio = info.get('trailingPE')

            if industry and pe_ratio and pe_ratio > 0:
                # Initialize the industry if it doesn't exist
                if industry not in self.industry_data:
                    self.industry_data[industry] = {'stocks': {}, 'total_pe': 0, 'count': 0}

                old_industry = None
                for ind, data in self.industry_data.items():
                    if ticker in data['stocks']:
                        old_industry = ind
                        break

                if old_industry and old_industry != industry:
                    # Remove from old industry
                    old_pe = self.industry_data[old_industry]['stocks'].pop(ticker, 0)
                    self.industry_data[old_industry]['total_pe'] -= old_pe
                    self.industry_data[old_industry]['count'] -= 1

                if ticker in self.industry_data[industry]['stocks']:
                    # Update existing entry
                    old_pe = self.industry_data[industry]['stocks'][ticker]
                    self.industry_data[industry]['total_pe'] = self.industry_data[industry]['total_pe'] - old_pe + pe_ratio
                else:
                    # Add new entry
                    self.industry_data[industry]['count'] += 1
                    self.industry_data[industry]['total_pe'] += pe_ratio

                self.industry_data[industry]['stocks'][ticker] = pe_ratio
                
                return f"Updated {ticker} in {industry} industry."
            else:
                return f"Could not update {ticker}. Missing industry or valid P/E ratio."
        except Exception as e:
            return f"Error updating {ticker}: {str(e)}"

    def update_stocks(self, tickers):
        results = []
        for ticker in tqdm(tickers, desc="Updating stocks"):
            result = self.update_stock(ticker)
            results.append(result)
        self.save_data()  # Save data after all updates
        return results

    def get_industry_pe(self, industry):
        data = self.industry_data[industry]
        if data['count'] > 0:
            return data['total_pe'] / data['count']
        return None

    def print_report(self):
        for industry, data in self.industry_data.items():
            avg_pe = self.get_industry_pe(industry)
            print(f"\nIndustry: {industry}")
            print(f"Average P/E Ratio: {avg_pe:.2f}" if avg_pe else "Average P/E Ratio: N/A")
            print("Stocks:")
            for ticker, pe in data['stocks'].items():
                print(f"  {ticker}: P/E {pe:.2f}")

# Usage example
tracker = StockIndustryTracker()

print(tracker.get_industry_pe("Discount Stores"))