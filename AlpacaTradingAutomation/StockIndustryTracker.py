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
            return defaultdict(lambda: {'stocks': {}, 'total_pe': 0, 'total_pb': 0, 'count': 0})

    def convert_to_dict_structure(self):
        for industry, data in self.industry_data.items():
            if isinstance(data['stocks'], list):
                data['stocks'] = {stock: {'pe': 0, 'pb': 0} for stock in data['stocks']}
                data['total_pe'] = 0
                data['total_pb'] = 0
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
            pb_ratio = info.get('priceToBook')

            if industry:
                # Initialize the industry if it doesn't exist
                if industry not in self.industry_data:
                    self.industry_data[industry] = {'stocks': {}, 'total_pe': 0, 'total_pb': 0, 'count': 0, 'pb_count': 0}

                # Initialize or update the stock entry
                if ticker not in self.industry_data[industry]['stocks']:
                    self.industry_data[industry]['stocks'][ticker] = {'pe': None, 'pb': None}
                    self.industry_data[industry]['count'] += 1

                # Update P/E ratio
                if pe_ratio and pe_ratio > 0:
                    old_pe = self.industry_data[industry]['stocks'][ticker]['pe'] or 0
                    self.industry_data[industry]['total_pe'] = self.industry_data[industry]['total_pe'] - old_pe + pe_ratio
                    self.industry_data[industry]['stocks'][ticker]['pe'] = pe_ratio

                # Update P/B ratio
                if pb_ratio and pb_ratio > 0:
                    old_pb = self.industry_data[industry]['stocks'][ticker]['pb'] or 0
                    self.industry_data[industry]['total_pb'] = self.industry_data[industry]['total_pb'] - old_pb + pb_ratio
                    self.industry_data[industry]['stocks'][ticker]['pb'] = pb_ratio
                    if old_pb == 0:
                        self.industry_data[industry]['pb_count'] += 1
                else:
                    self.industry_data[industry]['stocks'][ticker]['pb'] = None

                return f"Updated {ticker} in {industry} industry."
            else:
                return f"Could not update {ticker}. Missing industry information."
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

    def get_industry_pb(self, industry):
        data = self.industry_data[industry]
        if data['count'] > 0:
            return data['total_pb'] / data['count']
        return None

    def print_report(self):
        for industry, data in self.industry_data.items():
            avg_pe = self.get_industry_pe(industry)
            avg_pb = self.get_industry_pb(industry)
            print(f"\nIndustry: {industry}")
            print(f"Average P/E Ratio: {avg_pe:.2f}" if avg_pe else "Average P/E Ratio: N/A")
            print(f"Average P/B Ratio: {avg_pb:.2f}" if avg_pb else "Average P/B Ratio: N/A")
            print("Stocks:")
            for ticker, ratios in data['stocks'].items():
                print(f"  {ticker}: P/E {ratios['pe']:.2f}, P/B {ratios['pb']:.2f}")

    def add_pb_ratio_to_existing_stocks(self):
        updated_count = 0
        unavailable_count = 0
        error_count = 0

        for industry, data in tqdm(self.industry_data.items(), desc="Updating industries"):
            if 'pb_count' not in data:
                data['pb_count'] = 0
            if 'total_pb' not in data:
                data['total_pb'] = 0

            for ticker in tqdm(list(data['stocks'].keys()), desc=f"Updating stocks in {industry}", leave=False):
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    pb_ratio = info.get('priceToBook')
                    
                    if not isinstance(data['stocks'][ticker], dict):
                        data['stocks'][ticker] = {'pe': data['stocks'][ticker], 'pb': None}
                    
                    if pb_ratio and pb_ratio > 0:
                        old_pb = data['stocks'][ticker].get('pb') or 0
                        data['stocks'][ticker]['pb'] = pb_ratio
                        data['total_pb'] = data['total_pb'] - old_pb + pb_ratio
                        if old_pb == 0:
                            data['pb_count'] += 1
                        updated_count += 1
                    else:
                        data['stocks'][ticker]['pb'] = None
                        unavailable_count += 1
                except Exception as e:
                    print(f"Error updating P/B ratio for {ticker}: {str(e)}")
                    data['stocks'][ticker]['pb'] = None
                    error_count += 1
        
        self.save_data()
        print(f"P/B ratios update complete:")
        print(f"  Updated: {updated_count}")
        print(f"  Unavailable: {unavailable_count}")
        print(f"  Errors: {error_count}")


# Usage example
tracker = StockIndustryTracker()
tracker.add_pb_ratio_to_existing_stocks()
