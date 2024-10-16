import alpaca_trade_api as tradeapi
import yfinance as yf
from StockIndustryTracker import StockIndustryTracker

class UndervalueEstimator:
    def __init__(self, ticker, api):
        """
        Initialize the UndervalueEstimator with a stock ticker and Alpaca API object.
        
        :param ticker: The stock ticker symbol
        :param api: An instance of the Alpaca API
        """
        self.ticker = ticker
        self.api = api
        self.tracker = StockIndustryTracker()

    def get_industry_peers(self):
        """
        Fetch industry peers for the given stock ticker using StockIndustryTracker.
        
        :return: A list of peer stock tickers
        """
        stock = yf.Ticker(self.ticker)
        industry = stock.info.get('industry', 'Unknown')
        industry_peers = self.tracker.get_industry_data(industry)
        
        if industry in self.tracker.industry_data:
            return industry_peers
        else:
            print(f"No industry data found for {self.ticker} in industry: {industry}")
            return []
    @staticmethod
    def get_stock_metrics(ticker):
        """
        Fetch financial metrics for a stock ticker using yfinance.
        
        Metrics to retrieve:
        1. P/E Ratio (Price to Earnings Ratio)
        2. P/B Ratio (Price to Book Ratio)
        3. Debt to Equity Ratio
        4. Earnings Growth Rate
        5. Current Price
        6. Market Capitalization
        7. Revenue
        8. Net Income
        9. Total Assets
        10. Total Liabilities
        11. Free Cash Flow
        
        :param ticker: The stock ticker symbol
        :return: A dictionary containing the fetched financial metrics
        """
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        # print(stock.cash_flow)

        metrics = {
            "P/E Ratio": info.get("trailingPE"),
            "P/B Ratio": info.get("priceToBook"),
            "Debt to Equity Ratio": info.get("debtToEquity"),
            "Earnings Growth Rate": info.get("earningsQuarterlyGrowth"),
            "Current Price": info.get("currentPrice"),
            "Market Capitalization": info.get("marketCap"),
            "Revenue": financials.loc["Total Revenue"].iloc[0] if "Total Revenue" in financials.index else None,
            "Net Income": financials.loc["Net Income"].iloc[0] if "Net Income" in financials.index else None,
            "Total Assets": balance_sheet.loc["Total Assets"].iloc[0] if "Total Assets" in balance_sheet.index else None,
            "Total Liabilities": balance_sheet.loc["Total Liabilities Net Minority Interest"].iloc[0] if "Total Liabilities Net Minority Interest" in balance_sheet.index else None,
            "Free Cash Flow": cash_flow.loc["Free Cash Flow"].iloc[0] if "Free Cash Flow" in cash_flow.index else None,
            "PEG Ratio": info.get("pegRatio")
        }

        return metrics

    def _get_fundamental_metrics(self):
        """
        Get fundamental metrics for the stock using Alpaca API.
        
        :return: A dictionary of fundamental metrics
        """
        pass

    def _get_industry_average(self):
        """
        Calculate industry average P/E and P/B ratios using Alpaca API data.
        
        :return: A dictionary containing average P/E and P/B ratios
        """
        pass

    def evaluate_undervaluation(self):
        """
        Evaluate the undervaluation score of the stock based on various metrics.
        
        This method will:
        1. Check for P/E ratio undervaluation
        2. Check for P/B ratio undervaluation
        3. Perform DCF analysis
        4. Assess debt-to-equity ratio
        5. Evaluate earnings growth
        
        :return: An undervaluation score (0-100)
        """
        pass


# Usage example
estimator = UndervalueEstimator("AAPL", api=None)
peers = estimator.get_industry_peers()
print(peers)