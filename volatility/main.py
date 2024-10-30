from volatility import volest
import yfinance as yf
import pandas as pd
import time

# Common parameters
estimator = 'GarmanKlass'
window = 30
windows = [30, 60, 90, 120]
quantiles = [0.25, 0.75]
bins = 100
density = True
required_columns = ['Open', 'High', 'Low', 'Close']

# Define pairs to analyze with descriptions
pairs = [
    # Tech Stocks vs Indices
    ('AAPL', 'QQQ', 'Apple vs NASDAQ'),
    ('MSFT', 'SPY', 'Microsoft vs S&P 500'),
    ('GOOGL', 'QQQ', 'Google vs NASDAQ'),
    
    # Financial Sector
    ('JPM', 'XLF', 'JPMorgan vs Financial Sector'),
    ('GS', 'XLF', 'Goldman Sachs vs Financial Sector'),
    ('MS', 'XLF', 'Morgan Stanley vs Financial Sector'),
    
    # Energy Sector
    ('XOM', 'XLE', 'Exxon vs Energy Sector'),
    ('CVX', 'XLE', 'Chevron vs Energy Sector'),
    ('COP', 'XLE', 'ConocoPhillips vs Energy Sector'),
    
    # Consumer Sectors
    ('AMZN', 'XLY', 'Amazon vs Consumer Discretionary'),
    ('WMT', 'XLP', 'Walmart vs Consumer Staples'),
    ('PG', 'XLP', 'P&G vs Consumer Staples'),
    
    # Healthcare Sector
    ('JNJ', 'XLV', 'Johnson & Johnson vs Healthcare'),
    ('PFE', 'XLV', 'Pfizer vs Healthcare'),
    ('UNH', 'XLV', 'UnitedHealth vs Healthcare'),
    
    # Cross-Sector Comparisons
    ('TSLA', 'SPY', 'Tesla vs S&P 500'),
    ('NVDA', 'SMH', 'NVIDIA vs Semiconductor ETF'),
    ('META', 'SOCL', 'Meta vs Social Media ETF'),
    
    # Market Cap Comparisons
    ('IWM', 'SPY', 'Small Caps vs Large Caps'),
    ('MDY', 'SPY', 'Mid Caps vs Large Caps')
]

def analyze_pair(stock, benchmark, description):
    """Analyze a single stock-benchmark pair"""
    try:
        print(f"\n{'='*80}")
        print(f"Analyzing {description}")
        print(f"Stock: {stock}, Benchmark: {benchmark}")
        
        # Get data
        stock_data = yf.Ticker(stock).history(period="5y")
        bench_data = yf.Ticker(benchmark).history(period="5y")
        
        # Process data
        common_dates = stock_data.index.intersection(bench_data.index)
        if len(common_dates) < max(windows):
            print(f"Warning: Insufficient data points for {stock}-{benchmark} pair")
            return False
            
        stock_data = stock_data[required_columns].loc[common_dates]
        bench_data = bench_data[required_columns].loc[common_dates]
        
        # Set attributes
        stock_data = pd.DataFrame(stock_data)
        bench_data = pd.DataFrame(bench_data)
        setattr(stock_data, 'symbol', stock)
        setattr(bench_data, 'symbol', benchmark)
        
        print(f"Data period: {common_dates[0].strftime('%Y-%m-%d')} to {common_dates[-1].strftime('%Y-%m-%d')}")
        print(f"Number of trading days: {len(common_dates)}")
        
        # Create estimator
        vol = volest.VolatilityEstimator(
            price_data=stock_data,
            estimator=estimator,
            bench_data=bench_data
        )
        
        # Generate term sheet
        vol.term_sheet(
            window=window,
            windows=windows,
            quantiles=quantiles,
            bins=bins,
            density=density
        )
        
        return True
        
    except Exception as e:
        print(f"Error analyzing {description}: {str(e)}")
        return False

def main():
    start_time = time.time()
    successful_analyses = 0
    failed_analyses = 0
    
    print("Starting Volatility Analysis")
    print(f"Using {estimator} estimator")
    print(f"Window sizes: {windows}")
    print(f"Quantiles: {quantiles}")
    print("-" * 80)
    
    for stock, benchmark, description in pairs:
        if analyze_pair(stock, benchmark, description):
            successful_analyses += 1
        else:
            failed_analyses += 1
        
        # Add a small delay to prevent rate limiting
        time.sleep(1)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\nAnalysis Complete")
    print("-" * 80)
    print(f"Total pairs analyzed: {len(pairs)}")
    print(f"Successful analyses: {successful_analyses}")
    print(f"Failed analyses: {failed_analyses}")
    print(f"Total time taken: {duration:.2f} seconds")
    print(f"Average time per analysis: {duration/len(pairs):.2f} seconds")

if __name__ == "__main__":
    main()