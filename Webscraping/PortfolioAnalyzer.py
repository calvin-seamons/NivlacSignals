import json
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from pathlib import Path

class PortfolioAnalyzer:
    def __init__(self, portfolio_file):
        """Initialize the portfolio analyzer with a JSON file containing portfolio data."""
        self.portfolio_file = Path(portfolio_file)
        with open(self.portfolio_file, 'r') as f:
            self.portfolio = json.load(f)
        self.symbols = list(self.portfolio.keys())
        self.stock_data = None
        self._fetch_stock_data()
        
        # Create output directory for plots if it doesn't exist
        self.output_dir = self.portfolio_file.parent.parent / 'analysis_output'
        self.output_dir.mkdir(exist_ok=True)

    def generate_report(self):
        """Generate a comprehensive analysis report."""
        report = {
            'Portfolio Summary': self.get_portfolio_summary(),
            'Market Metrics': self.calculate_market_metrics(),
            'Sector Exposure': self.get_sector_exposure(),
            'Risk Metrics': self.get_risk_metrics_report()
        }
        
        # Generate and save all plots
        plot_messages = []
        plot_messages.append(self.plot_portfolio_allocation())
        plot_messages.append(self.plot_performance_comparison())
        plot_messages.append(self.plot_correlation_matrix())
        
        # Save report to CSV
        report_path = self.output_dir / 'portfolio_analysis_report.csv'
        pd.DataFrame(report['Market Metrics']).to_csv(report_path)
        
        return {
            'report': report,
            'plot_messages': plot_messages,
            'report_path': report_path
        }

    def _fetch_stock_data(self):
        """Fetch current market data for all stocks in portfolio using yfinance."""
        self.stock_data = {}
        
        # Fetch data for all symbols at once to be more efficient
        tickers_str = ' '.join(self.symbols)
        tickers = yf.Tickers(tickers_str)
        
        for symbol in self.symbols:
            try:
                clean_symbol = symbol.strip('/')
                stock = tickers.tickers.get(clean_symbol)
                
                if stock is None:
                    print(f"Warning: Could not fetch data for {symbol}")
                    continue
                    
                hist = stock.history(period='1y')
                if not hist.empty:
                    hist.index = hist.index.tz_localize(None)
                    
                    self.stock_data[clean_symbol] = {
                        'info': stock.info if hasattr(stock, 'info') else {},
                        'history': hist,
                        'dividends': stock.dividends if hasattr(stock, 'dividends') else pd.Series(),
                    }
                else:
                    print(f"Warning: No historical data available for {symbol}")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")


    def get_portfolio_summary(self):
        """Generate a summary of the portfolio including total value, returns, and basic metrics."""
        total_value = sum(stock['total_value'] for stock in self.portfolio.values())
        total_cost = sum(stock['cost_basis'] for stock in self.portfolio.values())
        total_return = (total_value - total_cost) / total_cost * 100

        # Calculate allocation by investment type
        type_allocation = {}
        for symbol, data in self.portfolio.items():
            inv_type = data['type']
            if inv_type not in ['-', 'to']:  # Skip invalid types
                type_allocation[inv_type] = type_allocation.get(inv_type, 0) + data['total_value']

        # Calculate percentage allocation
        type_percentages = {k: (v/total_value)*100 for k, v in type_allocation.items()}

        summary = {
            'Total Value': f"${total_value:,.2f}",
            'Total Cost': f"${total_cost:,.2f}",
            'Total Return': f"{total_return:.2f}%",
            'Number of Holdings': len(self.portfolio),
            'Top Holdings': self._get_top_holdings(5),
            'Investment Type Allocation': type_percentages
        }
        return pd.DataFrame([summary]).T

    def _get_top_holdings(self, n=5):
        """Get the top N holdings by value."""
        holdings = [(symbol, data['total_value']) for symbol, data in self.portfolio.items()]
        holdings.sort(key=lambda x: x[1], reverse=True)
        return [f"{symbol}: ${value:,.2f}" for symbol, value in holdings[:n]]

    def calculate_market_metrics(self):
        """Calculate various market metrics for each stock."""
        metrics = {}
        for symbol, stock_info in self.stock_data.items():
            if symbol not in self.portfolio:
                continue

            try:
                hist = stock_info['history']
                if hist.empty:
                    continue

                returns = hist['Close'].pct_change().dropna()
                
                info = stock_info['info']
                current_price = hist['Close'].iloc[-1]
                
                beta = self._calculate_beta(returns)
                volatility = returns.std() * np.sqrt(252)
                
                metrics[symbol] = {
                    'Current Price': f"${current_price:.2f}",
                    'Your Cost Basis': f"${self.portfolio[symbol]['cost_basis']:.2f}",
                    'Your Return': f"{self.portfolio[symbol]['total_change']}",
                    'Your Position': f"${self.portfolio[symbol]['total_value']:.2f}",
                    'Beta': f"{beta:.2f}",
                    'Volatility': f"{(volatility * 100):.2f}%",
                    'Market Cap': self._format_market_cap(info.get('marketCap', 'N/A')),
                    'P/E Ratio': f"{info.get('trailingPE', 'N/A')}",
                    'Dividend Yield': f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else 'N/A',
                }
            except Exception as e:
                print(f"Error calculating metrics for {symbol}: {e}")
                metrics[symbol] = {
                    'Error': 'Could not calculate metrics'
                }
        
        return pd.DataFrame(metrics).T

    def _format_market_cap(self, market_cap):
        """Format market cap into billions or millions."""
        if isinstance(market_cap, (int, float)):
            if market_cap >= 1e12:
                return f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                return f"${market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                return f"${market_cap/1e6:.2f}M"
            else:
                return f"${market_cap:,.0f}"
        return market_cap

    def _calculate_beta(self, stock_returns, market_symbol='^GSPC'):
        """Calculate beta relative to S&P 500 with proper timezone handling."""
        try:
            market = yf.download(market_symbol, 
                               start=datetime.now() - timedelta(days=365),
                               progress=False)
            
            # Convert timezone-aware index to timezone-naive for both series
            market.index = market.index.tz_localize(None)
            if hasattr(stock_returns.index, 'tz'):
                stock_returns.index = stock_returns.index.tz_localize(None)
            
            market_returns = market['Close'].pct_change().dropna()
            
            # Align the dates
            aligned_data = pd.concat([stock_returns, market_returns], axis=1, join='inner')
            aligned_data.columns = ['stock', 'market']
            
            if aligned_data.empty:
                return np.nan
            
            covar = aligned_data['stock'].cov(aligned_data['market'])
            market_var = aligned_data['market'].var()
            
            beta = covar / market_var if market_var != 0 else np.nan
            return beta if not np.isnan(beta) else 0
            
        except Exception as e:
            print(f"Error calculating beta: {e}")
            return 0

    def plot_portfolio_allocation(self):
        """Create a pie chart of portfolio allocation and save to file."""
        plt.figure(figsize=(12, 8))
        values = [stock['total_value'] for stock in self.portfolio.values()]
        labels = [f"{symbol} (${value:.2f})" for symbol, value in zip(self.symbols, values)]
        
        plt.pie(values, labels=labels, autopct='%1.1f%%')
        plt.title('Portfolio Allocation')
        plt.axis('equal')
        
        filepath = self.save_plot(plt, 'portfolio_allocation')
        return f"Portfolio allocation plot saved to: {filepath}"

    def plot_performance_comparison(self):
        """Plot performance comparison of portfolio stocks."""
        plt.figure(figsize=(15, 8))
        
        # Add S&P 500 as benchmark
        spy = yf.download('^GSPC', period='1y', progress=False)
        if not spy.empty:
            normalized_spy = spy['Close'] / spy['Close'].iloc[0]
            plt.plot(normalized_spy, label='S&P 500', linewidth=2, color='black', alpha=0.5)

        for symbol, stock_data in self.stock_data.items():
            if not stock_data['history'].empty:
                prices = stock_data['history']['Close']
                normalized_prices = prices / prices.iloc[0]
                plt.plot(normalized_prices, label=symbol)
        
        plt.title('1-Year Performance Comparison (Normalized)')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def get_sector_exposure(self):
        """Analyze sector exposure of the portfolio."""
        sector_exposure = {}
        total_value = 0
        
        for symbol, stock_info in self.stock_data.items():
            if symbol not in self.portfolio:
                continue
                
            value = self.portfolio[symbol]['total_value']
            total_value += value
            
            sector = stock_info['info'].get('sector', 'Unknown')
            if sector == 'Unknown':
                # Try to categorize ETFs based on symbol
                if any(x in symbol for x in ['VOO', 'SPY', 'IVV']):
                    sector = 'S&P 500 ETF'
                elif any(x in symbol for x in ['XLK', 'VGT']):
                    sector = 'Technology ETF'
                elif any(x in symbol for x in ['XLF', 'VFH']):
                    sector = 'Financial ETF'
                elif any(x in symbol for x in ['VTI', 'ITOT']):
                    sector = 'Total Market ETF'
                    
            sector_exposure[sector] = sector_exposure.get(sector, 0) + value
        
        # Convert to percentages
        sector_percentages = {k: (v/total_value)*100 for k, v in sector_exposure.items()}
        
        # Sort by exposure
        sector_percentages = dict(sorted(sector_percentages.items(), key=lambda x: x[1], reverse=True))
        
        return pd.Series(sector_percentages).round(2)

    def get_risk_metrics_report(self):
        """Generate a comprehensive risk metrics report."""
        metrics = {}
        
        # Calculate portfolio-wide metrics
        total_value = sum(self.portfolio[symbol]['total_value'] for symbol in self.portfolio)
        
        # Calculate weighted metrics
        betas = []
        volatilities = []
        weights = []
        
        for symbol, stock_data in self.stock_data.items():
            if symbol not in self.portfolio:
                continue
                
            try:
                hist = stock_data['history']
                if hist.empty:
                    continue
                    
                returns = hist['Close'].pct_change().dropna()
                weight = self.portfolio[symbol]['total_value'] / total_value
                
                beta = self._calculate_beta(returns)
                volatility = returns.std() * np.sqrt(252)
                
                if not np.isnan(volatility):
                    betas.append(beta)
                    volatilities.append(volatility)
                    weights.append(weight)
                    
            except Exception as e:
                print(f"Error calculating risk metrics for {symbol}: {e}")
                continue
        
        if weights:  # Only calculate if we have valid data
            weights = np.array(weights) / sum(weights)  # Renormalize weights
            portfolio_beta = np.average(betas, weights=weights)
            portfolio_volatility = np.sqrt(np.sum((weights * volatilities) ** 2))
            
            metrics['Portfolio Beta'] = f"{portfolio_beta:.2f}"
            metrics['Portfolio Volatility (Annual)'] = f"{portfolio_volatility*100:.2f}%"
            
            # Add risk concentration
            risk_concentration = {}
            for symbol, weight in zip(self.symbols, weights):
                if symbol in self.portfolio:
                    risk_concentration[symbol] = weight * 100
                    
            top_risk_positions = dict(sorted(risk_concentration.items(), 
                                           key=lambda x: x[1], 
                                           reverse=True)[:5])
            
            metrics['Top Risk Positions'] = {k: f"{v:.2f}%" for k, v in top_risk_positions.items()}
            
            # Calculate Sharpe Ratio (assuming risk-free rate of 5%)
            risk_free_rate = 0.05
            portfolio_return = sum(weight * stock_data['history']['Close'].pct_change().mean() * 252 
                                 for weight, (_, stock_data) in zip(weights, self.stock_data.items()))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            metrics['Portfolio Sharpe Ratio'] = f"{sharpe_ratio:.2f}"
            
        return pd.Series(metrics)
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix and save to file."""
        returns_data = {}
        for symbol, stock_data in self.stock_data.items():
            if not stock_data['history'].empty:
                returns_data[symbol] = stock_data['history']['Close'].pct_change()
        
        returns_df = pd.DataFrame(returns_data).dropna()
        correlation_matrix = returns_df.corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Stock Correlation Matrix')
        
        filepath = self.save_plot(plt, 'correlation_matrix')
        return f"Correlation matrix plot saved to: {filepath}"
    
    def plot_performance_comparison(self):
        """Plot performance comparison of portfolio stocks and save to file."""
        plt.figure(figsize=(15, 8))
        
        # Add S&P 500 as benchmark
        spy = yf.download('^GSPC', period='1y', progress=False)
        if not spy.empty:
            spy.index = spy.index.tz_localize(None)
            normalized_spy = spy['Close'] / spy['Close'].iloc[0]
            plt.plot(normalized_spy, label='S&P 500', linewidth=2, color='black', alpha=0.5)

        for symbol, stock_data in self.stock_data.items():
            if not stock_data['history'].empty:
                prices = stock_data['history']['Close']
                normalized_prices = prices / prices.iloc[0]
                plt.plot(normalized_prices, label=symbol)
        
        plt.title('1-Year Performance Comparison (Normalized)')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        filepath = self.save_plot(plt, 'performance_comparison')
        return f"Performance comparison plot saved to: {filepath}"
    
    def save_plot(self, plt, name):
        """Save plot to file and close it properly."""
        try:
            filepath = self.output_dir / f"{name}.png"
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close()
            return filepath
        except Exception as e:
            print(f"Error saving plot {name}: {e}")
            plt.close()
            return None
    
# Example usage:
if __name__ == "__main__":
    # Use Path for cross-platform compatibility
    file_path = Path("Webscraping/extracted_content/stocks_data.json")
    analyzer = PortfolioAnalyzer(file_path)
    
    # Generate comprehensive report
    results = analyzer.generate_report()
    
    # Print report contents
    print("\nPortfolio Summary:")
    print(results['report']['Portfolio Summary'])
    
    print("\nMarket Metrics:")
    print(results['report']['Market Metrics'])
    
    print("\nSector Exposure:")
    print(results['report']['Sector Exposure'])
    
    print("\nRisk Metrics:")
    print(results['report']['Risk Metrics'])
    
    # Print plot locations
    print("\nGenerated Plots:")
    for message in results['plot_messages']:
        print(message)
    
    print(f"\nDetailed report saved to: {results['report_path']}")