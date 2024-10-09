# NivlacSignals
Automated stock trading for the non weary of heart

**Python Version:** 3.12

## Overview
NivlacSignals is an automated trading application designed for those who want to dive into the world of algorithmic trading without hesitation. This bot identifies undervalued stocks, monitors price momentum, and automates buying and selling based on carefully crafted criteria. Whether you're a beginner or an experienced trader, NivlacSignals aims to make your trading life easier, more efficient, and more profitable.

## Features
- **Undervaluation Analysis**: Evaluate stocks to determine if they are undervalued based on fundamental metrics like P/E ratio, P/B ratio, and DCF.
- **Momentum-Based Entry**: Buy stocks that show a positive price movement of 3-4% to ride the wave of upward momentum.
- **Trailing Stop Sell**: Ride the stock's upward movement until it decreases by 3-4% from the peak, automating the selling process.
- **Paper Trading**: Test your strategies in a risk-free environment using Alpaca's paper trading feature.
- **Backtesting**: Use historical data to simulate and validate your trading strategy before going live.

## Installation
1. **Clone the Repository**
   ```sh
   git clone https://github.com/yourusername/NivlacSignals.git
   cd NivlacSignals
   ```

2. **Run the Setup Script**
- Run the corresponding script to your os.
- Located in the Scripts folder

## Configuration
1. **Set API Keys**: To interact with Alpaca, configure your API keys in the alpaca_config.yaml file

2. **Adjust Strategy Parameters**: You can customize the strategy parameters like the undervaluation threshold, buy percentage, and sell percentage in the configuration file.

## Usage
1. **Run the Application**
   ```sh
   python main.py
   ```

2. **Input Stock Tickers**: Provide the stock ticker(s) you want the bot to evaluate.

3. **Trade Execution**: Let the bot automate the buying and selling process based on the pre-defined criteria.

## Example Workflow
1. **Stock Evaluation**: Input the ticker of the stock you're interested in. The bot will assess whether the stock is undervalued using various fundamental metrics.
2. **Price Monitoring**: Once the stock passes the undervaluation test, the bot will monitor its price. When the stock's price increases by 3-4%, it will initiate a buy order.
3. **Riding the Momentum**: The bot will continue monitoring the stock's price movement. It will hold the position until the price drops by 3-4% from the highest point, triggering a sell order.

## Architecture
- **Data Source**: Data is pulled from `yfinance` and `Alpaca` for real-time analysis.
- **Broker API**: Orders are placed using the Alpaca API, which supports both paper trading and live trading.
- **Backtesting**: Historical data from Alpaca is used for backtesting via the `Backtrader` library.
- **Visualization**: Trades, performance metrics, and other analytics are visualized using `Matplotlib` and `Pandas`.

## Dependencies
See requirements.txt file

## Development Roadmap
- **Add More Fundamental Metrics**: Integrate more indicators for undervaluation, like Debt-to-Equity ratio and Earnings Yield.
- **Multi-Language Support**: Expand the bot to analyze stocks in different markets, not just the U.S.
- **Dashboard**: Develop a web-based dashboard to provide a live view of trading activity, stock evaluations, and performance metrics.
- **Machine Learning Integration**: Integrate ML models to enhance prediction capabilities for undervaluation and momentum-based signals.

## Risk Warning
This software is designed for educational purposes and **does not guarantee profits**. Algorithmic trading carries risks, and you should only invest money you are prepared to lose. Always do your due diligence before trading.

## License
NivlacSignals is licensed under the MIT License. See `LICENSE` for more information.

## Contribution
Feel free to open issues or contribute to the project! Pull requests are always welcome.

## Contact
For any questions or suggestions, please reach out to us at [calvinseamons35@gmail.com].

Happy Trading! 🚀📈
