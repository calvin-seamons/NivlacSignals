import yfinance as yf
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def test_yahoo_finance_price_target(ticker_symbol):
    """Fetches price target from Yahoo Finance using yfinance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        price_target = ticker.info.get('targetMeanPrice')
        print(f"[Yahoo Finance] Mean Price Target for {ticker_symbol}: {price_target}")
    except Exception as e:
        print(f"[Yahoo Finance] Error: {e}")

def test_marketbeat_price_target(ticker_symbol):
    """Fetches price target from MarketBeat by scraping the website."""
    try:
        url = f"https://www.marketbeat.com/stocks/NASDAQ/{ticker_symbol}/price-target/"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        price_target_span = soup.find("span", class_="rating-title")
        if price_target_span:
            price_target = price_target_span.text.strip()
            print(f"[MarketBeat] Price Target for {ticker_symbol}: {price_target}")
        else:
            raise ValueError("Price target data not found on MarketBeat.")
    except Exception as e:
        print(f"[MarketBeat] Error: {e}")

def test_zacks_price_target(ticker_symbol):
    """Fetches price target and related data from Zacks using Selenium for dynamic content."""
    try:
        url = f"https://www.zacks.com/stock/research/{ticker_symbol}/price-target-stock-forecast"
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        driver.get(url)
        
        # Wait for page to load (adjust time as necessary)
        driver.implicitly_wait(10)

        # Get page source after JavaScript loads
        page_source = driver.page_source
        driver.quit()

        # Use BeautifulSoup to parse the dynamically loaded page
        soup = BeautifulSoup(page_source, 'html.parser')
        table = soup.find("div", class_="key-expected-earnings-data-module price-targets").find("tbody")
        rows = table.find_all("tr")[0]

        average_price = rows.find_all("th", class_="align_center")[0].text.strip()
        high_price = rows.find_all("td", class_="align_center")[0].text.strip()
        low_price = rows.find_all("td", class_="align_center")[1].text.strip()
        upside_percentage = rows.find_all("td", class_="align_center")[2].text.strip()

        print(f"[Zacks] Price Targets for {ticker_symbol}:")
        print(f"  - Average: {average_price}")
        print(f"  - High: {high_price}")
        print(f"  - Low: {low_price}")
        print(f"  - Upside Percentage: {upside_percentage}")
    except Exception as e:
        print(f"[Zacks] Error: {e}")

def run_all_tests(ticker_symbol):
    """Runs all test functions for the given ticker symbol."""
    print(f"Running tests for {ticker_symbol}...\n")
    test_yahoo_finance_price_target(ticker_symbol)
    test_marketbeat_price_target(ticker_symbol)
    test_zacks_price_target(ticker_symbol)
    print("\nTesting complete.")

# Example usage
if __name__ == "__main__":
    run_all_tests("AAPL")
