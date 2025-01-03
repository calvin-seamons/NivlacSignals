from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import os
import time

class WebScraper:
    def __init__(self, headless: bool = True, slow_mo: int = 0):
        self.headless = headless
        self.slow_mo = slow_mo
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def start(self):
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(
                headless=self.headless,
                slow_mo=self.slow_mo,
                args=['--disable-gpu', '--disable-dev-shm-usage', '--no-sandbox']
            )
            self.context = self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                ignore_https_errors=True
            )
            
            # Enable request interception to block unnecessary resources
            # self.context.route("**/*.{png,jpg,jpeg,gif,svg,css,font,woff,woff2}", lambda route: route.abort())
            # self.context.route("**/analytics.js", lambda route: route.abort())
            # self.context.route("**/gtm.js", lambda route: route.abort())
            
            self.page = self.context.new_page()
            self.logger.info("Browser session started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start browser session: {str(e)}")
            raise

    def wait_for_page_load(self):
        """Wait for page load with shorter timeout"""
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=10000)
            if "login" in self.page.url or "auth" in self.page.url:
                self.page.wait_for_load_state("networkidle", timeout=10000)
        except PlaywrightTimeout:
            self.logger.warning("Timeout while waiting for page to load")
        except Exception as e:
            self.logger.error(f"Error waiting for page load: {str(e)}")

    def wait_for_full_page_load(self):
        """Wait for all network requests to finish and the page to be fully loaded"""
        try:
            self.page.wait_for_load_state("load", timeout=60000)
            time.sleep(10)
            self.page.wait_for_load_state("networkidle", timeout=60000)
            self.logger.info("Full page load completed successfully")
        except PlaywrightTimeout:
            self.logger.warning("Timeout while waiting for full page load")
        except Exception as e:
            self.logger.error(f"Error waiting for full page load: {str(e)}")

    def login(self, url: str, credentials: Dict[str, str], 
              selectors: Dict[str, str]) -> bool:
        try:
            # Navigate to login page
            self.page.goto(url, wait_until="domcontentloaded")
            
            # Wait for login form elements
            self.page.wait_for_selector(selectors['username_field'], timeout=10000)
            self.page.wait_for_selector(selectors['password_field'], timeout=10000)
            self.page.wait_for_selector(selectors['login_button'], timeout=10000)
            
            # Fill credentials
            self.page.fill(selectors['username_field'], credentials['username'])
            self.page.fill(selectors['password_field'], credentials['password'])
            
            # Click login and wait for navigation
            with self.page.expect_navigation(wait_until="networkidle"):
                self.page.click(selectors['login_button'])
            
            return True
                
        except PlaywrightTimeout:
            self.logger.error("Timeout while trying to log in")
            return False
        except Exception as e:
            self.logger.error(f"Login failed: {str(e)}")
            return False

    def capture_page_content(self) -> Optional[str]:
        """Capture page content"""
        try:
            # Get text content directly
            text_content = self.page.evaluate('''() => {
                return document.documentElement.outerText;
            }''')
            
            # Create folder to save extracted content
            output_folder = 'extracted_content'
            os.makedirs(output_folder, exist_ok=True)
            
            # Save text content
            with open(os.path.join(output_folder, 'text_content.txt'), 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            # Save page structure
            elements_info = self.page.evaluate('''() => {
                function getElementInfo(element, depth = 0) {
                    if (depth >= 5 || !element) return null;
                    return {
                        tag: element.tagName ? element.tagName.toLowerCase() : null,
                        id: element.id || null,
                        classes: element.classList ? Array.from(element.classList) : [],
                        text: element.innerText ? element.innerText.slice(0, 100) : null,
                        children: Array.from(element.children)
                            .map(child => getElementInfo(child, depth + 1))
                            .filter(Boolean)
                    };
                }
                return getElementInfo(document.body);
            }''')
            
            with open(os.path.join(output_folder, 'page_structure.json'), 'w', encoding='utf-8') as f:
                json.dump(elements_info, f, indent=2)
            
            return text_content
                    
        except Exception as e:
            self.logger.error(f"Error capturing page content: {str(e)}")
            return None

    def parse_stocks_from_text(self, text_content: str) -> dict:
        """Parse stock information"""
        stocks = {}
        try:
            lines = [line.strip() for line in text_content.split('\n') if line.strip()]
            
            current_stock = None
            data_keys = [
                'price', 'change_percent', 'type', 'est_return', 'est_max_drawdown',
                'allocation', 'shares', 'total_change', 'cost_basis', 'total_value'
            ]
            data_index = 0
            for line in lines:
                if line.isupper() and len(line) <= 5:  # Assuming stock symbols are uppercase and short
                    current_stock = line
                    stocks[current_stock] = {}
                    data_index = 0
                elif current_stock:
                    if data_index < len(data_keys):
                        key = data_keys[data_index]
                        if key == 'price' or key == 'cost_basis' or key == 'total_value':
                            if line.startswith('$'):
                                stocks[current_stock][key] = float(line.replace('$', '').replace(',', ''))
                                data_index += 1
                        elif key == 'change_percent' or key == 'total_change':
                            if '%' in line:
                                stocks[current_stock][key] = line
                                data_index += 1
                        elif key == 'allocation':
                            if '%' in line:
                                stocks[current_stock][key] = line
                                data_index += 1
                        elif key == 'shares':
                            try:
                                stocks[current_stock][key] = float(line)
                                data_index += 1
                            except ValueError:
                                continue
                        elif key == 'type' or key == 'est_return' or key == 'est_max_drawdown':
                            stocks[current_stock][key] = line
                            data_index += 1
        except Exception as e:
            self.logger.error(f"Error parsing stocks: {str(e)}")
        
        return stocks

    def scrape_fool_premium_stocks(self):
        """Navigate to Fool Premium Stocks page and extract content"""
        try:
            target_url = "https://www.fool.com/premium/my-stocks"
            self.page.goto(target_url, wait_until="domcontentloaded")
            self.wait_for_full_page_load()
            text_content = self.capture_page_content()
            if text_content:
                stocks_data = self.parse_stocks_from_text(text_content)
                output_folder = 'extracted_content'
                with open(os.path.join(output_folder, 'stocks_data.json'), 'w', encoding='utf-8') as f:
                    json.dump(stocks_data, f, indent=2)
            
            # Take a screenshot of the final page
            screenshot_path = os.path.join(output_folder, 'final_page_screenshot.png')
            self.page.screenshot(path=screenshot_path)
            self.logger.info(f"Screenshot saved to {screenshot_path}")
        except Exception as e:
            self.logger.error(f"Error navigating to Fool Premium Stocks page: {str(e)}")

    def close(self):
        """Clean up resources"""
        try:
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            self.logger.info("Browser session closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing browser session: {str(e)}")

def main():
    config = {
        'url': 'https://www.fool.com/premium',
        'credentials': {
            'username': 'calvinseamons35@gmail.com',
            'password': 'Ballislife35!'
        },
        'selectors': {
            'username_field': '#usernameOrEmail',
            'password_field': '#password',
            'login_button': 'button[type="submit"]'
        }
    }

    scraper = WebScraper(headless=True)

    try:
        scraper.start()
        
        if scraper.login(config['url'], config['credentials'], config['selectors']):
            scraper.scrape_fool_premium_stocks()
        else:
            print("Login failed")
        
    finally:
        scraper.close()

if __name__ == "__main__":
    main()
