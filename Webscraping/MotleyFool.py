from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import json
import logging
from datetime import datetime
import os
from typing import Optional, Dict, Any
import time

class WebScraper:
    def __init__(self, headless: bool = True, slow_mo: int = 100):
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
                slow_mo=self.slow_mo
            )
            self.context = self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            self.page = self.context.new_page()
            
            # Add event listeners for network and load states
            self.page.on("load", lambda: self.logger.info("Page load completed"))
            self.page.on("domcontentloaded", lambda: self.logger.info("DOM content loaded"))
            
            self.logger.info("Browser session started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start browser session: {str(e)}")
            raise

    def wait_for_page_load(self):
        """Wait for various page load states"""
        try:
            # Wait for network to be idle (no requests for 500ms)
            self.page.wait_for_load_state("networkidle", timeout=30000)
            
            # Wait for DOM content to be loaded
            self.page.wait_for_load_state("domcontentloaded")
            
            # Wait for page to be completely loaded
            self.page.wait_for_load_state("load")
            
            # Additional wait for any JavaScript frameworks
            time.sleep(2)  # Short delay for any final renders
            
        except PlaywrightTimeout:
            self.logger.warning("Timeout while waiting for page to load completely")
        except Exception as e:
            self.logger.error(f"Error waiting for page load: {str(e)}")

    def wait_for_network_idle(self):
        """Wait for network to become idle"""
        try:
            self.page.wait_for_load_state("networkidle", timeout=30000)
        except PlaywrightTimeout:
            self.logger.warning("Timeout while waiting for network to be idle")

    def login(self, url: str, credentials: Dict[str, str], 
              selectors: Dict[str, str], success_indicator: str) -> bool:
        try:
            # Navigate to login page with wait until network idle
            self.logger.info(f"Navigating to {url}")
            self.page.goto(url, wait_until="networkidle")
            
            # Wait for complete page load
            self.wait_for_page_load()
            
            # Ensure login form is fully loaded
            self.logger.info("Waiting for login form...")
            self.page.wait_for_selector(selectors['username_field'], 
                                      state='visible', 
                                      timeout=20000)
            
            # Additional wait to ensure form is interactive
            time.sleep(2)
            
            # Fill credentials with explicit waits
            self.page.fill(selectors['username_field'], credentials['username'])
            self.page.fill(selectors['password_field'], credentials['password'])
            
            # Click login and wait for navigation
            self.logger.info("Attempting login...")
            with self.page.expect_navigation(wait_until="networkidle"):
                self.page.click(selectors['login_button'])
            
            # Wait for page load after login
            self.wait_for_page_load()

            self.capture_page_content()
            
            # Check for login success
            try:
                self.logger.info("Checking for successful login...")
                self.page.wait_for_selector(success_indicator, timeout=30000)
                self.logger.info("Login successful")
                return True
            except PlaywrightTimeout:
                self.logger.error("Login success indicator not found")
                return False
            
        except PlaywrightTimeout:
            self.logger.error("Timeout while trying to log in")
            return False
        except Exception as e:
            self.logger.error(f"Login failed: {str(e)}")
            return False

    def scrape_content(self, selectors: Dict[str, str]) -> Dict[str, Any]:
        data = {}
        try:
            # Wait for page to be fully loaded before scraping
            self.wait_for_page_load()
            
            for key, selector in selectors.items():
                try:
                    # Wait for each element with increased timeout
                    self.logger.info(f"Waiting for element: {key}")
                    
                    # First wait for element to exist in DOM
                    self.page.wait_for_selector(selector, timeout=20000)
                    
                    # Then wait for element to be visible
                    self.page.wait_for_selector(selector, 
                                              state='visible', 
                                              timeout=20000)
                    
                    # Get text content
                    element = self.page.query_selector(selector)
                    if element:
                        data[key] = element.inner_text()
                        self.logger.info(f"Successfully scraped {key}")
                    else:
                        data[key] = None
                        self.logger.warning(f"Element found but no text content for: {key}")
                
                except PlaywrightTimeout:
                    self.logger.warning(f"Timeout waiting for element: {key}")
                    data[key] = None
            
            return data
        except Exception as e:
            self.logger.error(f"Error scraping content: {str(e)}")
            return data

    def save_data(self, data: Dict[str, Any], filename: str):
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{filename}_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                
            self.logger.info(f"Data saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")

    def capture_page_content(self):
        """Add this method to your WebScraper class"""
        try:
            # Get full HTML
            html_content = self.page.content()
            
            # Save HTML to file
            with open('page_content.html', 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Get all text content
            text_content = self.page.evaluate('''() => {
                return document.body.innerText;
            }''')
            
            # Save text to file
            with open('text_content.txt', 'w', encoding='utf-8') as f:
                f.write(text_content)
                
            # Get specific element structures
            elements_info = self.page.evaluate('''() => {
                function getElementInfo(element, depth = 0) {
                    const info = {
                        tag: element.tagName.toLowerCase(),
                        id: element.id,
                        classes: Array.from(element.classList),
                        attributes: {},
                        text: element.innerText.slice(0, 100), // First 100 chars
                        children: []
                    };
                    
                    // Get all attributes
                    for (const attr of element.attributes) {
                        info.attributes[attr.name] = attr.value;
                    }
                    
                    // Get children if not too deep
                    if (depth < 3) {  // Limit depth to avoid too much data
                        for (const child of element.children) {
                            info.children.push(getElementInfo(child, depth + 1));
                        }
                    }
                    
                    return info;
                }
                
                return getElementInfo(document.body);
            }''')
            
            # Save structured info to JSON
            with open('page_structure.json', 'w', encoding='utf-8') as f:
                json.dump(elements_info, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error capturing page content: {str(e)}")

    def close(self):
        try:
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            self.logger.info("Browser session closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing browser session: {str(e)}")

    def screenshot(self, path: str):
        try:
            self.page.screenshot(path=path)
            self.logger.info(f"Screenshot saved to {path}")
        except Exception as e:
            self.logger.error(f"Error taking screenshot: {str(e)}")

def main():
    # Configuration
    config = {
        'url': 'https://www.fool.com/premium',
        'credentials': {
            'username': 'calvinseamons35@gmail.com',
            'password': 'Ballislife35!'
        },
        'selectors': {
            'username_field': '#usernameOrEmail',
            'password_field': '#password',
            'login_button': 'button[type="submit"]',
            'content': {
                'title': '.cursor-pointer.inline-block',  # Fixed selector format
                'price': '[data-uw-rm-sr=""]'  # Added price selector
            }
        },
        'success_indicator': '.header-user-menu'  # Adjust this to match actual dashboard indicator
    }

    # Initialize scraper with longer delay for dynamic content
    scraper = WebScraper(headless=False, slow_mo=200)

    try:
        scraper.start()
        
        if scraper.login(
            config['url'],
            config['credentials'],
            config['selectors'],
            config['success_indicator']
        ):
            # Wait for everything to load
            scraper.wait_for_page_load()
            time.sleep(5)  # Give extra time for dynamic content
            
            # Capture the page content - this will create page_content.html and page_structure.json
            # scraper.capture_page_content()
            
    finally:
        scraper.close()

if __name__ == "__main__":
    main()