#!/usr/bin/env python3
from app.scraper import scrape_article

# Test URLs
test_urls = [
    "https://www.bbc.com/news",
    "https://www.cnn.com/2024/01/01/politics/example-news",
    "https://www.ndtv.com/india-news",
    "https://timesofindia.indiatimes.com/india/news",
    "https://www.reuters.com/world/"
]

def test_scraper():
    for url in test_urls:
        print(f"\nTesting: {url}")
        try:
            result = scrape_article(url)
            if result:
                print(f"✅ Success ({result['method']})")
                print(f"Title: {result['title'][:100]}...")
                print(f"Text: {result['text'][:200]}...")
            else:
                print("❌ Failed to extract content")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_scraper()