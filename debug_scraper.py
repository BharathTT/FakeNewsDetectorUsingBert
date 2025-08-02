#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

try:
    from scraper import scrape_article
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're in the virtual environment and dependencies are installed")
    sys.exit(1)

if __name__ == "__main__":
    url = input("Enter URL to test: ")
    print(f"Testing URL: {url}")
    
    result = scrape_article(url)
    if result:
        print(f"✅ SUCCESS")
        print(f"Method: {result['method']}")
        print(f"Title: {result['title']}")
        print(f"Text length: {len(result['text'])}")
        print(f"Text preview: {result['text'][:300]}...")
    else:
        print("❌ FAILED - No content extracted")