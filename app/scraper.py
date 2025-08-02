from newspaper import Article
import requests
from bs4 import BeautifulSoup
import re

def scrape_article(url):
    """Extract article text from URL using multiple methods"""
    
    # Method 1: newspaper3k
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        if article.text and len(article.text.strip()) > 50:
            return {
                'text': article.text.strip(),
                'title': article.title or '',
                'method': 'newspaper3k'
            }
    except Exception as e:
        print(f"Newspaper3k failed: {e}")
    
    # Method 2: Enhanced BeautifulSoup
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
            element.decompose()
        
        # Extended selectors for different news sites
        selectors = [
            'article', '.article-content', '.story-body', '.post-content', 
            'main', '.content', '.entry-content', '.article-body',
            '.story-content', '.news-content', '.article-text',
            '[data-module="ArticleBody"]', '.story__body',
            '.article__content', '.post-body', '.content-body'
        ]
        
        text = ''
        title = ''
        
        # Try to find title
        title_selectors = ['h1', '.headline', '.article-title', '.story-headline', 'title']
        for sel in title_selectors:
            title_elem = soup.select_one(sel)
            if title_elem and title_elem.get_text(strip=True):
                title = title_elem.get_text(strip=True)
                break
        
        # Try to find article content
        for selector in selectors:
            content = soup.select_one(selector)
            if content:
                text = content.get_text(separator=' ', strip=True)
                if len(text) > 100:  # Ensure we have substantial content
                    break
        
        # Fallback: get all paragraphs with better filtering
        if not text or len(text) < 200:
            paragraphs = soup.find_all('p')
            valid_paragraphs = []
            for p in paragraphs:
                p_text = p.get_text(strip=True)
                # Filter out navigation, ads, etc.
                if (len(p_text) > 30 and 
                    not any(skip in p_text.lower() for skip in ['subscribe', 'follow us', 'share', 'advertisement', 'cookie', 'privacy policy'])):
                    valid_paragraphs.append(p_text)
            text = ' '.join(valid_paragraphs)
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) > 100:
            return {
                'text': text,
                'title': title or soup.find('title').get_text(strip=True) if soup.find('title') else '',
                'method': 'beautifulsoup'
            }
    except Exception as e:
        print(f"BeautifulSoup failed: {e}")
    
    # Method 3: Simple text extraction
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; bot)'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get all text and filter
        all_text = soup.get_text(separator=' ', strip=True)
        sentences = [s.strip() for s in all_text.split('.') if len(s.strip()) > 30]
        text = '. '.join(sentences[:20])  # First 20 sentences
        
        if len(text) > 100:
            return {
                'text': text,
                'title': soup.find('title').get_text(strip=True) if soup.find('title') else '',
                'method': 'simple'
            }
    except Exception as e:
        print(f"Simple extraction failed: {e}")
    
    return None