from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pickle
import os
def scrape_article(url):
    try:
        import requests
        import re
        
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        html = response.text
        
        # Extract title
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
        title = title_match.group(1) if title_match else ''
        
        # Extract paragraphs
        p_matches = re.findall(r'<p[^>]*>([^<]+)</p>', html, re.IGNORECASE)
        text = ' '.join([p.strip() for p in p_matches if len(p.strip()) > 20])
        text = re.sub(r'\s+', ' ', text).strip()
        
        print(f"Extracted {len(text)} chars: {text[:100]}...")
        
        if len(text) > 50:
            return {
                'text': text,
                'title': title,
                'method': 'regex'
            }
        
        return None
    except Exception as e:
        print(f"Scraper error: {e}")
        return None

app = FastAPI()

class TextRequest(BaseModel):
    text: str
    model: str = 'auto'

class URLRequest(BaseModel):
    url: str
    model: str = 'auto'

# Global variables for models
baseline_model = None
vectorizer = None
hybrid_model = None
bert_tokenizer = None
bert_model = None

def load_baseline_model():
    global baseline_model, vectorizer
    
    if os.path.exists('./models/baseline_model.pkl') and os.path.exists('./models/vectorizer.pkl'):
        try:
            with open('./models/baseline_model.pkl', 'rb') as f:
                baseline_model = pickle.load(f)
            with open('./models/vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            print("Baseline model loaded successfully")
            return True
        except Exception as e:
            print(f"Failed to load baseline model: {e}")
            return False
    else:
        print("Baseline model files not found")
        return False

def load_hybrid_model():
    global hybrid_model, bert_tokenizer, bert_model
    
    if (os.path.exists('./models/hybrid_model.pkl') and 
        os.path.exists('./models/bert_tokenizer') and 
        os.path.exists('./models/bert_embedder')):
        try:
            with open('./models/hybrid_model.pkl', 'rb') as f:
                hybrid_model = pickle.load(f)
            bert_tokenizer = BertTokenizer.from_pretrained('./models/bert_tokenizer')
            bert_model = BertModel.from_pretrained('./models/bert_embedder')
            print("Hybrid model loaded successfully")
            return True
        except Exception as e:
            print(f"Failed to load hybrid model: {e}")
            return False
    else:
        print("Hybrid model files not found")
        return False

def predict_with_baseline(text):
    if baseline_model is None or vectorizer is None:
        return None
    
    text_vector = vectorizer.transform([text])
    prediction = baseline_model.predict(text_vector)[0]
    probabilities = baseline_model.predict_proba(text_vector)[0]
    confidence = float(max(probabilities))
    
    label = "Real" if prediction == 1 else "Fake"
    return {"label": label, "confidence": round(confidence, 3)}

def predict_with_hybrid(text):
    if hybrid_model is None or bert_tokenizer is None or bert_model is None:
        return None
    
    # Get BERT embedding
    bert_model.eval()
    with torch.no_grad():
        inputs = bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy().reshape(1, -1)
    
    # Predict with Random Forest
    prediction = hybrid_model.predict(embedding)[0]
    probabilities = hybrid_model.predict_proba(embedding)[0]
    confidence = float(max(probabilities))
    
    label = "Real" if prediction == 1 else "Fake"
    return {"label": label, "confidence": round(confidence, 3), "model": "hybrid"}

# Load models on startup
baseline_loaded = load_baseline_model()
hybrid_loaded = load_hybrid_model()

def predict_with_keywords(text):
    fake_keywords = ['fake', 'false', 'lie', 'hoax', 'scam', 'fraud']
    real_keywords = ['true', 'fact', 'verified', 'confirmed', 'official']
    
    text_lower = text.lower()
    fake_score = sum(1 for word in fake_keywords if word in text_lower)
    real_score = sum(1 for word in real_keywords if word in text_lower)
    
    if fake_score > real_score:
        return {"label": "Fake", "confidence": 0.6, "model": "keyword"}
    elif real_score > fake_score:
        return {"label": "Real", "confidence": 0.6, "model": "keyword"}
    else:
        return {"label": "Uncertain", "confidence": 0.5, "model": "keyword"}

@app.post("/predict")
def predict(request: TextRequest):
    text = request.text.strip()
    model_choice = request.model
    if not text:
        return {"error": "Empty text provided"}
    
    # Model selection logic
    if model_choice == 'hybrid':
        result = predict_with_hybrid(text)
        if result is None:
            return {"error": "Hybrid model not available"}
        return result
    elif model_choice == 'baseline':
        result = predict_with_baseline(text)
        if result is None:
            return {"error": "Baseline model not available"}
        result["model"] = "baseline"
        return result
    elif model_choice == 'keyword':
        return predict_with_keywords(text)
    else:  # auto
        # Try hybrid model first
        result = predict_with_hybrid(text)
        if result is not None:
            return result
        
        # Fallback to baseline
        result = predict_with_baseline(text)
        if result is not None:
            result["model"] = "baseline"
            return result
        
        # Final fallback: keyword
        return predict_with_keywords(text)

@app.post("/predict_url")
def predict_url(request: URLRequest):
    url = request.url.strip()
    model_choice = request.model
    if not url:
        return {"error": "Empty URL provided"}
    
    # Scrape article
    try:
        print(f"Attempting to scrape: {url}")
        article_data = scrape_article(url)
        print(f"Scrape result: {article_data}")
        if not article_data:
            return {"error": "Failed to extract article content from URL"}
    except Exception as e:
        print(f"Scraping error: {str(e)}")
        return {"error": f"Scraping failed: {str(e)}"}
    
    text = article_data['text']
    
    # Model selection logic
    if model_choice == 'hybrid':
        result = predict_with_hybrid(text)
        if result is None:
            return {"error": "Hybrid model not available"}
    elif model_choice == 'baseline':
        result = predict_with_baseline(text)
        if result is None:
            return {"error": "Baseline model not available"}
        result["model"] = "baseline"
    elif model_choice == 'keyword':
        result = predict_with_keywords(text)
    else:  # auto
        # Try hybrid model first
        result = predict_with_hybrid(text)
        if result is not None:
            pass
        else:
            # Fallback to baseline
            result = predict_with_baseline(text)
            if result is not None:
                result["model"] = "baseline"
            else:
                # Final fallback: keyword
                result = predict_with_keywords(text)
    
    # Add article metadata
    result.update({
        "title": article_data['title'],
        "scraper_method": article_data['method'],
        "text_preview": text[:200] + "..." if len(text) > 200 else text,
        "full_text": text
    })
    return result

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "baseline_loaded": baseline_model is not None,
        "hybrid_loaded": hybrid_model is not None
    }