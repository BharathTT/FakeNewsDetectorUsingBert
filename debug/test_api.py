#!/usr/bin/env python3
import requests
import json

# Test statements
test_cases = [
    "Scientists discovered aliens on Mars yesterday",
    "The stock market closed higher today", 
    "COVID vaccines are 95% effective",
    "The moon is made of cheese"
]

def test_api():
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        health = requests.get(f"{base_url}/health")
        print("Health Check:", health.json())
    except:
        print("❌ API not running. Start with: make api")
        return
    
    # Test predictions
    for text in test_cases:
        try:
            response = requests.post(f"{base_url}/predict", json={"text": text})
            result = response.json()
            print(f"'{text}' → {result['label']} ({result['confidence']}) [{result.get('model', 'unknown')}]")
        except Exception as e:
            print(f"Error testing '{text}': {e}")

if __name__ == "__main__":
    test_api()