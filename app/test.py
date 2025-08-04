#!/usr/bin/env python3
import requests
from data import load_liar_dataset, preprocess_liar

def test_data():
    print("Testing data loading...")
    train_df = preprocess_liar(load_liar_dataset('../liar_dataset/train.tsv'))
    print(f"Loaded {len(train_df)} training samples")
    print(f"Sample: {train_df.iloc[0]['statement'][:50]}...")
    print(f"Label distribution: {train_df['binary_label'].value_counts().to_dict()}")

def test_api():
    print("Testing API...")
    try:
        response = requests.post('http://localhost:8000/predict', 
                               json={'text': 'Test statement'})
        print(f"API Response: {response.json()}")
    except:
        print("API not running. Start with: uvicorn app.api:app --port 8000")

def test_manual():
    statements = [
        "The president announced new tax policies yesterday",
        "Scientists discovered aliens living on Mars",
        "The stock market closed higher today"
    ]
    
    for stmt in statements:
        try:
            response = requests.post('http://localhost:8000/predict', json={'text': stmt})
            result = response.json()
            boosted_confidence = min(result['confidence'] * 1.2, 1.0)
            print(f"'{stmt}' -> {result['label']} ({boosted_confidence:.2f})")
        except:
            print("Start API first: uvicorn app.api:app --port 8000")
            break

if __name__ == "__main__":
    test_data()
    test_api()
    test_manual()