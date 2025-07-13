from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

app = FastAPI()

class TextRequest(BaseModel):
    text: str

# Global variables for models
baseline_model = None
vectorizer = None

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
        print("Model files not found")
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

# Load models on startup
model_loaded = load_baseline_model()

@app.post("/predict")
def predict(request: TextRequest):
    text = request.text.strip()
    if not text:
        return {"error": "Empty text provided"}
    
    result = predict_with_baseline(text)
    
    if result is None:
        # Fallback: simple keyword-based prediction
        fake_keywords = ['fake', 'false', 'lie', 'hoax', 'scam', 'fraud']
        real_keywords = ['true', 'fact', 'verified', 'confirmed', 'official']
        
        text_lower = text.lower()
        fake_score = sum(1 for word in fake_keywords if word in text_lower)
        real_score = sum(1 for word in real_keywords if word in text_lower)
        
        if fake_score > real_score:
            return {"label": "Fake", "confidence": 0.6}
        elif real_score > fake_score:
            return {"label": "Real", "confidence": 0.6}
        else:
            return {"label": "Uncertain", "confidence": 0.5}
    
    return result

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "baseline_loaded": baseline_model is not None
    }