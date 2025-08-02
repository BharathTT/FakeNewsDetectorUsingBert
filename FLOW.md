# System Flow Documentation

## 1. Data Flow
```
LIAR Dataset → Preprocessing → Binary Labels
├── pants-fire, false, barely-true → Fake (0)
└── half-true, mostly-true, true → Real (1)
```

## 2. Training Flow
```
Text Statements → BERT Tokenizer → BERT Model → [CLS] Embeddings → Random Forest → Trained Model
```

## 3. Prediction Flow
```
User Input → API → Model Priority:
├── 1st: Hybrid (BERT + Random Forest)
├── 2nd: Baseline (TF-IDF + Logistic Regression)  
└── 3rd: Keyword Matching
```

## 4. API Response Flow
```
Text → Embedding → Prediction → {"label": "Fake/Real", "confidence": 0.85, "model": "hybrid"}
```

## 5. Architecture Flow
```
Flask Web UI → FastAPI Backend → Model Inference → Response
     ↓              ↓                    ↓
  User Input → JSON Request → BERT Embeddings → Classification
```

## 6. Deployment Flow
```
Docker Compose → API Container (Port 8000) + Web Container (Port 5000)
```

## Key Steps:
1. **Load** → Models loaded at startup
2. **Input** → User submits text via web/API
3. **Process** → Text → BERT embeddings → Random Forest
4. **Predict** → Binary classification (Fake/Real)
5. **Return** → JSON response with confidence score

The system prioritizes accuracy (hybrid model) with robust fallbacks for reliability.