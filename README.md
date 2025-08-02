# Fake News Detection System

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Dataset](#dataset)
5. [Installation](#installation)
6. [Data Preprocessing](#data-preprocessing)
7. [Modeling](#modeling)

   * [Model Selection](#model-selection)
   * [Training Pipeline](#training-pipeline)
   * [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Evaluation](#evaluation)
9. [Web Interface](#web-interface)
10. [API Reference](#api-reference)
11. [Deployment](#deployment)
12. [Performance Metrics](#performance-metrics)
13. [Future Work](#future-work)
14. [Contributing](#contributing)
15. [License](#license)

---

## Project Overview

The Fake News Detection System is an open-source, end-to-end platform to classify political statements and claims as **real** or **fake** using the renowned LIAR dataset. Utilizing free NLP libraries (Transformers, scikit-learn) and standard Python tooling, this system runs entirely on local or community-hosted infrastructure, avoiding paid APIs and services.

**Objectives:**

* Develop a fully free ML pipeline for fake news detection.
* Provide a lightweight web interface and REST API without paid hosting.
* Benchmark transformer-based and classical NLP models using the LIAR dataset from PolitiFact.

---

## Features

* **Open-Source Transformers**: Fine-tune BERT-like models via Hugging Face’s `transformers`.
* **Data Augmentation**: Apply back-translation with free libraries or synonym replacement without proprietary tools.
* **Model Comparison**: Evaluate BERT, DistilBERT, and logistic regression baselines.
* **Local Web UI**: Flask-based front end; works on any machine without external dependencies.
* **RESTful API**: Built with FastAPI or Flask-RESTful; no paid API Gateway needed.
* **Basic Logging**: Uses Python’s `logging` module and local log files for monitoring.

---

## Architecture

```plaintext
+-----------+      +-------------+      +------------+      +------------------+
|  User UI  | ---> |  API Layer  | ---> |  Model     | ---> |  Prediction      |
| (Flask)   |      | (FastAPI)   |      |  Inference |      |  & Logging       |
+-----------+      +-------------+      +------------+      +------------------+
```

**Project Structure:**
```
FakeNewsDetectorUsingBert/
├── app/                    # Main application code
│   ├── api.py             # FastAPI backend server
│   ├── web.py             # Flask web interface
│   ├── data.py            # Data processing & LIAR dataset
│   ├── train.py           # Model training (baseline + BERT)
│   ├── test.py            # Testing & validation
│   └── models/            # Trained models (created after training)
│       ├── full_bert/     # BERT model files
│       ├── baseline_model.pkl
│       └── vectorizer.pkl
├── liar_dataset/           # Dataset files (download required)
│   ├── train.tsv
│   ├── valid.tsv
│   └── test.tsv
├── requirements.txt        # Python dependencies
├── Makefile               # Easy commands
├── docker-compose.yml     # Docker deployment
├── Dockerfile             # Container configuration
├── README.md              # This file
└── .gitignore             # Git ignore rules
```

Components:

* **Frontend**: Flask with inline HTML (app/web.py)
* **Backend**: FastAPI prediction API (app/api.py)
* **Model**: BERT transformer + logistic regression baseline
* **Data**: LIAR dataset preprocessing (app/data.py)
* **Training**: Unified training pipeline (app/train.py)

---

## 🧰 Features to be Added

- URL-to-text conversion via `newspaper3k` or `BeautifulSoup`
- Classification for news headlines only
- Real-time API integration (Google News / NDTV RSS)
- Web interface with input for text, headline, and URL
- Explainability using **LIME** / **SHAP**
- Model comparison dashboard (accuracy, F1-score)
- Multilingual support (future scope)

---


## Dataset

**LIAR Dataset**: A benchmark dataset for fake news detection containing 12,836 human-labeled short statements from PolitiFact's API.

| Attribute | Details |
| --------- | ------- |
| **Source** | PolitiFact fact-checking website |
| **Size** | 12,836 statements |
| **Labels** | 6 fine-grained labels: pants-fire, false, barely-true, half-true, mostly-true, true |
| **Binary Labels** | Converted to fake (pants-fire, false, barely-true) vs real (half-true, mostly-true, true) |
| **Features** | Statement text, subject, speaker, job title, state, party, context |
| **Splits** | Train: 10,269, Validation: 1,284, Test: 1,283 |

**Dataset Structure:**
```
liar_dataset/
├── train.tsv
├── valid.tsv
└── test.tsv
```


**Preprocessing:**

1. Load TSV files with tab-separated columns.
2. Extract statement text (column 2) and labels (column 1).
3. Convert 6-class labels to binary (fake/real).
4. Clean text: remove special characters, normalize whitespace.
5. Handle class imbalance through weighted sampling.

---

## Installation & Setup

### **System Requirements**
* Python 3.8+ (3.9+ recommended)
* Git
* 4GB+ RAM
* 2GB+ free disk space
* (Optional) GPU with CUDA for faster training

### **Step 1: Clone Repository**
```bash
git clone https://github.com/your-username/FakeNewsDetectorUsingBert.git
cd FakeNewsDetectorUsingBert
```

### **Step 2: Setup Python Environment**

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### **Step 3: Download LIAR Dataset**
```bash
# Create dataset directory
mkdir -p liar_dataset

# Download dataset files
wget -O liar_dataset/train.tsv https://raw.githubusercontent.com/thiagocastroferreira/FakeNewsCorpus/master/liar_dataset/train.tsv
wget -O liar_dataset/valid.tsv https://raw.githubusercontent.com/thiagocastroferreira/FakeNewsCorpus/master/liar_dataset/valid.tsv
wget -O liar_dataset/test.tsv https://raw.githubusercontent.com/thiagocastroferreira/FakeNewsCorpus/master/liar_dataset/test.tsv
```

**Alternative (Manual Download):**
1. Go to [LIAR Dataset Repository](https://github.com/thiagocastroferreira/FakeNewsCorpus)
2. Download `train.tsv`, `valid.tsv`, `test.tsv`
3. Place files in `liar_dataset/` folder

### **Step 4: Train Models**
```bash
# Train baseline model (2-3 minutes)
make baseline

# Train BERT model (30-60 minutes)
make train

# Or train both
python app/train.py
```

## Usage

### **Quick Start (After Setup)**

#### **1. Verify Installation**
```bash
# Test data loading and models
make test
```
**Expected Output:**
```
Testing data loading...
Loaded 10240 training samples
Baseline Results: ~60% accuracy
```

#### **2. Start API Server**
```bash
# Method 1: Using Makefile
make api

# Method 2: Direct command
cd app && uvicorn api:app --reload --port 8000
```
**API will be available at:** http://localhost:8000

#### **3. Start Web Interface**
```bash
# In a new terminal (keep API running)
make web

# Or directly
cd app && python web.py
```
**Web interface:** http://localhost:5000

#### **4. Test API Endpoint**
```bash
# Test with curl
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Scientists discovered aliens on Mars yesterday"}'

# Expected response
{"label": "Fake", "confidence": 0.85}
```

### **Manual Testing**
Try these example statements in the web interface:
- ✅ **Real**: "The stock market closed higher today"
- ❌ **Fake**: "Aliens landed in New York yesterday"
- ✅ **Real**: "COVID vaccines are 95% effective"
- ❌ **Fake**: "The moon is made of cheese"

### **Model Performance**
- **Baseline Model**: 60.54% accuracy
- **BERT Model**: 61.80% accuracy
- **Training Time**: 30-60 minutes (BERT)
- **Inference Time**: ~100ms per prediction

---

## Data Preprocessing

All data processing is handled in `app/data.py`:

```python
from app.data import load_liar_dataset, preprocess_liar, LiarDataset

# Load and preprocess
train_df = preprocess_liar(load_liar_dataset('liar_dataset/train.tsv'))
val_df = preprocess_liar(load_liar_dataset('liar_dataset/valid.tsv'))
test_df = preprocess_liar(load_liar_dataset('liar_dataset/test.tsv'))

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(list(train_df['statement']), padding=True, truncation=True, max_length=128)
```

* LIAR statements are typically short (avg ~18 words), so max_length=128 is sufficient.
* Binary classification simplifies the 6-class problem while maintaining meaningful distinction.
* All preprocessing functions are consolidated in a single module.

---
## 🧩 Hybrid Model (Planned)

> Combine BERT embeddings with an SVM classifier:

```python
from transformers import BertTokenizer, BertModel
from sklearn.svm import SVC

# Extract BERT embeddings for headlines
embeddings = get_bert_embeddings(texts)

# Train SVM on those embeddings
clf = SVC(kernel='linear')
clf.fit(embeddings, labels)
```

---

## 🔍 Explainability: LIME / SHAP

- **LIME**: Local Interpretable Model-agnostic Explanations – shows which words influenced classification.
- **SHAP**: SHapley Additive exPlanations – shows feature contributions per prediction.

> These will be used for model transparency.

---

## Modeling

### Model Selection

| Model                        | Size | Notes                           |
| ---------------------------- | ---- | ------------------------------- |
| `bert-base-uncased`          | 110M | Good baseline                   |
| `distilbert-base-uncased`    | 66M  | Faster inference                |
| `sklearn.LogisticRegression` | \~1M | Simple, fast classical baseline |

### Training Pipeline

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np

# Handle class imbalance in LIAR dataset
class_weights = compute_class_weight('balanced', classes=np.unique(train_df['binary_label']), 
                                   y=train_df['binary_label'])
class_weights = torch.tensor(class_weights, dtype=torch.float)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
args = TrainingArguments(
    output_dir='./models/liar_bert',
    num_train_epochs=4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
```

### Hyperparameter Tuning

**Optimized for LIAR Dataset:**
* Learning rates: `[2e-5, 3e-5, 5e-5]`
* Epochs: `[3, 4, 5]` (LIAR benefits from more epochs due to complexity)
* Batch sizes: `[16, 32]` depending on GPU memory
* Max sequence length: `128` (optimal for LIAR's short statements)
* Weight decay: `[0.01, 0.1]` for regularization

---

## Evaluation

Evaluation is built into the training pipeline:

```bash
# Test everything
make test
# OR
python app/test.py

# Train with automatic evaluation
make train
```

**Metrics reported:**
* **Accuracy** - Overall correctness
* **Precision** - True positives / (True positives + False positives)
* **Recall** - True positives / (True positives + False negatives)
* **F1-Score** - Harmonic mean of precision and recall

**Expected Performance:**
* Baseline (Logistic Regression): ~62% accuracy
* BERT: ~75-80% accuracy

---

## Web Interface

1. **Run API**:
   ```bash
   make api
   # OR
   uvicorn app.api:app --reload --port 8000
   ```

2. **Run Frontend**:
   ```bash
   make web
   # OR
   python app/web.py
   ```

3. **Open browser**: http://localhost:5000

The web interface is a single file (`app/web.py`) with inline HTML for maximum simplicity.

---

## API Reference

### **Base URL**
- **Local**: http://localhost:8000
- **Docker**: http://localhost:8000

### **Endpoints**

#### **POST /predict**
Classify a statement as fake or real news.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your statement here"}'
```

**Request Body:**
```json
{
  "text": "The president announced new policies today"
}
```

**Response:**
```json
{
  "label": "Real",
  "confidence": 0.87
}
```

**Response Fields:**
- `label`: "Fake" or "Real"
- `confidence`: Float between 0.0-1.0 (higher = more confident)

#### **GET /health**
Check API health status.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy"
}
```

#### **GET /docs**
Interactive API documentation (Swagger UI).

**URL**: http://localhost:8000/docs

### **Example Usage**

#### **Python**
```python
import requests

response = requests.post(
    'http://localhost:8000/predict',
    json={'text': 'Scientists discovered aliens on Mars'}
)
result = response.json()
print(f"Label: {result['label']}, Confidence: {result['confidence']}")
```

#### **JavaScript**
```javascript
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({text: 'The economy is growing rapidly'})
})
.then(response => response.json())
.then(data => console.log(data));
```

#### **cURL Examples**
```bash
# Test fake news
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Aliens invaded Earth yesterday"}'

# Test real news
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Stock market reached new highs"}'
```

### **Error Responses**

**400 Bad Request:**
```json
{
  "detail": "Text field is required"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Model prediction failed"
}
```

---

## Deployment

### **Docker Deployment (Recommended)**

#### **Prerequisites**
- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed

#### **Quick Deploy**
```bash
# Build and start all services
docker-compose up --build

# Or use Makefile
make docker
```

#### **Access Services**
- **API**: http://localhost:8000
- **Web Interface**: http://localhost:5000
- **API Docs**: http://localhost:8000/docs

#### **Production Deploy**
```bash
# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### **Manual Deployment**

#### **Local Development**
```bash
# Terminal 1: Start API
source venv/bin/activate
make api

# Terminal 2: Start Web Interface
source venv/bin/activate
make web
```

#### **Production Server**
```bash
# Install production server
pip install gunicorn

# Run API with Gunicorn
cd app && gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app --bind 0.0.0.0:8000

# Run Web with Gunicorn
cd app && gunicorn -w 2 web:app --bind 0.0.0.0:5000
```

### **Cloud Deployment**

#### **Free Hosting Options**
- **Render.com**: Connect GitHub repo, auto-deploy
- **Railway.app**: One-click deploy
- **Heroku**: Use provided Dockerfile
- **Google Cloud Run**: Serverless container deployment

#### **Environment Variables**
```bash
# Optional configurations
export MODEL_PATH="./models/full_bert"
export MAX_LENGTH=128
export BATCH_SIZE=32
```

---

## Performance Metrics

| Metric     | Observed                      |
| ---------- | ----------------------------- |
| Latency    | \~100ms                       |
| Throughput | \~50 req/s                    |
| Uptime     | Dependant on host (aim \~99%) |

## Troubleshooting

### **Common Issues & Solutions**

#### **1. Installation Issues**

**Problem**: `pip: command not found`
```bash
# Solution: Use pip3 or python -m pip
pip3 install -r requirements.txt
# OR
python3 -m pip install -r requirements.txt
```

**Problem**: `externally-managed-environment`
```bash
# Solution: Use virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### **2. Dataset Issues**

**Problem**: `FileNotFoundError: liar_dataset/train.tsv`
```bash
# Solution: Download dataset manually
mkdir liar_dataset
# Download files from GitHub or use wget commands above
```

**Problem**: Dataset download fails
```bash
# Alternative: Manual download
# 1. Go to https://github.com/thiagocastroferreira/FakeNewsCorpus
# 2. Download train.tsv, valid.tsv, test.tsv
# 3. Place in liar_dataset/ folder
```

#### **3. Training Issues**

**Problem**: CUDA/GPU errors
```bash
# Solution: Force CPU training
export CUDA_VISIBLE_DEVICES=""
make train
```

**Problem**: Out of memory during training
```bash
# Solution: Reduce batch size in app/train.py
# Change: per_device_train_batch_size=16 → 8
# Change: per_device_eval_batch_size=32 → 16
```

**Problem**: Training takes too long
```bash
# Solution: Train baseline only (2-3 minutes)
make baseline

# Or reduce BERT epochs in app/train.py
# Change: num_train_epochs=4 → 2
```

#### **4. Runtime Issues**

**Problem**: Port already in use
```bash
# Solution: Use different ports
cd app && uvicorn api:app --port 8001
cd app && python web.py  # Edit port in web.py
```

**Problem**: API not responding
```bash
# Check if API is running
curl http://localhost:8000/health

# Restart API
make api
```

**Problem**: Models not found
```bash
# Solution: Retrain models
make train

# Or check if models directory exists
ls -la app/models/
```

### **Performance Issues**

**Slow predictions:**
- Use baseline model for faster inference
- Reduce max_length in tokenization
- Use CPU for small batches

**Low accuracy:**
- Ensure full dataset is downloaded
- Train for more epochs
- Check data preprocessing

### **Quick Diagnostic Commands**
```bash
# Test everything
make test

# Check Python environment
which python
pip list | grep -E "torch|transformers|sklearn"

# Check data loading
python -c "from app.data import *; print('✅ Data loading works')"

# Check model files
ls -la app/models/

# Test API health
curl http://localhost:8000/health
```

### **Getting Help**

1. **Check logs**: Look for error messages in terminal output
2. **Verify setup**: Run `make test` to check all components
3. **Check requirements**: Ensure all dependencies are installed
4. **Try Docker**: Use `make docker` for isolated environment
5. **Reset environment**: Delete `venv/` and recreate

### **System Requirements Check**
```bash
# Check Python version (need 3.8+)
python --version

# Check available memory (need 4GB+)
free -h  # Linux
top -l 1 | grep PhysMem  # macOS

# Check disk space (need 2GB+)
df -h .
```

---

## Commands Reference

### **Makefile Commands**
```bash
make install    # Install dependencies
make test      # Run all tests
make baseline  # Train baseline model only (fast)
make train     # Train all models (baseline + BERT)
make api       # Start API server
make web       # Start web interface
make docker    # Deploy with Docker
make clean     # Clean cache files
```

### **Direct Commands**
```bash
# Training
cd app && python train.py              # Train all models
cd app && python -c "from train import train_baseline; train_baseline()"  # Baseline only

# Testing
cd app && python test.py               # Run tests

# Servers
cd app && uvicorn api:app --reload --port 8000    # API server
cd app && python web.py                           # Web server

# Docker
docker-compose up --build              # Build and run
docker-compose up -d                   # Run in background
docker-compose logs -f                 # View logs
docker-compose down                    # Stop services
```

---

## Model Information

### **Dataset: LIAR**
- **Source**: PolitiFact fact-checking website
- **Size**: 12,836 human-labeled statements
- **Labels**: 6-class converted to binary (Fake/Real)
- **Features**: Statement text, speaker, subject, context
- **Splits**: Train (10,269), Validation (1,284), Test (1,283)

### **Models**

#### **Baseline Model**
- **Algorithm**: Logistic Regression
- **Features**: TF-IDF with n-grams (1,2)
- **Accuracy**: ~60.5%
- **Training Time**: 2-3 minutes
- **File Size**: ~1MB

#### **BERT Model**
- **Architecture**: BERT-base-uncased
- **Parameters**: 110M
- **Accuracy**: ~61.8%
- **Training Time**: 30-60 minutes
- **File Size**: ~400MB

### **Performance Metrics**
| Metric | Baseline | BERT |
|--------|----------|------|
| Accuracy | 60.54% | 61.80% |
| Precision (Fake) | 54.67% | 56.47% |
| Recall (Fake) | 56.06% | 54.43% |
| F1-Score (Fake) | 55.36% | 55.43% |
| Precision (Real) | 65.29% | 65.67% |
| Recall (Real) | 64.01% | 67.51% |
| F1-Score (Real) | 64.64% | 66.57% |

---

## Future Enhancements

### **Model Improvements**
- [ ] RoBERTa/DeBERTa models for better accuracy
- [ ] Ensemble methods combining multiple models
- [ ] Feature engineering with speaker/context data
- [ ] Data augmentation techniques

### **System Features**
- [ ] Real-time news feed analysis
- [ ] Browser extension for fact-checking
- [ ] Multilingual support
- [ ] Confidence explanation/reasoning
- [ ] User feedback integration

### **Technical Improvements**
- [ ] Model quantization for faster inference
- [ ] Caching for repeated queries
- [ ] Batch prediction API
- [ ] Monitoring and logging
- [ ] A/B testing framework

---

## Contributing

### **How to Contribute**
1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### **Development Setup**
```bash
# Clone your fork
git clone https://github.com/your-username/FakeNewsDetectorUsingBert.git
cd FakeNewsDetectorUsingBert

# Setup development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
make test
```

### **Code Style**
- Follow PEP 8 for Python code
- Add docstrings for functions
- Include tests for new features
- Keep functions small and focused

### **Reporting Issues**
- Use GitHub Issues
- Include system information
- Provide reproduction steps
- Add error logs if applicable

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **LIAR Dataset**: Wang, William Yang. "Liar, liar pants on fire": A new benchmark dataset for fake news detection. ACL 2017.
- **BERT Model**: Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019.
- **Transformers Library**: Hugging Face Transformers
- **Dataset Source**: PolitiFact.com

---

## Contact

For questions or support:
- **GitHub Issues**: [Create an issue](https://github.com/your-username/FakeNewsDetectorUsingBert/issues)
- **Email**: your-email@example.com
- **Documentation**: This README and code comments

---

**⭐ Star this repository if you found it helpful!**
