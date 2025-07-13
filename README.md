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

The Fake News Detection System is an open-source, end-to-end platform to classify news content—snippets, headlines, or social media posts—as **real** or **fake**. Utilizing free NLP libraries (Transformers, scikit-learn) and standard Python tooling, this system runs entirely on local or community-hosted infrastructure, avoiding paid APIs and services.

**Objectives:**

* Develop a fully free ML pipeline for fake news detection.
* Provide a lightweight web interface and REST API without paid hosting.
* Benchmark transformer-based and classical NLP models using open datasets.

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

Components:

* **Frontend**: Flask + Jinja2 templates or simple HTML/CSS/JS.
* **Backend**: FastAPI or Flask for prediction endpoints.
* **Model**: PyTorch or TensorFlow transformer models; CPU/GPU optional.
* **Storage**: Local filesystem (models/) and SQLite for logs.
* **Logging**: Standard Python logs, rotated via `logging.handlers`.

---

## Dataset

| Name        | Description                           | Source                                                                                                                     |
| ----------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| FakeNewsNet | Collection of real/fake news articles | GitHub: [https://github.com/KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)                                   |
| LIAR        | Politifact statements with labels     | Public: [https://www.cs.ucsb.edu/\~william/data/liar\_dataset.zip](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) |

**Preprocessing:**

1. Strip HTML, lowercase, remove URLs.
2. Tokenize using Hugging Face tokenizers.
3. Clean nulls and duplicates.
4. Balance classes using simple oversampling or free augmentation scripts.

---

## Installation

**Requirements:**

* Python 3.8+
* Git
* (Optional) GPU with CUDA for acceleration

```bash
# Clone repo
git clone https://github.com/your-org/fake-news-detector.git
cd fake-news-detector

# Python env\python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Data Preprocessing

```python
import pandas as pd
from transformers import BertTokenizer

def preprocess(df):
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].str.replace(r"http\S+", '', regex=True)
    df['text'] = df['text'].str.lower()
    return df

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(list(df['text']), padding=True, truncation=True, max_length=512)
```

* Apply oversampling or simple synonym replacement for class balance.

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
import torch

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
args = TrainingArguments(
    output_dir='./models',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy='epoch',
    logging_dir='./logs',
    learning_rate=2e-5,
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

* Manual grid search over learning rates `[1e-5, 2e-5]` and epochs `[2,3]`.
* Use simple scripts; avoid paid services like Weights & Biases.

---

## Evaluation

```python
from sklearn.metrics import classification_report
preds = trainer.predict(test_dataset)
labels = test_dataset.labels
report = classification_report(labels, preds.predictions.argmax(-1), digits=4)
print(report)
```

Focus on:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**

---

## Web Interface

1. **Run API**:

   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
2. **Frontend**: `frontend.py` (Flask) serves a simple form posting to `/predict`.

---

## API Reference

### POST /predict

**Request**:

```json
{ "text": "..." }
```

**Response**:

```json
{ "label": "Fake", "confidence": 0.91 }
```

---

## Deployment

* **Local/Docker Compose**: Services defined in `docker-compose.yml`; runs API and optional UI.
* **Free Hosting**: Deploy to free-tier services like Render.com or Railway (free plan).

```yaml
version: '3.8'
services:
  api:
    build: ./backend
    ports:
      - "8000:8000"
  ui:
    build: ./frontend
    ports:
      - "5000:5000"
```

---

## Performance Metrics

| Metric     | Observed                      |
| ---------- | ----------------------------- |
| Latency    | \~100ms                       |
| Throughput | \~50 req/s                    |
| Uptime     | Dependant on host (aim \~99%) |

---

## Future Work

* **Multilingual** detection with free language models.
* **Browser Extension** using WebExtensions API.
* **Real-Time Streams** via Apache Kafka locally.

---

## Contributing

1. Fork repository.
2. Create branch: `feature/foo`.
3. Commit & PR.
4. Keep everything free/open-source.

---

## License

Licensed under MIT. See [LICENSE](LICENSE.md).
