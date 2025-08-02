# 🔍 Fake News Detection System

## ⚡ Quick Start

### **🚀 5-Minute Setup**
```bash
# 1. Clone & Setup
git clone https://github.com/BharathTT/FakeNewsDetectorUsingBert.git
cd FakeNewsDetectorUsingBert
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Train Models (2-3 minutes)
make baseline

# 3. Run System
make api     # Terminal 1
make web     # Terminal 2 (new terminal)

# 4. Test at http://localhost:5001
```

### **🧪 Quick Test**
1. Go to http://localhost:5001
2. Select "Hybrid" model from dropdown
3. Enter: "Scientists found aliens on Mars"
4. Click "🔍 Analyze Statement"
5. Should show: ❌ **Fake** (85% confidence)

### **🔧 Test Files & Commands**
```bash
# Test scraper functionality
python3 debug_scraper.py
# Enter URL when prompted: https://www.bbc.com/news

# Test API directly
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Breaking news from Mars", "model": "hybrid"}'

# Test URL analysis
curl -X POST "http://localhost:8000/predict_url" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://www.ndtv.com/india-news/...", "model": "auto"}'

# Run all tests
make test
```

### **📋 Test Examples**
- **Real News**: "The stock market closed higher today"
- **Fake News**: "Aliens landed in New York yesterday"
- **Test URLs**: 
  - https://www.bbc.com/news
  - https://www.ndtv.com/india-news/...
  - https://www.businessworld.in/article/...

### **🐳 Docker (Even Easier)**
```bash
make docker  # One command runs everything
# Web: http://localhost:5000 | API: http://localhost:8000
```

---

## Table of Contents

1. [⚡ Quick Start](#quick-start)
2. [Project Overview](#project-overview)
3. [✨ Features](#features)
3. [🏗️ Architecture](#architecture)
4. [📊 Dataset](#dataset)
5. [🚀 Installation & Setup](#installation--setup)
6. [💻 Usage](#usage)
7. [🤖 Model Selection](#model-selection)
8. [🌐 Web Interface](#web-interface)
9. [📡 API Reference](#api-reference)
10. [🐳 Deployment](#deployment)
11. [🔧 Commands Reference](#commands-reference)
12. [🚨 Troubleshooting](#troubleshooting)
13. [📈 Performance Metrics](#performance-metrics)
14. [🔮 Future Enhancements](#future-enhancements)
15. [🤝 Contributing](#contributing)
16. [📄 License](#license)

---

## Project Overview

The **Fake News Detection System** is a comprehensive, open-source platform that classifies news statements and articles as **real** or **fake** using advanced NLP models. Built with modern web technologies and AI models, it provides both a beautiful web interface and robust REST API.

**🎯 Key Objectives:**
- Develop a fully free ML pipeline for fake news detection
- Provide multiple model options (BERT, TF-IDF, Keyword matching)
- Beautiful, responsive web interface with URL scraping capabilities
- RESTful API for integration with other applications
- Easy deployment with Docker support

---

## ✨ Features

### 🧠 **AI Models**
- **🤖 Hybrid Model**: BERT embeddings + Random Forest classifier
- **📊 Baseline Model**: TF-IDF + Logistic Regression
- **🔤 Keyword Model**: Simple pattern matching
- **🎛️ Model Selection**: Choose specific models via dropdown

### 🌐 **Web Interface**
- **🎨 Modern UI**: Beautiful gradient design with animations
- **📱 Responsive**: Works on desktop, tablet, and mobile
- **📝 Text Analysis**: Direct text input for fact-checking
- **🔗 URL Analysis**: Automatic web scraping from news URLs
- **📖 Read More/Less**: Expandable full article view
- **💡 Examples**: Quick-test buttons with sample statements

### 🔧 **Web Scraping**
- **📰 News Sites**: Supports major news websites (NDTV, BBC, CNN, etc.)
- **🤖 Multiple Methods**: newspaper3k + BeautifulSoup fallback
- **📄 Content Extraction**: Automatic title and article text extraction
- **🛡️ Error Handling**: Graceful fallbacks when scraping fails

### 🚀 **API & Deployment**
- **⚡ FastAPI**: High-performance REST API
- **📚 Auto Documentation**: Swagger UI at `/docs`
- **🐳 Docker Support**: Easy containerized deployment
- **🔄 Health Checks**: API status monitoring
- **📊 Model Info**: Shows which model was used for each prediction

---

## 🏗️ Architecture

```plaintext
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   FastAPI       │    │   AI Models     │
│   (Flask)       │───▶│   Backend       │───▶│   Selection     │
│   - Text Input  │    │   - /predict    │    │   - Hybrid      │
│   - URL Input   │    │   - /predict_url│    │   - Baseline    │
│   - Model Select│    │   - Web Scraper │    │   - Keyword     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**📁 Project Structure:**
```
FakeNewsDetectorUsingBert/
├── app/                    # 🏠 Main application
│   ├── api.py             # 🚀 FastAPI backend server
│   ├── web.py             # 🌐 Flask web interface
│   ├── scraper.py         # 🕷️ Web scraping module
│   ├── data.py            # 📊 Data processing
│   ├── train.py           # 🎓 Model training
│   ├── test.py            # 🧪 Testing & validation
│   ├── static/            # 🎨 CSS and assets
│   │   └── style.css      # 💅 Beautiful styling
│   └── models/            # 🤖 Trained models (created after training)
│       ├── hybrid_model.pkl
│       ├── baseline_model.pkl
│       └── vectorizer.pkl
├── liar_dataset/           # 📚 Dataset files
│   ├── train.tsv
│   ├── valid.tsv
│   └── test.tsv
├── requirements.txt        # 📦 Python dependencies
├── Makefile               # ⚡ Easy commands
├── docker-compose.yml     # 🐳 Docker deployment
├── Dockerfile             # 📦 Container config
├── .gitignore             # 🚫 Git ignore rules
└── README.md              # 📖 This file
```

---

## 📊 Dataset

**LIAR Dataset**: Benchmark dataset with 12,836 human-labeled political statements from PolitiFact.

| Attribute | Details |
|-----------|---------|
| **📰 Source** | PolitiFact fact-checking website |
| **📊 Size** | 12,836 statements |
| **🏷️ Labels** | 6-class → Binary (Fake/Real) |
| **📝 Features** | Statement text, speaker, context |
| **📈 Splits** | Train: 10,269, Val: 1,284, Test: 1,283 |

---

## 🚀 Installation & Setup

### **📋 System Requirements**
- 🐍 Python 3.8+ (3.9+ recommended)
- 💾 4GB+ RAM
- 💽 2GB+ free disk space
- 🌐 Internet connection (for model downloads)

### **1️⃣ Clone Repository**
```bash
git clone https://github.com/BharathTT/FakeNewsDetectorUsingBert.git
cd FakeNewsDetectorUsingBert
```

### **2️⃣ Setup Python Environment**

**🍎 macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**🪟 Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### **3️⃣ Setup Dataset**
```bash
# Create dataset directory
mkdir -p liar_dataset

# Download sample data (included in repo)
# For full dataset, visit: https://arxiv.org/abs/1705.00648
```

### **4️⃣ Train Models**
```bash
# Train baseline model (2-3 minutes)
make baseline

# Train hybrid model (10-15 minutes)
make hybrid

# Train all models
make train
```

---

## 💻 Usage

### **🚀 Quick Start**

#### **1️⃣ Start API Server**
```bash
# Terminal 1: Start API
make api
```
**🌐 API available at:** http://localhost:8000

#### **2️⃣ Start Web Interface**
```bash
# Terminal 2: Start Web Interface
make web
```
**🖥️ Web interface:** http://localhost:5001

#### **3️⃣ Test the System**

**📝 Text Analysis:**
1. Go to http://localhost:5001
2. Select "📝 Text Analysis" tab
3. Enter: "Scientists discovered aliens on Mars yesterday"
4. Choose model from dropdown
5. Click "🔍 Analyze Statement"

**🔗 URL Analysis:**
1. Select "🔗 URL Analysis" tab
2. Enter: "https://www.bbc.com/news/world-..."
3. Choose model from dropdown
4. Click "🔍 Analyze Article"

**🧪 API Testing:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "The moon is made of cheese", "model": "hybrid"}'
```

---

## 🤖 Model Selection

The system offers **4 model options** via dropdown menu:

### **🎯 Auto (Recommended)**
- Tries Hybrid → Baseline → Keyword
- Best accuracy with fallback support
- **Use when**: You want the best possible results

### **🧠 Hybrid Model**
- BERT embeddings + Random Forest
- **Accuracy**: ~62%
- **Speed**: ~200ms
- **Use when**: Maximum accuracy is needed

### **📊 Baseline Model**
- TF-IDF + Logistic Regression
- **Accuracy**: ~60%
- **Speed**: ~50ms
- **Use when**: Fast predictions needed

### **🔤 Keyword Model**
- Simple pattern matching
- **Accuracy**: ~55%
- **Speed**: ~10ms
- **Use when**: Ultra-fast screening needed

**💡 Model Selection Examples:**
```bash
# Force hybrid model
curl -X POST "http://localhost:8000/predict" \
     -d '{"text": "Breaking news!", "model": "hybrid"}'

# Force baseline model
curl -X POST "http://localhost:8000/predict" \
     -d '{"text": "Breaking news!", "model": "baseline"}'
```

---

## 🌐 Web Interface

### **🎨 Features**
- **🌈 Beautiful Design**: Modern gradient background with animations
- **📱 Responsive**: Works on all devices
- **🎛️ Model Selection**: Dropdown to choose AI model
- **📝 Dual Input**: Text analysis + URL analysis tabs
- **📖 Read More/Less**: Expandable article content
- **💡 Quick Examples**: One-click test buttons
- **🔄 Loading States**: Smooth loading animations
- **🎯 Color-coded Results**: Green (Real), Red (Fake), Yellow (Uncertain)

### **🖱️ How to Use**
1. **Choose Model**: Select from dropdown (Auto/Hybrid/Baseline/Keyword)
2. **Select Tab**: Text Analysis or URL Analysis
3. **Enter Input**: Type text or paste news URL
4. **Analyze**: Click the analyze button
5. **View Results**: See prediction with confidence score
6. **Read More**: Click to expand full article (for URLs)

### **📱 Mobile Support**
- Responsive design works on phones/tablets
- Touch-friendly buttons and inputs
- Optimized for mobile browsing

---

## 📡 API Reference

### **🌐 Base URL**
- **Local**: http://localhost:8000
- **Docker**: http://localhost:8000
- **Docs**: http://localhost:8000/docs

### **📋 Endpoints**

#### **POST /predict** - Text Analysis
Analyze text statements for fake news.

**📤 Request:**
```json
{
  "text": "Scientists discovered aliens on Mars",
  "model": "hybrid"  // optional: "auto", "hybrid", "baseline", "keyword"
}
```

**📥 Response:**
```json
{
  "label": "Fake",
  "confidence": 0.85,
  "model": "hybrid"
}
```

#### **POST /predict_url** - URL Analysis
Analyze news articles from URLs.

**📤 Request:**
```json
{
  "url": "https://www.bbc.com/news/world-...",
  "model": "auto"
}
```

**📥 Response:**
```json
{
  "label": "Real",
  "confidence": 0.78,
  "model": "hybrid",
  "title": "Article Title Here",
  "text_preview": "First 200 characters...",
  "full_text": "Complete article text...",
  "scraper_method": "newspaper3k"
}
```

#### **GET /health** - Health Check
```json
{
  "status": "healthy",
  "baseline_loaded": true,
  "hybrid_loaded": true
}
```

### **🧪 Testing Examples**

**🐍 Python:**
```python
import requests

# Test text analysis
response = requests.post('http://localhost:8000/predict', json={
    'text': 'The stock market crashed today',
    'model': 'hybrid'
})
print(response.json())

# Test URL analysis
response = requests.post('http://localhost:8000/predict_url', json={
    'url': 'https://www.ndtv.com/india-news/...',
    'model': 'auto'
})
print(response.json())
```

**🌐 JavaScript:**
```javascript
// Text analysis
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    text: 'Breaking: Major discovery announced',
    model: 'baseline'
  })
}).then(r => r.json()).then(console.log);
```

---

## 🐳 Deployment

### **🚀 Docker (Recommended)**

**Quick Deploy:**
```bash
# Build and start all services
docker-compose up --build

# Or use Makefile
make docker
```

**🌐 Access:**
- **API**: http://localhost:8000
- **Web**: http://localhost:5000
- **Docs**: http://localhost:8000/docs

**🔧 Production:**
```bash
# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### **💻 Manual Deployment**

**Local Development:**
```bash
# Terminal 1: API
make api

# Terminal 2: Web
make web
```

**🌍 Production Server:**
```bash
# Install production server
pip install gunicorn

# Run API
cd app && gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app --bind 0.0.0.0:8000

# Run Web
cd app && gunicorn -w 2 web:app --bind 0.0.0.0:5000
```

### **☁️ Cloud Deployment**
- **Render.com**: Connect GitHub, auto-deploy
- **Railway.app**: One-click deploy
- **Heroku**: Use Dockerfile
- **Google Cloud Run**: Serverless containers

---

## 🔧 Commands Reference

### **⚡ Makefile Commands**
```bash
make install    # 📦 Install dependencies
make test      # 🧪 Run all tests
make baseline  # 🎓 Train baseline model (2-3 min)
make hybrid    # 🧠 Train hybrid model (10-15 min)
make train     # 🎓 Train all models
make api       # 🚀 Start API server
make web       # 🌐 Start web interface
make docker    # 🐳 Deploy with Docker
make clean     # 🧹 Clean cache files
```

### **🐍 Direct Python Commands**
```bash
# Training
cd app && python train.py                    # Train all models
cd app && python test.py                     # Run tests

# Servers
cd app && uvicorn api:app --reload --port 8000    # API server
cd app && python web.py                           # Web server

# Docker
docker-compose up --build                    # Build and run
docker-compose logs -f                       # View logs
```

---

## 🚨 Troubleshooting

### **🔧 Common Issues**

#### **📦 Installation Problems**
```bash
# Python not found
python3 -m pip install -r requirements.txt

# Virtual environment issues
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### **🚀 Server Issues**
```bash
# Port already in use
lsof -ti:8000 | xargs kill -9
make api

# API not responding
curl http://localhost:8000/health
```

#### **🤖 Model Issues**
```bash
# Models not found
make train

# Training fails
make baseline  # Try baseline only first
```

#### **🌐 Web Scraping Issues**
```bash
# URL analysis fails
# Try different news URLs
# Check API logs for error messages
```

### **🩺 Quick Diagnostics**
```bash
# Test everything
make test

# Check Python environment
python --version
pip list | grep -E "torch|transformers|sklearn"

# Test API health
curl http://localhost:8000/health

# Check model files
ls -la app/models/
```

---

## 📈 Performance Metrics

| **Metric** | **Hybrid** | **Baseline** | **Keyword** |
|------------|------------|--------------|-------------|
| **🎯 Accuracy** | ~62% | ~60% | ~55% |
| **⚡ Speed** | ~200ms | ~50ms | ~10ms |
| **💾 Size** | ~400MB | ~1MB | <1KB |
| **🧠 Complexity** | High | Medium | Low |

**📊 Use Cases:**
- **🎯 High Accuracy**: Use Hybrid model
- **⚡ Fast Response**: Use Baseline model  
- **🚀 Ultra Fast**: Use Keyword model
- **🎛️ Best Balance**: Use Auto mode

---

## 🔮 Future Enhancements

### **🤖 AI Improvements**
- [ ] RoBERTa/DeBERTa models
- [ ] Ensemble methods
- [ ] Multilingual support
- [ ] Confidence explanations (LIME/SHAP)

### **🌐 Web Features**
- [ ] User accounts and history
- [ ] Batch URL analysis
- [ ] Browser extension
- [ ] Real-time news monitoring
- [ ] Social media integration

### **🔧 Technical**
- [ ] Model quantization
- [ ] Caching system
- [ ] A/B testing
- [ ] Advanced monitoring
- [ ] Auto-scaling

---

## 🤝 Contributing

### **🚀 How to Contribute**
1. **🍴 Fork** the repository
2. **🌿 Create** feature branch: `git checkout -b feature/amazing-feature`
3. **💾 Commit** changes: `git commit -m 'Add amazing feature'`
4. **📤 Push** to branch: `git push origin feature/amazing-feature`
5. **🔄 Open** Pull Request

### **🛠️ Development Setup**
```bash
# Clone your fork
git clone https://github.com/your-username/FakeNewsDetectorUsingBert.git
cd FakeNewsDetectorUsingBert

# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
make test
```

### **📝 Code Style**
- Follow PEP 8 for Python
- Add docstrings for functions
- Include tests for new features
- Keep functions focused and small

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **📚 LIAR Dataset**: Wang, William Yang. "Liar, liar pants on fire": A new benchmark dataset for fake news detection. ACL 2017.
- **🤖 BERT Model**: Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019.
- **🔧 Hugging Face**: Transformers library
- **📰 PolitiFact**: Dataset source

---

## 📞 Contact & Support

- **🐛 Issues**: [GitHub Issues](https://github.com/BharathTT/FakeNewsDetectorUsingBert/issues)
- **📧 Email**: bharath@example.com
- **📖 Documentation**: This README + code comments
- **💬 Discussions**: GitHub Discussions

---

**⭐ Star this repository if you found it helpful!**

**🚀 Ready to detect fake news? Let's get started!**