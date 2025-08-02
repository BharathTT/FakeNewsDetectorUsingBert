from flask import Flask, render_template_string, request, jsonify
import requests

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔍 Fake News Detector</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <h1>🔍 Fake News Detector</h1>
        <p class="subtitle">AI-powered fact checking using BERT + Random Forest</p>
        
        <div class="tabs">
            <button class="tab-btn active" id="text-btn">📝 Text Analysis</button>
            <button class="tab-btn" id="url-btn">🔗 URL Analysis</button>
        </div>
        
        <div class="model-selector">
            <label for="model-choice">🤖 Choose Model:</label>
            <select id="model-choice">
                <option value="auto">Auto (Best Available)</option>
                <option value="hybrid">Hybrid (BERT + Random Forest)</option>
                <option value="baseline">Baseline (TF-IDF + Logistic Regression)</option>
                <option value="keyword">Keyword Matching</option>
            </select>
        </div>
        
        <div id="text-tab" class="tab-content active">
            <form id="text-form">
                <textarea id="text" placeholder="Enter a news statement or claim to verify..."></textarea>
                <br>
                <button type="submit">🔍 Analyze Statement</button>
            </form>
        </div>
        
        <div id="url-tab" class="tab-content">
            <form id="url-form">
                <input type="url" id="url" placeholder="Enter news article URL (e.g., https://www.ndtv.com/...)">
                <br>
                <button type="submit">🔍 Analyze Article</button>
            </form>
        </div>
        
        <div class="loading" id="loading">🔄 Analyzing...</div>
        <div id="result" class="result" style="display: none;"></div>
        
        <div class="examples">
            <h3>💡 Try these examples:</h3>
            <div class="example" onclick="setExample('Scientists discovered aliens on Mars yesterday')">🛸 Scientists discovered aliens on Mars yesterday</div>
            <div class="example" onclick="setExample('The stock market closed higher today')">📈 The stock market closed higher today</div>
            <div class="example" onclick="setExample('COVID vaccines are 95% effective against severe illness')">💉 COVID vaccines are 95% effective against severe illness</div>
            <div class="example" onclick="setExample('The moon is made of cheese')">🧀 The moon is made of cheese</div>
        </div>
    </div>
    
    <script>
        // Tab switching
        document.getElementById('text-btn').onclick = function() {
            document.getElementById('text-btn').classList.add('active');
            document.getElementById('url-btn').classList.remove('active');
            document.getElementById('text-tab').classList.add('active');
            document.getElementById('url-tab').classList.remove('active');
        };
        
        document.getElementById('url-btn').onclick = function() {
            document.getElementById('url-btn').classList.add('active');
            document.getElementById('text-btn').classList.remove('active');
            document.getElementById('url-tab').classList.add('active');
            document.getElementById('text-tab').classList.remove('active');
        };
        
        function setExample(text) {
            document.getElementById('text').value = text;
        }
        
        function toggleFullText() {
            const fullText = document.getElementById('full-text');
            const preview = document.querySelector('.text-preview');
            const btn = document.querySelector('.read-more-btn');
            
            if (fullText && fullText.style.display === 'none') {
                fullText.style.display = 'block';
                if (preview) preview.style.display = 'none';
                btn.textContent = '📖 Read Less';
            } else if (fullText) {
                fullText.style.display = 'none';
                if (preview) preview.style.display = 'block';
                btn.textContent = '📖 Read More';
            }
        }
        
        // Text form
        document.getElementById('text-form').onsubmit = function(e) {
            e.preventDefault();
            const text = document.getElementById('text').value.trim();
            if (!text) return;
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            const model = document.getElementById('model-choice').value;
            fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: 'text=' + encodeURIComponent(text) + '&model=' + encodeURIComponent(model)
            })
            .then(r => r.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('result').style.display = 'flex';
                
                if (data.error) {
                    document.getElementById('result').innerHTML = `<div>⚠️ ${data.error}</div>`;
                    document.getElementById('result').className = 'result uncertain';
                } else {
                    const label = data.label.toLowerCase();
                    const emoji = label === 'fake' ? '❌' : label === 'real' ? '✅' : '⚠️';
                    const confidence = Math.round(data.confidence * 100);
                    
                    document.getElementById('result').className = `result ${label}`;
                    document.getElementById('result').innerHTML = `
                        <div>
                            ${emoji} <strong>${data.label}</strong> (${confidence}% confidence)
                            <div class="model-info">Model: ${data.model || 'unknown'}</div>
                        </div>
                    `;
                }
            })
            .catch(e => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('result').style.display = 'flex';
                document.getElementById('result').className = 'result uncertain';
                document.getElementById('result').innerHTML = `<div>⚠️ Error: ${e.message}</div>`;
            });
        };
        
        // URL form
        document.getElementById('url-form').onsubmit = function(e) {
            e.preventDefault();
            const url = document.getElementById('url').value.trim();
            if (!url) return;
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            const model = document.getElementById('model-choice').value;
            fetch('/predict_url', {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: 'url=' + encodeURIComponent(url) + '&model=' + encodeURIComponent(model)
            })
            .then(r => r.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('result').style.display = 'flex';
                
                if (data.error) {
                    document.getElementById('result').innerHTML = `<div>⚠️ ${data.error}</div>`;
                    document.getElementById('result').className = 'result uncertain';
                } else {
                    const label = data.label.toLowerCase();
                    const emoji = label === 'fake' ? '❌' : label === 'real' ? '✅' : '⚠️';
                    const confidence = Math.round(data.confidence * 100);
                    
                    let html = `
                        <div>
                            ${emoji} <strong>${data.label}</strong> (${confidence}% confidence)
                            <div class="model-info">Model: ${data.model || 'unknown'}</div>
                    `;
                    
                    if (data.title) {
                        html += `<div class="article-title"><strong>Title:</strong> ${data.title}</div>`;
                    }
                    
                    if (data.text_preview) {
                        html += `<div class="text-preview"><strong>Preview:</strong> ${data.text_preview}</div>`;
                        if (data.full_text && data.full_text.length > data.text_preview.length) {
                            html += `<button class="read-more-btn" onclick="toggleFullText()">📖 Read More</button>`;
                            html += `<div class="full-text" id="full-text" style="display: none;"><strong>Full Article:</strong><br>${data.full_text.replace(/\\n/g, '<br>')}</div>`;
                        }
                    }
                    
                    html += '</div>';
                    document.getElementById('result').className = `result ${label}`;
                    document.getElementById('result').innerHTML = html;
                }
            })
            .catch(e => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('result').style.display = 'flex';
                document.getElementById('result').className = 'result uncertain';
                document.getElementById('result').innerHTML = `<div>⚠️ Error: ${e.message}</div>`;
            });
        };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    model = request.form.get('model', 'auto')
    try:
        response = requests.post('http://localhost:8000/predict', json={'text': text, 'model': model}, timeout=30)
        return jsonify(response.json())
    except:
        return jsonify({"error": "API not available"})

@app.route('/predict_url', methods=['POST'])
def predict_url():
    url = request.form.get('url')
    model = request.form.get('model', 'auto')
    try:
        response = requests.post('http://localhost:8000/predict_url', json={'url': url, 'model': model}, timeout=30)
        return jsonify(response.json())
    except:
        return jsonify({"error": "API not available"})

if __name__ == '__main__':
    app.run(debug=True, port=5001)