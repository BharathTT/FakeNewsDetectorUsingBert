from flask import Flask, render_template_string, request, jsonify
import requests

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html>
<head><title>Fake News Detector</title></head>
<body>
    <h1>Fake News Detection</h1>
    <form id="form">
        <textarea id="text" placeholder="Enter statement..."></textarea><br>
        <button type="submit">Check</button>
    </form>
    <div id="result"></div>
    <script>
        document.getElementById('form').onsubmit = async function(e) {
            e.preventDefault();
            const text = document.getElementById('text').value;
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: 'text=' + encodeURIComponent(text)
            });
            const result = await response.json();
            document.getElementById('result').innerHTML = result.label + ' (' + result.confidence + ')';
        };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    response = requests.post('http://localhost:8000/predict', json={'text': text})
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(debug=True, port=5000)