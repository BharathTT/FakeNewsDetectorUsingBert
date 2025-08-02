from transformers import BertForSequenceClassification, BertTokenizer, BertModel
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import torch
import numpy as np
from data import load_liar_dataset, preprocess_liar, LiarDataset

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def train_baseline():
    print("Training baseline...")
    import os
    import pickle
    os.makedirs('./models', exist_ok=True)
    
    train_df = preprocess_liar(load_liar_dataset('../liar_dataset/train.tsv'))
    test_df = preprocess_liar(load_liar_dataset('../liar_dataset/test.tsv'))
    
    # Improved TF-IDF with better parameters
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),  # Include bigrams
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    X_train = vectorizer.fit_transform(train_df['statement'])
    X_test = vectorizer.transform(test_df['statement'])
    
    # Improved LogisticRegression with class balancing
    model = LogisticRegression(
        class_weight='balanced',
        C=1.0,
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, train_df['binary_label'])
    
    # Save models
    with open('./models/baseline_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('./models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("Baseline model saved")
    
    predictions = model.predict(X_test)
    print(classification_report(test_df['binary_label'], predictions, digits=4))

def get_bert_embeddings(texts, tokenizer, model):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
            outputs = model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(embedding)
    return np.array(embeddings)

def train_hybrid():
    print("Training Hybrid BERT + Random Forest...")
    import os
    import pickle
    os.makedirs('./models', exist_ok=True)
    
    train_df = preprocess_liar(load_liar_dataset('../liar_dataset/train.tsv'))
    test_df = preprocess_liar(load_liar_dataset('../liar_dataset/test.tsv'))
    
    # Load BERT model for embeddings
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    # Get BERT embeddings
    print("Extracting BERT embeddings...")
    X_train = get_bert_embeddings(list(train_df['statement']), tokenizer, bert_model)
    X_test = get_bert_embeddings(list(test_df['statement']), tokenizer, bert_model)
    
    # Train Random Forest on BERT embeddings
    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, train_df['binary_label'])
    
    # Save models
    with open('./models/hybrid_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    tokenizer.save_pretrained('./models/bert_tokenizer')
    bert_model.save_pretrained('./models/bert_embedder')
    print("Hybrid model saved")
    
    # Evaluate
    predictions = rf_model.predict(X_test)
    print(classification_report(test_df['binary_label'], predictions, digits=4))

if __name__ == "__main__":
    train_baseline()
    train_hybrid()