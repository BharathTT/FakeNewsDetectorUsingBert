from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
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

def train_bert():
    print("Training BERT...")
    import os
    os.makedirs('./models', exist_ok=True)
    
    train_df = preprocess_liar(load_liar_dataset('../liar_dataset/train.tsv'))
    val_df = preprocess_liar(load_liar_dataset('../liar_dataset/valid.tsv'))
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(list(train_df['statement']), padding=True, truncation=True, max_length=128)
    val_encodings = tokenizer(list(val_df['statement']), padding=True, truncation=True, max_length=128)
    
    train_dataset = LiarDataset(train_encodings, list(train_df['binary_label']))
    val_dataset = LiarDataset(val_encodings, list(val_df['binary_label']))
    
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    args = TrainingArguments(
        output_dir='./models/liar_bert',
        num_train_epochs=2,  # Reduced for faster training
        per_device_train_batch_size=16,  # Smaller batch for stability
        per_device_eval_batch_size=32,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        learning_rate=2e-5,  # Lower learning rate
        weight_decay=0.01,
        warmup_steps=100,  # Reduced warmup
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        logging_steps=50,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    
    # Save the final model
    model.save_pretrained('./models/liar_bert')
    tokenizer.save_pretrained('./models/liar_bert')
    print("BERT model saved")

if __name__ == "__main__":
    train_baseline()
    train_bert()