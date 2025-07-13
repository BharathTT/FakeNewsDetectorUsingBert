import pandas as pd
import torch
from transformers import BertTokenizer

def load_liar_dataset(file_path):
    columns = ['label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'barely_true_counts', 
               'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context']
    return pd.read_csv(file_path, sep='\t', header=None, names=columns)

def preprocess_liar(df):
    fake_labels = ['pants-fire', 'false', 'barely-true']
    df['binary_label'] = df['label'].apply(lambda x: 0 if x in fake_labels else 1)
    df['statement'] = df['statement'].str.replace(r'[^\w\s]', ' ', regex=True)
    df['statement'] = df['statement'].str.replace(r'\s+', ' ', regex=True).str.strip()
    return df[['statement', 'binary_label']].dropna()

class LiarDataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)