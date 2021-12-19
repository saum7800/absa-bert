import torch
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer

class ABSABertDataset(torch.utils.data.Dataset):
    def __init__(self, text, aspects, labels):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encodings = tokenizer(text, aspects, truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)