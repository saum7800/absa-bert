import torch
from transformers import BertTokenizer, T5Tokenizer

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

class T5Dataset(torch.utils.data.Dataset):
    def __init__(self, text, aspects, labels):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        label_map = {2:"positive", 0:"negative", 1:"neutral"}
        max_source_length = 512
        max_target_length = 10
        
        aspect_prefix = "aspect: "
        sentence_prefix = " sentence: "

        self.text = text

        self.input_encodings = self.tokenizer([aspect_prefix + aspects[x] + sentence_prefix + text[x] for x in range(len(aspects))],
                                                padding='longest', 
                                                max_length=max_source_length, 
                                                truncation=True, 
                                                return_tensors="pt")

        self.target_encodings = self.tokenizer([label_map[x] for x in labels],
                                                padding='longest', 
                                                max_length=max_target_length, 
                                                truncation=True)
        self.labels = self.target_encodings.input_ids
        self.labels = torch.tensor(self.labels)
        self.labels[self.labels == self.tokenizer.pad_token_id] = -100

        self.labels_pure = [label_map[x] for x in labels]


    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.input_encodings.items()}
        item['labels'] = self.labels[idx]
        item['labels_pure'] = self.labels_pure[idx]
        return item

    def __len__(self):
        return len(self.text)