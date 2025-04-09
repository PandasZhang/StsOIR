from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch

def load_banking77(tokenizer, max_length):
    dataset = load_dataset("PolyAI/banking77")
    label_map = {label: i for i, label in enumerate(dataset["train"].features["label"].names)}
    
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)
    
    dataset = dataset.map(tokenize, batched=True)
    
    def make_loader(split):
        data = dataset[split]
        return DataLoader(
            TensorDataset(
                torch.tensor(data["input_ids"]),
                torch.tensor(data["attention_mask"]),
                torch.tensor(data["label"])
            ), batch_size=32, shuffle=(split=="train")
        )
    
    return make_loader("train"), make_loader("test"), label_map
