from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch

# Define a simple dataset class for demonstration
class FrustrationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Sample data for demonstration
texts = [
    "I am so frustrated with this service.",
    "This is taking forever and I'm really unhappy.",
    "Thank you for your help!",
    "Everything is fine, I appreciate your support."
]
labels = [1, 1, 0, 0]  # 1 = Frustrated, 0 = Not Frustrated

from transformers import BertTokenizer, BertForSequenceClassification

# Replace <your_token> with your actual Hugging Face token
token = "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"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_auth_token=token)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, use_auth_token=token)

# Load tokenizer and initialize dataset
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#dataset = FrustrationDataset(texts, labels, tokenizer)

# Initialize DataLoader
dataloader = DataLoader(model, batch_size=2, shuffle=True)

# Load pre-trained model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Test one batch of data (forward pass)
batch = next(iter(dataloader))
input_ids = batch['input_ids']
attention_mask = batch['attention_mask']
labels = batch['label']

# Forward pass through the model
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
logits = outputs.logits

# Convert logits to probabilities and labels
probabilities = torch.softmax(logits, dim=1)
predicted_labels = torch.argmax(probabilities, dim=1)

# Package the results
results = {
    "texts": texts[:2],  # Texts in the batch
    "true_labels": labels.tolist(),
    "predicted_labels": predicted_labels.tolist(),
    "probabilities": probabilities.tolist()
}

print(results)
