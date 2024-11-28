from datasets import load_dataset
from transformers import DistilBertTokenizer

# Load the dataset
dataset = load_dataset('csv', data_files='D:/ai model/datasets.csv')

# Load pre-trained tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Apply tokenization to the dataset
tokenized_datasets = dataset['train'].map(tokenize_function, batched=True)

# Verify tokenized data
print(tokenized_datasets)

