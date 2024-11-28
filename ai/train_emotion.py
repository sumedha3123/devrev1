from datasets import load_dataset, DatasetDict
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Load the Dataset
dataset = load_dataset('csv', data_files='datasets.csv')

# 2. Split the Dataset (if only 'train' split is available)
if 'train' in dataset:
    train_test_split = dataset['train'].train_test_split(test_size=0.2)  # 80% train, 20% test
    dataset = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })

# 3. Load Pretrained Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenization Function with padding and truncation
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

# Tokenize Dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. Create Label Mapping (map string labels to numeric labels)
label_map = {"happiness": 0, "frustration": 1}  # Adjust based on your labels

# 4. Create Label Mapping (map string labels to numeric labels)
label_map = {"happiness": 0, "frustration": 1}  # Adjust based on your labels

# Map string labels to integers in the dataset
def map_labels(example):
    # Ensure the label is valid
    if example['label'] in label_map:
        example['label'] = label_map[example['label']]
    else:
        # Log unexpected labels instead of stopping execution
        print(f"Warning: Unexpected label encountered: {example['label']}")
        example['label'] = -1  # Assign a default value for invalid labels
    return example

# Apply label mapping and filter out invalid examples
tokenized_datasets = tokenized_datasets.map(map_labels)

# Filter out invalid labels (-1)
def filter_invalid_labels(example):
    return example['label'] != -1

tokenized_datasets = tokenized_datasets.filter(filter_invalid_labels)

# 5. Prepare the Model
labels = list(label_map.values())  # The unique labels based on the mapping
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

# Load Pretrained Model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# 6. Define Metrics for Evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# 7. Training Arguments
training_args = TrainingArguments(
    output_dir='./results',               # Output directory
    eval_strategy="epoch",                # Evaluate at the end of each epoch
    learning_rate=2e-5,                   # Learning rate
    per_device_train_batch_size=4,        # Batch size for training
    per_device_eval_batch_size=4,         # Batch size for evaluation
    num_train_epochs=3,                   # Number of epochs
    weight_decay=0.01,                    # Weight decay
    save_steps=100,                       # Save checkpoint every 100 steps
    save_total_limit=2,                   # Limit total checkpoints to 2
    log_level="info",
    fp16=False                            # Disable mixed precision (for CPU)
)

# 8. Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 9. Train the Model
trainer.train()

# 10. Evaluate the Model
metrics = trainer.evaluate()
print(metrics)

# 11. Save the Model and Tokenizer
model.save_pretrained('./emotion_model')
tokenizer.save_pretrained('./emotion_model')
