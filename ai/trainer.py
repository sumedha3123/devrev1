from datasets import load_dataset, ClassLabel 
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

# Step 1: Load the dataset
dataset = load_dataset('csv', data_files='D:/ai model/datasets.csv')

# Step 2: Process the labels
def process_labels(dataset):
    unique_labels = set(dataset['train']['label'])  # Find unique labels
    label_class = ClassLabel(names=list(unique_labels))  # type: ignore # Create ClassLabel
    
    def encode_labels(example):
        example['label'] = label_class.str2int(example['label'])  # Convert string to int
        return example

    return dataset.map(encode_labels), len(label_class.names)

dataset, num_labels = process_labels(dataset)

# Step 3: Load the model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=num_labels
)

# Step 4: Define training arguments
training_args = TrainingArguments(
    output_dir='./results',           # Output directory
    evaluation_strategy="epoch",      # Evaluation strategy
    learning_rate=2e-5,               # Learning rate
    per_device_train_batch_size=2,    # Batch size for training
    per_device_eval_batch_size=2,     # Batch size for evaluation
    num_train_epochs=3,               # Number of training epochs
    weight_decay=0.01,                # Weight decay
)

# Step 5: Define the Trainer
trainer = Trainer(
    model=model,                      # The model to be trained
    args=training_args,               # Training arguments
    train_dataset=dataset['train'],   # Training dataset
    eval_dataset=dataset['test'],     # Evaluation dataset
)

# Step 6: Train the model
trainer.train()
