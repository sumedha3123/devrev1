
from datasets import load_dataset, ClassLabel

# Step 1: Load the dataset
dataset = load_dataset('csv', data_files='D:/ai model/datasets.csv')

# Step 2: Convert 'label' column to integers
def process_labels(dataset):
    # Get unique labels from the 'label' column
    unique_labels = set(dataset['train']['label'])  # Find unique labels
    label_class = ClassLabel(names=list(unique_labels))  # Create a ClassLabel
    
    # Map the string labels to integers
    def encode_labels(example):
        example['label'] = label_class.str2int(example['label'])  # Convert string to int
        return example

    # Apply this function to the dataset
    return dataset.map(encode_labels), len(label_class.names)

# Step 3: Apply the transformation
dataset, num_labels = process_labels(dataset)

# Print the first few rows to verify the conversion
print(dataset['train'][:5])  # This will show the first 5 rows with numeric labels
print(f"Number of labels: {num_labels}")  # This should output the number of unique labels (e.g., 2)

# Step 4: Load the pre-trained model
from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=num_labels
)

# Print model architecture (optional, for verification)
print(model)
