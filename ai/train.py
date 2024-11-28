from datasets import load_dataset, DatasetDict

# Load dataset
dataset = load_dataset('csv', data_files='datasets.csv')

# Check if 'train' split exists
if 'train' in dataset:
    # Split into train and test
    train_test_split = dataset['train'].train_test_split(test_size=0.2)
    dataset = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })
else:
    print("The dataset does not have a 'train' split!")

# Verify the splits
print(dataset)
