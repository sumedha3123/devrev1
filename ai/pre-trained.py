from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Load the trained model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('./emotion_model')  # Ensure path is correct
tokenizer = DistilBertTokenizer.from_pretrained('./emotion_model')

# Example input
text = "I have been waiting since 2 months!"


# Tokenize the input
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Make predictions
outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax().item()

# Map the predicted class to emotion
emotion_labels = ["frustration", "happiness"]  # Adjust based on your labels
print(f"Predicted emotion: {emotion_labels[predicted_class]}")
