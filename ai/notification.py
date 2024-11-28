from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Assuming model and tokenizer are already loaded
model = DistilBertForSequenceClassification.from_pretrained('./emotion_model')
tokenizer = DistilBertTokenizer.from_pretrained('./emotion_model')

# Function to predict emotion from text
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1).item()  # Get the predicted label
    return predicted_class

# Map label back to emotion
emotion_map = {0: "happiness", 1: "frustration"}  # Your mapping

def get_emotion_from_text(text):
    predicted_class = predict_emotion(text)
    return emotion_map.get(predicted_class, "Unknown")
