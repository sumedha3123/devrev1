from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertForSequenceClassification
import torch

# Load your custom model and tokenizer
def load_model(model_name='DistilBertForSequenceClassification'):
    """
    Load the tokenizer and model for emotion detection.
    
    Args:
        model_name (str): The name or path to the custom model.
    
    Returns:
        tuple: Loaded tokenizer and model.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")


def detect_emotion(text, tokenizer, model, labels=['Frustration', 'Happiness']):
    """
    Detect emotions in text using a custom-trained model.
    
    Args:
        text (str): The input text to classify.
        tokenizer: The tokenizer for the model.
        model: The pre-trained classification model.
        labels (list): The list of emotion labels corresponding to the model's output.
    
    Returns:
        str: The detected emotion with the highest probability.
    """
    try:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()

        # Map predicted label to emotion
        emotion = labels[predicted_label]
        confidence = probabilities[0][predicted_label].item()

        return f"Detected Emotion: {emotion}"
    except Exception as e:
        return f"Error occurred: {str(e)}"
# Load the model and tokenizer
model_name = "D:/ai model/emotion_model"
tokenizer, model = load_model(model_name)

# Example text input
text_input = "Hurry up!!"
result = detect_emotion(text_input, tokenizer, model)

print(result)

