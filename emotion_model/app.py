from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertForSequenceClassification
import torch

# Load the custom emotion detection model and tokenizer
model_name = "D:/ai model/emotion_model"  # Replace with your model path or Hugging Face model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define the FastAPI app
app = FastAPI()

# Define input structure
class TextInput(BaseModel):
    text: str

# Emotion detection function
def detect_emotion(text: str, labels=['Frustration', 'Happiness']):
    """
    Detect emotions in text using the loaded model and tokenizer.

    Args:
        text (str): The input text to analyze.
        labels (list): The emotion labels corresponding to the model's outputs.

    Returns:
        dict: The detected emotion and confidence score.
    """
    try:
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()

        # Get emotion and confidence
        emotion = labels[predicted_label]
        confidence = probabilities[0][predicted_label].item()

        return {"emotion": emotion}
    except Exception as e:
        return {"error": str(e)}

# FastAPI POST endpoint
@app.post("/detect-emotion")
def process_text(input_data: TextInput):
    """
    API endpoint to detect emotion from text input.

    Args:
        input_data (TextInput): Input text in JSON format.

    Returns:
        dict: The detected emotion and confidence.
    """
    result = detect_emotion(input_data.text)
    return result
