from huggingface_hub import HfApi, HfFolder
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Load your trained model (replace './emotion_model' with the correct path to your model files)
model = DistilBertForSequenceClassification.from_pretrained("D:/ai model/emotion_model")
tokenizer = DistilBertTokenizer.from_pretrained("D:/ai model/emotion_model") 

# Replace with your Hugging Face repository name
repo_name = "Vaishnaviks/customer-frustration-detection"

# Upload the model and tokenizer
model.push_to_hub("customer-frustration-detection")
tokenizer.push_to_hub("customer-frustration-detection")



