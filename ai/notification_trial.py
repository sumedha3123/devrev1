from notification import get_emotion_from_text
from plyer import notification

# Function to trigger desktop notification
def send_notification(message):
    notification.notify(
        title='Emotion Detected in Conversation',
        message=message,
        timeout=10  # Notification will be displayed for 10 seconds
    )

# Example trigger
def detect_and_notify(conversation):
    for turn in conversation:
        emotion = get_emotion_from_text(turn)
        print(f"Turn: {turn} | Emotion: {emotion}")
        
        if emotion == "frustration":
            send_notification("Frustration detected! Please check the conversation.")
            
# Example conversation (list of turns)
conversation = [
    "I am really happy with the service!",
    "But I had an issue with my last order.",
    "This is so frustrating, I can't get in touch with support.",
    "Oh, finally, I got through. Thank you for your help!"
]

# Detect emotion and trigger notification if frustration is found
detect_and_notify(conversation)
