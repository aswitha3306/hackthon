import pyttsx3
from datetime import datetime
import os
import cv2

# Initialize voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('volume', 1.0)

# Create evidence folder
if not os.path.exists("evidence"):
    os.makedirs("evidence")

def trigger_voice_alert(reg_no, name, violation_type, frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evidence/{reg_no}_{timestamp}.jpg"
    
    # Save evidence screenshot
    cv2.imwrite(filename, frame)

    message = f"""
    Critical Alert.
    Candidate Registration Number {reg_no}.
    Name {name}.
    Malpractice detected.
    Type {violation_type}.
    """

    engine.say(message)
    engine.runAndWait()

    print(f"[ALERT] {reg_no} - {violation_type}")
