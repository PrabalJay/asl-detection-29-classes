import cv2
import numpy as np
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Load model and class names
model = keras.models.load_model('models/model.keras')
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("ASL Detection", cv2.WINDOW_NORMAL)

# Detection parameters
confidence_threshold = 0.7 
no_hand_message = "Put your hand in the green box"
exit_message = "Press 'Q' to quit"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    roi_top, roi_bottom = 100, 300
    roi_left, roi_right = 300, 500
    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    resized = cv2.resize(roi, (100, 100))
    normalized = resized / 255.0
    
    pred = model.predict(np.expand_dims(normalized, axis=0), verbose=0)[0]
    pred_class = np.argmax(pred)
    confidence = np.max(pred)
    
    # Display instructions
    cv2.putText(frame, no_hand_message, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, exit_message, (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if confidence > confidence_threshold and pred_class < len(class_names):
        cv2.putText(frame, f"Sign: {class_names[pred_class]} ({confidence:.2f})", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No sign detected", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("ASL Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera released. Detection stopped.")