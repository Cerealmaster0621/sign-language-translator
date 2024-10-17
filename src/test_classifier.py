import cv2
import mediapipe as mp
import pickle
import numpy as np
import logging

## After creating data.pickle file, run this file to test your model
## Run this file with python test_classifier.py

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the model
try:
    with open("./model.pickle", "rb") as f:
        model_dict = pickle.load(f)
    model = model_dict["model"]
except FileNotFoundError:
    logging.error("model.pickle file not found. Please run train_classifier.py first.")
    exit(1)

# Set up MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        logging.error("Can't receive frame (stream end?). Exiting ...")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
            )

        data_aux = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = landmark.x * frame.shape[1]
                y = landmark.y * frame.shape[0]
                data_aux.extend([x, y])

        # Predict the digit
        prediction = model.predict([np.asarray(data_aux)])
        cv2.putText(frame, f"Prediction: {prediction[0]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
