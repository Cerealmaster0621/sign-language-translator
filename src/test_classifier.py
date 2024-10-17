import cv2
import mediapipe as mp
import pickle
import numpy as np
import logging

def load_model(model_path):
    try:
        with open(model_path, "rb") as f:
            model_dict = pickle.load(f)
        return model_dict["model"]
    except FileNotFoundError:
        logging.error(f"{model_path} not found. Please run train_classifier.py first.")
        return None

def setup_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    return mp_hands, mp_drawing, mp_drawing_styles, hands

def process_frame(frame, hands, mp_drawing, mp_hands, mp_drawing_styles, model):
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

        prediction = model.predict([np.asarray(data_aux)])
        cv2.putText(frame, f"Prediction: {prediction[0]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def run_classifier(model_path):
    logging.basicConfig(level=logging.INFO)

    model = load_model(model_path)
    if model is None:
        return

    mp_hands, mp_drawing, mp_drawing_styles, hands = setup_mediapipe()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        frame = process_frame(frame, hands, mp_drawing, mp_hands, mp_drawing_styles, model)
        
        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

run_classifier("./model.pickle")
