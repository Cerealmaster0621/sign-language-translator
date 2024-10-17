import os
import mediapipe as mp
import cv2
import pickle
from constants import DATA_PATH_NUMBERS
import logging

# Hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

logging.basicConfig(level=logging.INFO)

# Visualize data
print(f"Checking directory: {DATA_PATH_NUMBERS}")
if not os.path.exists(DATA_PATH_NUMBERS):
    print(f"Error: Directory not found: {DATA_PATH_NUMBERS}")
    exit(1)

data = []
labels = []
for dir_ in os.listdir(DATA_PATH_NUMBERS):
    dir_path = os.path.join(DATA_PATH_NUMBERS, dir_)
    if not os.path.isdir(dir_path):
        continue
    logging.info(f"Processing directory: {dir_}")
    for image_path in os.listdir(dir_path):
        full_image_path = os.path.join(dir_path, image_path)
        image = cv2.imread(full_image_path)
        if image is None:
            logging.error(f"Unable to read image: {full_image_path}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image_rgb)
        # Draw hand landmarks on the image
        if results.multi_hand_landmarks:
            data_aux = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    data_aux.extend([x, y])
            
            if data_aux:  # Only append if landmarks were detected
                data.append(data_aux)
                labels.append(dir_)
        else:
            logging.warning(f"No hand landmarks detected in {full_image_path}")

# Save data
output_file = "data.pickle"
with open(output_file, "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

logging.info(f"Dataset created and saved to {output_file}")
logging.info(f"Total samples: {len(data)}")
