import os
import mediapipe as mp
import cv2
import pickle
import logging
from typing import List, Tuple
from config import DATA_PATH

def setup_hand_detection():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

def process_image(image_path: str, hands) -> Tuple[List[int], str]:
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Unable to read image: {image_path}")
        return None, None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        logging.warning(f"No hand landmarks detected in {image_path}")
        return None, None

    data_aux = []
    for hand_landmarks in results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            data_aux.extend([x, y])

    return data_aux, os.path.basename(os.path.dirname(image_path))

def create_dataset(data_path: str) -> Tuple[List[List[int]], List[str]]:
    hands = setup_hand_detection()
    data = []
    labels = []

    if not os.path.exists(data_path):
        logging.error(f"Error: Directory not found: {data_path}")
        return data, labels

    # Process numbers first
    numbers_path = os.path.join(DATA_PATH, "numbers")
    if os.path.exists(numbers_path):
        logging.info(f"Processing numbers directory: {numbers_path}")
        for dir_ in sorted(os.listdir(numbers_path)):
            dir_path = os.path.join(numbers_path, dir_)
            if not os.path.isdir(dir_path):
                continue
            for image_name in os.listdir(dir_path):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_image_path = os.path.join(dir_path, image_name)
                    data_aux, label = process_image(full_image_path, hands)
                    if data_aux and label:
                        data.append(data_aux)
                        labels.append(f"num_{label}")

    # Then process alphabets
    alphabets_path = os.path.join(DATA_PATH, "alphabets")
    if os.path.exists(alphabets_path):
        logging.info(f"Processing alphabets directory: {alphabets_path}")
        for dir_ in sorted(os.listdir(alphabets_path)):
            dir_path = os.path.join(alphabets_path, dir_)
            if not os.path.isdir(dir_path):
                continue
            for image_name in os.listdir(dir_path):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_image_path = os.path.join(dir_path, image_name)
                    data_aux, label = process_image(full_image_path, hands)
                    if data_aux and label:
                        data.append(data_aux)
                        labels.append(f"alpha_{label}")

    if not data:
        logging.error("No valid images found in the dataset.")
    else:
        logging.info(f"Total samples processed: {len(data)}")

    return data, labels

def save_dataset(data: List[List[int]], labels: List[str], output_file: str):
    with open(output_file, "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)
    logging.info(f"Dataset created and saved to {output_file}")
    logging.info(f"Total samples: {len(data)}")

## After collecting images, run this file to create a dataset for your model
## Run this file with python create_dataset.py
## The images will be saved as pickle file (data.pickle)
def create_and_save_dataset(data_path: str, output_file: str):
    logging.basicConfig(level=logging.INFO)
    if os.path.exists(output_file):
        os.remove(output_file)
        logging.info(f"Deleted existing {output_file}")
    data, labels = create_dataset(data_path)
    save_dataset(data, labels, output_file)
