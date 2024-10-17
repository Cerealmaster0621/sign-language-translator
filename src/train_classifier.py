import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging
import os

def load_data(file_path):
    try:
        with open(file_path, "rb") as f:
            data_dict = pickle.load(f)
        return np.asarray(data_dict["data"]), np.asarray(data_dict["labels"])
    except FileNotFoundError:
        logging.error(f"{file_path} not found. Please run create_dataset.py first.")
        return None, None

def train_model(data, labels):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return model, x_test, y_test

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy:.2%}")
    logging.info("Classification Report:")
    logging.info("\n" + classification_report(y_test, y_pred))

def save_model(model, output_file):
    try:
        with open(output_file, "wb") as f:
            pickle.dump({"model": model}, f)
        logging.info(f"Model saved to {output_file}")
    except IOError:
        logging.error(f"Error saving model to {output_file}")

def train_and_save_classifier(input_file, output_file):
    logging.basicConfig(level=logging.INFO)
    
    if os.path.exists(output_file):
        os.remove(output_file)
        logging.info(f"Deleted existing {output_file}")
    
    data, labels = load_data(input_file)
    if data is None or labels is None:
        return
    
    model, x_test, y_test = train_model(data, labels)
    evaluate_model(model, x_test, y_test)
    save_model(model, output_file)
