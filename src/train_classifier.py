import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging
import os
from xgboost import XGBClassifier

def load_data(file_path):
    try:
        with open(file_path, "rb") as f:
            data_dict = pickle.load(f)
        data = np.asarray(data_dict["data"])
        labels = np.asarray(data_dict["labels"])
        unique_labels = np.unique(labels)
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        labels = np.array([label_to_int[label] for label in labels])
        return data, labels
    except FileNotFoundError:
        logging.error(f"{file_path} not found. Please run create_dataset.py first.")
        return None, None

def train_model(data, labels):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

    # Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train)
    rf_score = rf_model.score(x_test, y_test)
    
    # XGBoost Classifier
    xgb_model = XGBClassifier(n_estimators=100, random_state=42)
    xgb_model.fit(x_train, y_train)
    xgb_score = xgb_model.score(x_test, y_test)
    
    # Determine which model is more efficient
    scores = {'rf': rf_score, 'xgb': xgb_score}
    more_efficient = max(scores, key=scores.get)
    logging.info(f"{more_efficient.upper()} model is more efficient")
    
    # Voting Classifier
    ensemble_model = VotingClassifier(
        estimators=[('rf', rf_model), ('xgb', xgb_model)],
        voting='soft'
    )
    
    ensemble_model.fit(x_train, y_train)
    
    return ensemble_model, x_test, y_test

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
