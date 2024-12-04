import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
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
        
        # Extract the actual label from the prefix format
        cleaned_labels = np.array([label.split('_')[1] for label in labels])
        unique_labels = sorted(np.unique(cleaned_labels))
        
        logging.info(f"Loaded dataset with {len(unique_labels)} unique classes")
        logging.info(f"Labels: {unique_labels}")
        return data, cleaned_labels
    except FileNotFoundError:
        logging.error(f"{file_path} not found. Please run create_dataset.py first.")
        return None, None

def train_model(data, labels):
    # Create a LabelEncoder instance
    le = LabelEncoder()
    # Fit and transform the labels
    encoded_labels = le.fit_transform(labels)
    
    x_train, x_test, y_train, y_test = train_test_split(
        data, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )

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
        voting='hard'
    )
    
    ensemble_model.fit(x_train, y_train)
    y_pred = ensemble_model.predict(x_test)
    
    # Convert predictions back to original labels for the report
    y_test_original = le.inverse_transform(y_test)
    y_pred_original = le.inverse_transform(y_pred)
    
    accuracy = accuracy_score(y_test_original, y_pred_original)
    report = classification_report(y_test_original, y_pred_original)
    
    logging.info(f"Accuracy: {accuracy:.2%}")
    logging.info("Classification Report:")
    logging.info(f"\n{report}")
    
    # Save the label encoder along with the model
    return {"model": ensemble_model, "label_encoder": le}, x_test, y_test

def train_and_save_classifier(data_file, model_file):
    data, labels = load_data(data_file)
    if data is None or labels is None:
        return
    
    model_dict, x_test, y_test = train_model(data, labels)
    
    with open(model_file, "wb") as f:
        pickle.dump(model_dict, f)
    logging.info(f"Model saved to {model_file}")
