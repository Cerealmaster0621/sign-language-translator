import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging

## After creating data.pickle file, run this file to train your model
## Run this file with python train_classifier.py
## The model will be saved as model.pickle file

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the data
try:
    with open("./data.pickle", "rb") as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    logging.error("data.pickle file not found. Please run create_dataset.py first.")
    exit(1)

data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Accuracy: {accuracy:.2%}")

# Print detailed classification report
logging.info("Classification Report:")
logging.info("\n" + classification_report(y_test, y_pred))

# Save the model
output_file = "model.pickle"
try:
    with open(output_file, "wb") as f:
        pickle.dump({"model": model}, f)
    logging.info(f"Model saved to {output_file}")
except IOError:
    logging.error(f"Error saving model to {output_file}")
