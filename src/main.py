import cv2
import numpy as np
import os

DATA_PATH = "./data"
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

number_of_classes = 10
dataset_size = 1000

cap = cv2.VideoCapture(0)
for i in range(number_of_classes):
    if not os.path.exists(f"{DATA_PATH}/{i}"):
        os.makedirs(f"{DATA_PATH}/{i}")

    print(f"Collecting data for class {i}")

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, "Ready? Press Q", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    
    
