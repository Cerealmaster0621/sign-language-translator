import cv2
import os
from constants import DATA_PATH

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def collect_images(number_of_classes, dataset_size, subdirectory):
    create_directory(DATA_PATH)
    create_directory(f"{DATA_PATH}/{subdirectory}")

    cap = cv2.VideoCapture(0)

    for i in range(number_of_classes):
        class_path = f"{DATA_PATH}/{subdirectory}/{i}"
        create_directory(class_path)
        print(f"Collecting data for class {i}")

        while len(os.listdir(class_path)) < dataset_size:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                continue
            
            cv2.putText(frame, f"Class {i}. Image count: {len(os.listdir(class_path))}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to capture or 'q' to quit", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("c"):
                cv2.imwrite(f"{class_path}/{len(os.listdir(class_path))}.jpg", frame)
            elif key == ord("q"):
                break

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    number_of_classes = 3
    dataset_size = 100
    subdirectory = "numbers"
    collect_images(number_of_classes, dataset_size, subdirectory)