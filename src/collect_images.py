import cv2
import os
import string
from config import DATA_PATH

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def collect_images(start, end, dataset_size, subdirectory):
    create_directory(DATA_PATH)
    create_directory(f"{DATA_PATH}/{subdirectory}")

    cap = cv2.VideoCapture(0)

    if subdirectory in ["alphabets", "numbers"]:
        if subdirectory == "alphabets":
            classes = list(string.ascii_lowercase[string.ascii_lowercase.index(start):string.ascii_lowercase.index(end)+1])
        else:
            classes = range(int(start), int(end) + 1)
    else:
        classes = [start]

    for class_name in classes:
        class_path = f"{DATA_PATH}/{subdirectory}/{class_name}"
        create_directory(class_path)
        print(f"Collecting data for class {class_name}")

        while len(os.listdir(class_path)) < dataset_size:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                continue
            
            cv2.putText(frame, f"Class {class_name}. Image count: {len(os.listdir(class_path))}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to capture or 'q' to quit", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("c"):
                cv2.imwrite(f"{class_path}/{len(os.listdir(class_path))}.jpg", frame)
            elif key == ord("q"):
                break

            frame = cv2.flip(frame, 1)  # Flip the frame horizontally

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    collect_images('a', 'd', 100, "alphabets")
    # or
    # collect_images('0', '2', 100, "numbers")
    # or
    # collect_images('custom_gesture', 100, "custom")
