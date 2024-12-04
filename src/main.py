import argparse
import os
from create_dataset import create_and_save_dataset
from train_classifier import train_and_save_classifier
from test_classifier import run_classifier
from collect_images import collect_images
from config import DATA_PATH
import string
import sys

def validate_alphabets(start, end):
    if start not in string.ascii_lowercase or end not in string.ascii_lowercase:
        raise ValueError("Start and end must be lowercase alphabets")
    if start > end:
        raise ValueError("Start alphabet must come before end alphabet")

def main():
    parser = argparse.ArgumentParser(description="Hand Gesture Recognition System", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--collect", nargs=4, metavar=('           SUBDIR', '  START', '   END', ' SIZE'),
                        help="Collect data mode: collect data for training.\n"
                             "SUBDIR: 'alphabets' or 'numbers' or others\n"
                             "START: starting letter/number. When SUBDIR is other than 'alphabets' or 'numbers', START is the name of the subdirectory of SUBDIR\n"
                             "i.e. python main.py --collect conversation hello 100\n"
                             "this will create a folder named 'hello' under 'conversation' under 'data' folder and collect 100 images for 'hello'\n\n"
                             "END: ending letter/number. When SUBDIR is other than 'alphabets' or 'numbers'\n"
                             "SIZE: dataset size")
    parser.add_argument("--test", action="store_true", help="Run the classifier in test mode")
    parser.add_argument("--train", nargs='?', const='all', metavar='SUBDIR',
                        help="Refresh the entire model (retrain the model with data folder).\n"
                             "SUBDIR: Optional. If provided, only train on the specified subdirectory.")

    args = parser.parse_args()
    if args.train:
        if args.train == 'all':
            create_and_save_dataset(DATA_PATH, "data.pickle")
        else:
            subdir_path = os.path.join(DATA_PATH, args.train)
            if not os.path.exists(subdir_path):
                raise ValueError(f"Subdirectory {args.train} does not exist in {DATA_PATH}")
            create_and_save_dataset(subdir_path, "data.pickle")
        train_and_save_classifier("data.pickle", "model.pickle")
    elif args.collect:
        subdirectory, start, end, dataset_size = args.collect
        data_path = f"{DATA_PATH}/{subdirectory}"
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        
        if subdirectory == "alphabets":
            validate_alphabets(start, end)
        elif subdirectory == "numbers":
            start = int(start)
            end = int(end)
            if start > end:
                raise ValueError("Start number must be less than or equal to end number")
        else:
            if not os.path.exists(f"{data_path}/{subdirectory}/{start}"):
                os.makedirs(f"{data_path}/{subdirectory}/{start}")

        dataset_size = int(dataset_size)
        collect_images(start, end, dataset_size, subdirectory)
        create_and_save_dataset(data_path, "data.pickle")
        train_and_save_classifier("data.pickle", "model.pickle")
    elif args.test:
        run_classifier("model.pickle")

if __name__ == "__main__":
    main()
