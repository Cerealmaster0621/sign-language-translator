# Sign Language Translator

A real-time sign language translation system using computer vision and machine learning.

## Features

- Real-time hand gesture recognition
- Support for alphabet and number gestures (or custom gestures such as words, sentences, etc.)
- Train your own model with your own dataset
- Easy-to-use command-line interface

## Installation

```bash
git clone https://github.com/Cerealmaster0621/sign-language-translator.git # Clone the repository
pip install -r requirements.txt # Install dependencies
cd sign-language-translator/src # Go to the src directory
python3 main.py --help
```

## Usage

- Collect data: `python main.py --collect [SUBDIR] [START] [END] [SIZE]`
- Train model: `python main.py --train [SUBDIR](optional)`
- Test model: `python main.py --test(optional)`

For detailed usage instructions, run: `python main.py --help`

## Technologies Used

- Python
- OpenCV
- MediaPipe
- scikit-learn

## License

This project is licensed under the MIT License - see the LICENSE file for details.
