# Sign Language Translator

A real-time sign language translation system using computer vision and machine learning.

## Demo

![Demo Video](./images/demo.gif)

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
  - Example: `python main.py --collect numbers 0 10 100` => Collect 100 images of numbers from 0 to 10<br><br>
- Train model: `python main.py --train [SUBDIR](optional)`
  - Example: `python main.py --train numbers` => Train the model with the numbers dataset(subdirectory of data folder)<br><br>
- Test model: `python main.py --test(optional)`
  For detailed usage instructions, run: `python main.py --help`<br><br>

## Technologies Used

- Python
- OpenCV
- MediaPipe
- scikit-learn

## License

This project is licensed under the MIT License - see the LICENSE file for details.
