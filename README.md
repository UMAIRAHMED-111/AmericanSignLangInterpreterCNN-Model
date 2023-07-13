# American Sign Language Interpreter Python Program

This repository contains a Python program for an American Sign Language (ASL) interpreter. The program utilizes the dataset available at [Kaggle's Synthetic ASL Alphabet](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet) to recognize and interpret hand gestures corresponding to individual letters of the ASL alphabet.

## Dataset

The ASL interpreter program uses the Synthetic ASL Alphabet dataset available on Kaggle. The dataset consists of 87,000 images of individual letters, each representing a specific hand gesture from the ASL alphabet. These images are collected using synthetic hand models and provide a diverse range of hand poses, lighting conditions, and backgrounds.

The dataset is not included in this repository but can be downloaded from [here](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet). Please ensure you have the dataset available locally before running the program.

## Requirements

To run the ASL interpreter program, you need to have the following software and libraries installed:

- Python 3.6 or above
- OpenCV (cv2)
- TensorFlow
- Keras
- Numpy
- Pandas
- Matplotlib

You can install the required Python libraries using pip with the following command:


## Usage

1. Clone this repository to your local machine or download the program file (`asl_interpreter.py`) directly.
2. Download the Synthetic ASL Alphabet dataset from [Kaggle](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet).
3. Extract the dataset and ensure that the images are organized in the appropriate folder structure.
4. Update the `dataset_path` variable in the `asl_interpreter.py` file to point to the directory where you extracted the dataset.
5. Open a terminal or command prompt and navigate to the directory containing `asl_interpreter.py`.
6. Run the program using the following command:


Please note that the accuracy of the ASL interpreter depends on the quality of the dataset and the training of the CNN model. Feel free to experiment with different datasets and models to improve the performance.

## Acknowledgments

- The Synthetic ASL Alphabet dataset used in this program was created by the [LexSet](https://www.lexset.ai/) team and made available on Kaggle.
- Freecodecamp Tensorflow Keras Code.


