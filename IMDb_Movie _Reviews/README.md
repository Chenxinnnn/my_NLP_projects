# IMDb_Movie_Reviews

This project focuses on building a robust movie review sentiment classifier using BERT, and enhancing model performance under real-world noisy inputs by introducing typo-based data augmentation.

## Directory Structure
```
robust_text_classification/
├── main.py                          # Main script to train and evaluate the model
├── utils.py                         # Typo generation and file handling utilities
├── requirements.txt                 # Python environment dependencies
├── out_original.txt                 # Predictions on original clean test set
├── out_transformed.txt              # Predictions on typo-injected noisy test set
├── out_augmented_original.txt       # Trained with augmented data, tested on clean test set
├── out_augmented_transformed.txt    # Trained with augmented data, tested on noisy test set
└── README.md                        # This file
```

## Project Overview

Goal:
- Build a movie review sentiment classifier that is resilient to real-world typos and text noise.

Method:
- Fine-tune a pre-trained BERT model on the IMDb Movie Reviews dataset.
- Create a noisy version of the data by injecting realistic keyboard typos into words.
- Apply data augmentation: augment 30% of training samples with synthetic typos.
- Evaluate the model’s robustness before and after augmentation.

## Dataset

Name: IMDb Movie Reviews
Task: Binary classification (Positive sentiment = 1, Negative sentiment = 0)
Size: 25,000 training examples, 25,000 test examples
Source: HuggingFace Datasets

Example loading:
from datasets import load_dataset
dataset = load_dataset("imdb")

## Approach

1. Baseline Model
- Model: bert-base-uncased
- Trained on clean training data
- Tested on clean test set (out_original.txt)

2. Typo-Based Perturbation
- Randomly insert typos based on QWERTY keyboard neighbors
- Control typo frequency (about 1–2 typos per word)
- Generate noisy test set (out_transformed.txt)

3. Augmented Training
- Inject typos into 30% of the training samples
- Retrain BERT with the augmented training set
- Evaluate on both clean and noisy test sets (out_augmented_original.txt, out_augmented_transformed.txt)

## Typo Simulation Example

- "movie" becomes "movue" or "mpvie"
- "great" becomes "greqt"

Typo simulation uses realistic keyboard neighbor errors.

## Output Files

out_original.txt: Predictions on clean test data without augmentation
out_transformed.txt: Predictions on typo-injected noisy test data
out_augmented_original.txt: Predictions from model trained with augmented data on clean test data
out_augmented_transformed.txt: Predictions from model trained with augmented data on noisy test data

Each file contains one prediction (0 or 1) per line.

## Key Findings

- Standard BERT models perform poorly when input text contains typos.
- Training with typo-augmented data significantly improves robustness on noisy inputs.
- Minor decrease in clean test accuracy, but substantial robustness gain.

## Requirements

transformers
datasets
torch
tqdm

Install all dependencies:
pip install -r requirements.txt

