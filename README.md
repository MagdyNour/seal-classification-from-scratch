# Seal Classification from Scratch

## Overview

A machine learning project exploring image-based seal classification using algorithms implemented in Python. The project compares a custom decision tree classifier with a k-nearest neighbours (k-NN) approach.

## Methods

- Image data loading and preprocessing with NumPy
- Decision tree classification
- k-nearest neighbours classification
- Hyperparameter tuning
- Confusion matrix analysis and misclassification visualization

## Project Structure

```text
seal-classification-from-scratch/
├── data/       # Local datasets (not tracked by Git)
├── figures/    # Generated visualizations
├── src/
│   └── seals.py
├── .gitignore
└── README.md
```

## Requirements

- Python 3
- NumPy
- Matplotlib
- scikit-learn (if used by the script)

## Data

The original CSV datasets are not included in this repository. They must be obtained separately, subject to the applicable course and data-sharing permissions.

## Running the Project

From the project root, run:

```bash
python src/seals.py
```

## Notes

This project was developed in an academic context. Public release and redistribution of the source code or data are subject to the course's applicable rules and permissions.