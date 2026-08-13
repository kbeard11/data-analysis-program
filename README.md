# Data Analysis Program
A Python command‑line application for exploring datasets, generating summary statistics, visualising relationships, and training multiple machine‑learning regression models.

## Overview
This project is a Python‑based command‑line application that provides an end‑to‑end workflow for exploring, analysing, and modelling a dataset containing age, height, and weight variables. The program includes dataset inspection tools, summary statistics, exploratory data visualisations, correlation analysis, and multiple supervised machine‑learning regression models.

It is designed as a practical learning tool for Python data analysis, statistics, and machine learning fundamentals.

## Features
###Dataset Inspection
- View first 5 rows
- View last 5 rows
- Review summary of dataset structure.

### Summary Statistics
- Full descriptive statistics
- Variable‑specific statistics for:
    - Age
    - Height
    - Weight
- Mean, median, and standard deviation calculations.

### Exploratory Data Analysis (EDA)
- Histograms for age, height, and weight
- Scatter plots with trend lines:
    - Age → Height
    - Age → Weight
    - Height → Weight
- Group‑by visualisations showing average values per category.

### Correlation Analysis
- Optional correlation matrix heatmap using Seaborn.

### Machine Learning Module
Supports multiple regression models:
- Linear Regression
- Support Vector Regression (SVR)
- Random Forest Regression
- Decision Tree Regression

Includes:
- Train/test split
- Feature scaling (StandardScaler)
- Model training
- Predictions on test data
- Performance metrics:
    - R²
    - MSE
    - RMSE
    - MAE
- Optional bar‑chart comparison of training vs testing performance

## How It Works
The program loads a CSV dataset and presents an interactive menu system. Users can navigate through modules to inspect data, compute statistics, generate visualisations, and train machine‑learning models.

The ML workflow:
- User selects a target variable (height or weight).
- Features are scaled using StandardScaler.
- The chosen model is trained on the training set.
- Predictions are generated for both training and testing sets.
- Performance metrics are displayed.
- Optional visualisation compares training/testing scores side‑by‑side.

## Future Improvements
- Add classification models
- Add PCA dimensionality reduction
- Add hyperparameter tuning
- Add cross‑validation
- Add exportable reports (PDF/HTML)
- Add GUI version (Tkinter or PyQt)
- Add logging and error‑handling improvements

## Why I Built This
This project demonstrates a complete data‑analysis workflow in Python, from raw data to trained machine‑learning models. It serves as a practical learning tool for statistics, EDA, and supervised ML, and provides a strong foundation for more advanced analytics projects.

## License
MIT License
