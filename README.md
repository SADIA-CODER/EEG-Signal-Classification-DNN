# EEG Signal Classification using Deep Neural Networks

An Artificial Intelligence-based EEG signal classification system that uses statistical feature engineering and a Deep Neural Network (DNN) to classify EEG recordings into **Alcoholic** and **Control** categories.

The project includes complete data preprocessing, exploratory data analysis (EDA), statistical feature extraction, feature normalization, DNN model training, performance evaluation, and an interactive Streamlit dashboard for visualization.

---

## Overview

Electroencephalography (EEG) is a non-invasive technique used to record the electrical activity of the brain through electrodes placed on the scalp. EEG signals are widely used in neurological research, medical analysis, cognitive studies, and brain-computer interface applications.

However, EEG signals are complex, non-linear, non-stationary, and sensitive to noise and artifacts. Analyzing these signals manually can therefore be difficult and time-consuming.

This project develops an AI-based system that automatically classifies EEG recordings into two categories:

- **Alcoholic**
- **Control (Non-Alcoholic)**

Instead of directly feeding the high-dimensional raw EEG signals into the neural network, statistical characteristics are extracted from the signals. These features are then normalized and provided to a Deep Neural Network for binary classification.

The system also includes an interactive **Streamlit dashboard** that allows users to explore the dataset, visualize EEG signals, examine feature correlations, and view model performance.

---

## Objectives

The main objectives of this project are:

1. Preprocess and organize the provided EEG datasets.
2. Clean the data and handle missing values.
3. Assign binary class labels to Alcoholic and Control EEG recordings.
4. Perform Exploratory Data Analysis (EDA).
5. Extract meaningful statistical features from EEG signals.
6. Normalize the extracted features using StandardScaler.
7. Design and implement a Deep Neural Network for binary classification.
8. Train, validate, and test the DNN model.
9. Evaluate the model using standard classification metrics.
10. Generate a confusion matrix and ROC curve.
11. Calculate the Area Under the Curve (AUC).
12. Develop an interactive Streamlit dashboard for data and result visualization.

---

## Project Workflow

The complete workflow of the system is:

```text
                    EEG Dataset
                         |
                         v
                Data Collection
                         |
                         v
                 Data Cleaning
                         |
                         v
              Class Label Creation
                         |
                         v
              Exploratory Data Analysis
                         |
                         v
             Statistical Feature Extraction
                         |
                         v
                Feature Normalization
                    StandardScaler
                         |
                         v
                  Train/Test Split
                      80 / 20
                         |
                         v
              Deep Neural Network
                         |
                         v
                   Model Training
                         |
                         v
                  Model Evaluation
                         |
          +--------------+--------------+
          |              |              |
       Accuracy      Confusion       ROC-AUC
                     Matrix
                         |
                         v
              Streamlit Dashboard
