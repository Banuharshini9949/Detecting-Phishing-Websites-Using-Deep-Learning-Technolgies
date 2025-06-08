# Detecting Phishing Websites Using Deep Learning Technolgies
Phishing attacks are one of the most widespread cyber threats, tricking users into revealing sensitive information through seemingly legitimate websites. This project presents a deep learning-based solution using a Fully Connected Neural Network (FCNN) to detect phishing websites with high accuracy.

# Table of Contents
- Overview
- Model Architecture
- Dataset Description
- Technologies Used
- Model Performance
- GUI Application
- End Results

## Overview
As internet usage grows, phishing websites are becoming increasingly sophisticated. Traditional rule-based detection methods are not adaptive enough to handle evolving attack strategies. This project addresses these shortcomings by leveraging deep learning techniques to develop a robust phishing detection system.

- Detects phishing websites using a deep learning model.
- Uses an FCNN that learns features directly from raw URL-based data.

## Model Architecture

The core model is a **Fully Connected Neural Network (FCNN)** with the following structure:

- Input Layer: Scaled URL-based features
- Hidden Layers:
  - Dense (128 units, ReLU) + Dropout
  - Dense (64 units, ReLU) + Dropout
  - Dense (32 units, ReLU)
- Output Layer: 1 neuron (Sigmoid activation for binary classification)

**Loss Function**: Binary Crossentropy  
**Optimizer**: Adam

## Dataset Description

- The dataset includes URL-based features labeled as **phishing (1)** or **legitimate (0)**.
- Preprocessing steps:
  - Handling missing values
  - Feature selection
  - Feature scaling (StandardScaler/MinMaxScaler)
 
## Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy / Pandas**
- **Matplotlib / Seaborn**
- **Tkinter (for GUI)**
- **Scikit-learn**

## Model Performance

| Metric         | Score    |
|----------------|----------|
| Training Accuracy | 85%   |
| Testing Accuracy  | 84%   |
| Loss (Val)        | ~0.32 |

![image](https://github.com/user-attachments/assets/3fba58ed-e04b-470a-9278-3f7e2ad6399a)

## GUI Application

A Tkinter-based graphical user interface allows users to:
- Input a domain or URL
- Click on “Predict” to see if it is **Legitimate** or **Phishing**
- Reset input fields for new predictions

## End Results
After training and evaluating the model, here are the final results of the phishing website detection system:

Metric	Value:
Training Accuracy	85%
Testing Accuracy	84%
Validation Loss	~0.32
Model Type	Fully Connected Neural Network (FCNN)
Feature Engineering	Scaled & selected URL-based features
Prediction Output	Legitimate (0) or Phishing (1)
Interface Type	Tkinter GUI

The model demonstrates strong capability in classifying phishing URLs, achieving high accuracy without the need for manual feature engineering. The GUI enables users to interactively test URLs and receive instant predictions.
![image](https://github.com/user-attachments/assets/67b0648f-68f7-4754-a0a8-119fd1a0ceaa)
![image](https://github.com/user-attachments/assets/a67edc0c-dd3e-471e-b789-5c07dbccb374)

