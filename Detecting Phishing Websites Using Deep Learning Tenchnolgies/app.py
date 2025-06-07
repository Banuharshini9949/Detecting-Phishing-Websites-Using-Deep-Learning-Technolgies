# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
from urllib.parse import urlparse
import re
import socket

import re
from urllib.parse import urlparse

def extract_features(domain):
    features = {}
    features['DomainLength'] = len(domain)
    features['NumDots'] = domain.count('.')
    features['NumHyphens'] = domain.count('-')
    features['HasAtSymbol'] = int('@' in domain)
    features['HasIPAddress'] = int(bool(re.search(r'(\d{1,3}\.){3}\d{1,3}', domain)))
    features['UsesHTTPS'] = int(domain.startswith("https"))
    features['NumDigits'] = sum(c.isdigit() for c in domain)
    features['NumSpecialChars'] = sum(c in '@!?$&#%' for c in domain)
    return list(features.values())


# Load dataset
data = pd.read_csv('urldata.csv')
X = data.drop(columns=['Domain', 'Label'])
y = data['Label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define FCNN model
def create_fcnn_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train model
fcnn_model = create_fcnn_model()
fcnn_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

# Feature extractor from domain name
def extract_features_from_domain(domain):
    parsed = urlparse(domain)
    hostname = parsed.netloc if parsed.netloc else parsed.path

    have_ip = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', hostname) else -1
    have_at = 1 if "@" in domain else -1
    url_length = len(domain)
    url_depth = domain.count('/')
    redirection = 1 if "//" in domain[domain.find("//")+2:] else -1
    https_domain = 1 if "https" in parsed.netloc else -1
    tinyurl = 1 if len(domain) < 20 else -1
    prefix_suffix = 1 if '-' in hostname else -1
    dns_record = 1
    web_traffic = 1
    domain_age = 1
    domain_end = 1
    iframe = -1
    mouse_over = -1
    right_click = -1
    web_forwards = 0

    return [
        have_ip, have_at, url_length, url_depth, redirection,
        https_domain, tinyurl, prefix_suffix, dns_record, web_traffic,
        domain_age, domain_end, iframe, mouse_over, right_click, web_forwards
    ]

# Predict by domain

def predict_by_domain(domain_name):
    domain_data = data[data['Domain'] == domain_name]
    if not domain_data.empty:
        features = domain_data.drop(columns=['Domain', 'Label']).iloc[0].tolist()
    else:
        features = extract_features_from_domain(domain_name)

    features_scaled = scaler.transform([features])
    prediction = fcnn_model.predict(features_scaled)[0][0]
    result = "Phishing" if prediction >= 0.5 else "Safe"
    return domain_name, result

# GUI Background
def add_background(root):
    bg_image_path = "background.jpg"
    try:
        bg_image = Image.open(bg_image_path)
        bg_image = bg_image.resize((1500, 1200), Image.Resampling.LANCZOS)
        bg_photo = ImageTk.PhotoImage(bg_image)
        bg_label = tk.Label(root, image=bg_photo)
        bg_label.image = bg_photo
        bg_label.place(relwidth=1, relheight=1)
    except FileNotFoundError:
        messagebox.showerror("Error", "Background image not found. Please ensure 'background.jpg' exists.")

# Homepage
def show_homepage():
    clear_window()
    add_background(root)
    title_label = tk.Label(root, text="PHISHING WEBSITE DETECTION", font=("Arial", 30, "bold"))
    title_label.pack(pady=20)

    buttons = [
        ("Upload Dataset", upload_dataset),
        ("Preview Dataset", preview_dataset),
        ("Testing", show_testing_section),
        ("Data Visualization", show_visualization_dashboard),
        ("How It Works", show_how_it_works)
    ]

    for text, command in buttons:
        btn = tk.Button(root, text=text, command=command, font=("Arial", 14), width=20)
        btn.pack(pady=10)

# Upload dataset
def upload_dataset():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        global data, X, y, X_scaled, X_train, X_test, y_train, y_test
        data = pd.read_csv(file_path)
        X = data.drop(columns=['Domain', 'Label'])
        y = data['Label']
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        messagebox.showinfo("Success", "Dataset uploaded and processed successfully!")

# Preview dataset
def preview_dataset():
    clear_window()
    add_background(root)
    tk.Label(root, text="Dataset Preview", font=("Arial", 20)).pack(pady=10)
    text = tk.Text(root, wrap=tk.WORD, height=20, width=80)
    preview_text = data.head().to_string() + "\n\n" + data.tail().to_string()
    text.insert(tk.END, preview_text)
    text.pack(pady=10)
    tk.Button(root, text="Back to Homepage", command=show_homepage, font=("Arial", 14)).pack(pady=10)

# Testing Section
def show_testing_section():
    clear_window()
    add_background(root)

    tk.Label(root, text="Testing Section", font=("Arial", 20)).pack(pady=10)
    tk.Label(root, text="Enter Domain Name:", font=("Arial", 14)).pack(pady=10)

    domain_input = tk.Entry(root, width=50, font=("Arial", 14))
    domain_input.pack(pady=10)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    def predict_action():
        domain_name = domain_input.get().strip()
        if not domain_name:
            messagebox.showerror("Input Error", "Please enter a domain name.")
            return
        _, prediction = predict_by_domain(domain_name)
        result_label.config(text=f"Prediction: {prediction}", fg="green" if prediction == "Safe" else "red")

    def reset_action():
        domain_input.delete(0, tk.END)
        result_label.config(text="")

    tk.Button(button_frame, text="Predict", command=predict_action, font=("Arial", 14)).grid(row=0, column=0, padx=10)
    tk.Button(button_frame, text="Reset", command=reset_action, font=("Arial", 14)).grid(row=0, column=1, padx=10)

    result_label = tk.Label(root, text="", font=("Arial", 16))
    result_label.pack(pady=10)

    tk.Button(root, text="Back to Homepage", command=show_homepage, font=("Arial", 14)).pack(pady=10)

# Visualization
def show_visualization_dashboard():
    clear_window()
    add_background(root)

    tk.Label(root, text="Data Visualization Dashboard", font=("Arial", 20)).pack(pady=10)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    data['Label'].value_counts().plot.pie(ax=axs[0], autopct="%1.1f%%", labels=["Safe", "Phishing"],
                                          colors=['#4CAF50', '#F44336'])
    axs[0].set_title("Distribution of Safe vs. Phishing")

    feature_means = X.mean()
    feature_names = X.columns
    axs[1].barh(feature_names, feature_means, color="#3E7BFA")
    axs[1].set_title("Feature Distribution")
    plt.tight_layout()
    plt.savefig("visualization.png")
    plt.close()

    img = Image.open("visualization.png")
    img = ImageTk.PhotoImage(img)
    label_img = tk.Label(root, image=img)
    label_img.image = img
    label_img.pack()

    tk.Button(root, text="Back to Homepage", command=show_homepage, font=("Arial", 14)).pack(pady=10)

# How It Works
def show_how_it_works():
    clear_window()
    add_background(root)
    tk.Label(root, text="How It Works", font=("Arial", 20)).pack(pady=10)
    explanation = (
        "1. Dataset Processing:\n"
        "   - Upload a dataset containing domain attributes.\n"
        "   - The data is preprocessed, scaled, and split into training/testing.\n\n"
        "2. Model Training:\n"
        "   - A neural network learns to classify 'Safe' vs 'Phishing'.\n\n"
        "3. Prediction:\n"
        "   - Enter a domain name to test.\n"
        "   - If not in dataset, features are extracted manually.\n\n"
        "4. Visualization:\n"
        "   - See distribution of data and feature metrics."
    )
    tk.Label(root, text=explanation, font=("Arial", 14), wraplength=600, justify="left").pack(pady=10)
    tk.Button(root, text="Back to Homepage", command=show_homepage, font=("Arial", 14)).pack(pady=10)

# Clear Window
def clear_window():
    for widget in root.winfo_children():
        widget.destroy()

# Main App Loop
root = tk.Tk()
root.title("Phishing Website Detection")
root.geometry("1024x768")
show_homepage()
root.mainloop()
