import streamlit as st
import numpy as np
import pandas as pd
from sklearn import metrics
import warnings
import pickle
from feature import FeatureExtraction
import re
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure CPU usage
device = torch.device("cpu")

# Load phishing model
phishing_model_path = r"pickle/model_new.pkl" #changed to relative path 
with open(phishing_model_path, "rb") as file:
    gbc = pickle.load(file)

# error checking temp1
if not os.path.exists(phishing_model_path):
    st.error(f"File not found: {phishing_model_path}. Check if the file is uploaded and the path is correct.")
else:
    with open(phishing_model_path, "rb") as file:
        gbc = pickle.load(file)
# Extract URLs from text
def extract_urls(text):
    # Use regular expression to find URLs starting with 'http' or 'https'
    return re.findall(r'(https?://\S+)', text)

# Function to make predictions on URLs
def predict_link(link):
    # Extract features from the URL using the FeatureExtraction class
    obj = FeatureExtraction(link)
    x = np.array(obj.getFeaturesList()).reshape(1, 30)  # Reshape to fit the model input

    # Make prediction
    y_pred = gbc.predict(x)[0]
    # Get prediction probabilities for phishing and non-phishing
    y_pro_phishing = gbc.predict_proba(x)[0, 0]
    y_pro_non_phishing = gbc.predict_proba(x)[0, 1]

    return y_pred, y_pro_phishing, y_pro_non_phishing


# Load smishing model and tokenizer
smishing_model_path = r"smishing_model" #changed to relative path 
smishing_model = AutoModelForSequenceClassification.from_pretrained(smishing_model_path, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(smishing_model_path, trust_remote_code=True)

# Function to predict smishing
def predict_smishing(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = smishing_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()
    return scores[1], scores[0]  # Return probabilities for "smishing" and "not smishing"

# Gmail API setup (currently disabled)
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
CLIENT_SECRET_FILE = r"D:\CapstoneTest\PhishingLogicER\client_secret.json"

# Main title
st.title("Phishing and Smishing Detector")
st.write("Welcome to ElderRakshak's Detection System!")

# Option to select detection type
option = st.radio("Choose input method:", ('Enter URL', 'Smishing (SMS Text)'))

if option == 'Enter URL':
    # Input URL from user
    url = st.text_input("Enter the URL:")

    if st.button("Predict"):
        if url:
            y_pred, y_pro_phishing, y_pro_non_phishing = predict_link(url)
            if y_pred == 1:
                st.success(f"It is **{y_pro_non_phishing * 100:.2f}%** safe to go.")
            else:
                st.error(f"It is **{y_pro_phishing * 100:.2f}%** unsafe to go.")
        else:
            st.warning("Please enter a URL.")

elif option == 'Smishing (SMS Text)':
    # Input SMS text from user
    sms_text = st.text_area("Enter the SMS text:")
    if st.button("Predict SMS"):
        if sms_text:
            prob_smishing, prob_not_smishing = predict_smishing(sms_text)
            if prob_smishing > prob_not_smishing:
                st.error(f"This SMS is likely a **smishing** attempt ({prob_smishing * 100:.2f}% confidence).")
            else:
                st.success(f"This SMS is not a smishing attempt ({prob_not_smishing * 100:.2f}% confidence).")
        else:
            st.warning("Please enter the SMS text.")

# Run the Streamlit application
if __name__ == "__main__":
    st._is_running_with_streamlit = True
