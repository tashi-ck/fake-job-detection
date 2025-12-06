import streamlit as st
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os
import joblib  # Alternative to pickle
import warnings


# --------------------------------------------
# Load BERT Model + Classifier
# --------------------------------------------
@st.cache_resource
def load_bert():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_model():
    model_path = "models/best_bert_model.pkl"

    # First try with pickle
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        #st.success("Model loaded with pickle successfully!")
        return model
    except Exception as e:
        #st.warning(f"Pickle loading failed: {e}")

        # Try with joblib
        try:
            model = joblib.load(model_path)
            #st.success("Model loaded with joblib successfully!")
            return model
        except Exception as je:
            #st.error(f"Joblib loading also failed: {je}")

            # Last resort: Check what type of file it might be
            st.info("Checking file type...")
            with open(model_path, 'rb') as f:
                first_bytes = f.read(100)

            # Check if it might be a PyTorch model
            if b'torch' in first_bytes or b'state_dict' in first_bytes:
                st.info("File appears to be a PyTorch model. Loading with torch...")
                try:
                    import torch
                    model = torch.load(model_path, map_location='cpu')
                    return model
                except Exception as te:
                    st.error(f"Torch loading failed: {te}")

            # If all fails, load a simple placeholder or train a new one
            st.error("Could not load the model file. Using a placeholder model.")
            st.info("You may need to retrain your model.")

            # Create a simple dummy model for demonstration
            from sklearn.linear_model import LogisticRegression
            dummy_model = LogisticRegression()
            # Fit with dummy data so predict works
            import numpy as np
            dummy_model.fit(np.random.randn(10, 384), np.random.randint(0, 2, 10))
            return dummy_model


# Load models
bert = load_bert()
model = load_model()

# --------------------------------------------
# Streamlit UI
# --------------------------------------------
st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🕵️‍♂️", layout="wide")

st.title("🕵️‍♂️ Fake Job Posting Detection")
st.write("Enter job details and the model will predict if the posting is **Real** or **Fake**.")

# --------------------------------------------
# Input Fields
# --------------------------------------------
title = st.text_input("Job Title")
location = st.text_input("Location")
company = st.text_input("Company Name")
description = st.text_area("Job Description", height=200)

if st.button("Predict"):
    if len(description.strip()) == 0:
        st.error("Job description cannot be empty!")
        st.stop()

    # Combine important fields
    combined_text = f"{title} {location} {company} {description}"

    # --------------------------------------------
    # BERT Embedding
    # --------------------------------------------
    embedding = bert.encode([combined_text])

    # --------------------------------------------
    # Prediction
    # --------------------------------------------
    try:
        # Try different prediction methods depending on model type
        if hasattr(model, 'predict'):
            # Scikit-learn style model
            pred = model.predict(embedding)[0]
        elif hasattr(model, '__call__') or hasattr(model, 'forward'):
            # PyTorch or neural network style
            import torch

            with torch.no_grad():
                if isinstance(embedding, np.ndarray):
                    embedding = torch.from_numpy(embedding).float()
                output = model(embedding)
                pred = torch.argmax(output, dim=1).item()
        else:
            # Unknown model type
            st.error("Unknown model type. Cannot make prediction.")
            st.stop()

        # 0 = Real, 1 = Fake
        label = "Fake" if pred == 1 else "Real"

        # --------------------------------------------
        # Output UI
        # --------------------------------------------
        if pred == 1:
            st.error("🚨 The job posting appears **FAKE**!")
        else:
            st.success("✅ The job posting appears **REAL**.")

        st.write("Prediction Code:", label)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("The model might not be compatible with the expected input format.")