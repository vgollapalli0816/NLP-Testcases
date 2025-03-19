import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import urllib.request

# Download necessary NLTK data if not already done
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

import numpy as np
import io
import joblib

# Function to download and load a model
def load_model(url):
    with urllib.request.urlopen(url) as response:
        content = response.read()
        return joblib.load(io.BytesIO(content))

# Load all models into variables
knn_model_results = load_model("https://raw.githubusercontent.com/vgollapalli0816/NLP-Testcases/main/knn_model_expected.joblib")
knn_model_steps = load_model("https://raw.githubusercontent.com/vgollapalli0816/NLP-Testcases/main/knn_model_steps.joblib")
tfidf_vectorizer = load_model("https://raw.githubusercontent.com/vgollapalli0816/NLP-Testcases/main/tfidf_vectorizer.pkl")

lemmatizer = WordNetLemmatizer()

# Function to preprocess input text
def preprocess_text(text):
    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.strip()  # Remove leading/trailing whitespaces

    # 2. Tokenize the text
    tokens = word_tokenize(text)

    # 3. Lowercasing
    tokens = [token.lower() for token in tokens]

    # 5. Lemmatization
    tokens_lemmatized = [lemmatizer.lemmatize(token) for token in tokens]

    # Rejoin tokens into cleaned text string
    cleaned_text = ' '.join(tokens_lemmatized)
    return cleaned_text

# Function to generate test steps using the trained KNN model
def generate_output(acceptance_criteria, model, vectorizer):
    # Preprocess the acceptance criteria text
    cleaned_text = preprocess_text(acceptance_criteria)    
    # Generate embeddings using the same embedding model
    acceptance_criteria_vector = vectorizer.transform([cleaned_text])
    
    # Use the KNN model to predict the closest test steps (using the generated embeddings)
    predicted_test_steps = model.predict(acceptance_criteria_vector)
    
    # Return the predicted test steps (you can customize this logic further)
    action_keywords = ['user', 'tap', 'verify', 'click', 'navigate']
    
    # Use regex to insert a special delimiter before each action keyword
    for keyword in action_keywords:
        predicted_test_steps[0] = re.sub(r'(\s?)' + keyword, r'.\1' + keyword, predicted_test_steps[0], flags=re.IGNORECASE)
    
    # Split the text into individual points using the period as delimiter
    test_steps = predicted_test_steps[0].split('.')
    
    # Remove any empty or unnecessary points
    test_steps = [step.strip() for step in test_steps if step.strip()]
    
    # Format the list of points for display (numbering each test step)
    formatted_test_steps = "\n".join([f"{i+1}. {step.strip()}" for i, step in enumerate(test_steps)])
    
    return formatted_test_steps

def generate_output_result(acceptance_criteria, model, vectorizer):
    # Preprocess the acceptance criteria text
    cleaned_text = preprocess_text(acceptance_criteria)
    
    # Generate embeddings using the same embedding model
    acceptance_criteria_vector = vectorizer.transform([cleaned_text])
    
    # Use the KNN model to predict the closest test steps (using the generated embeddings)
    predicted_test_steps = model.predict(acceptance_criteria_vector)

    return predicted_test_steps

# Streamlit UI
st.title("Test Case Steps Generator")
st.write("This app generates test steps based on Test Case Acceptance Criteria.")
# User input
acceptance_criteria = st.text_area("Enter Test Case Acceptance Criteria")

# Generate test steps and result when the user enters acceptance criteria
if st.button("Generate Test Steps and Results"):
    if acceptance_criteria:
        test_steps = generate_output(acceptance_criteria, knn_model_steps, tfidf_vectorizer)
        expected_results = generate_output_result(acceptance_criteria, knn_model_results, tfidf_vectorizer)
        st.subheader("Generated Test Steps")
        st.write(test_steps)
        
        st.subheader("Generated Expected Results")
        expected_results_str = " ".join(expected_results)
        st.markdown(expected_results_str)
    else:
        st.warning("Please enter the Acceptance Criteria text first.")
