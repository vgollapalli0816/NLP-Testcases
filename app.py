import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data if not already done
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from git import Repo
import joblib
import os

# Function to load models using GitPython
@st.cache_resource
def load_model_with_git(repo_url, file_path):
    repo_dir = "temp_repo"  # Directory to clone into
    if not os.path.exists(repo_dir):
        Repo.clone_from(repo_url, repo_dir)  # Clone the repository

    model_path = os.path.join(repo_dir, file_path)
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    else:
        raise FileNotFoundError(f"The file  {file_path} does not exist in the repository.")

# Repository URL and model file paths
repo_url = "https://github.com/vgollapalli0816/NLP-Testcases.git"
models_to_load = {
    "knn_model_expected": "knn_model_expected.joblib",
    "knn_model_steps": "knn_model_steps.joblib",
    "tfidf_vectorizer": "tfidf_vectorizer.pkl"
}

# Load each model
loaded_models = {}
try:
    for model_name, file_path in models_to_load.items():
        loaded_models[model_name] = load_model_with_git(repo_url, file_path)
        print(f"{model_name} loaded successfully.")
except Exception as e:
    print(f"An error occurred: {e}")

# Optional cleanup: Remove the cloned repo after use
import shutil
if os.path.exists("temp_repo"):
    shutil.rmtree("temp_repo")


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

    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

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

knn_model_results = loaded_models.get("knn_model_expected")
knn_model_steps = loaded_models.get("knn_model_steps")
tfidf_vectorizer = loaded_models.get("tfidf_vectorizer")

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
