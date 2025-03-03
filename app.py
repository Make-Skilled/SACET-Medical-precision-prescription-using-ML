import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chardet
import pickle

# Step 1: Detect the file encoding
file_path = "medical_dataset.csv"  # Replace with your dataset file path

with open(file_path, 'rb') as file:
    result = chardet.detect(file.read(10000))  # Detect encoding on a sample
    encoding = result['encoding']
    print(f"Detected file encoding: {encoding}")

# Step 2: Load the dataset with proper encoding
try:
    with open(file_path, 'r', encoding=encoding, errors='replace') as file:
        df = pd.read_csv(file)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Step 3: Rename columns to expected names
df.rename(columns={
    'Questions': 'Symptoms',
    'Recommending medical tests': 'Tests',
    'Disease ': 'Disease'
}, inplace=True)

# Ensure the dataset has the required columns
required_columns = ["Symptoms", "Tests", "Disease"]
if not all(column in df.columns for column in required_columns):
    print(f"Error: Dataset is missing required columns. Found columns: {df.columns}")
    exit()

# Step 4: Save the processed DataFrame as a pickle file
with open("processed_dataframe.pkl", "wb") as df_file:
    pickle.dump(df, df_file)
print("Processed DataFrame saved as 'processed_dataframe.pkl'.")

# Step 5: Define a function to recommend tests and disease based on user input
def recommend_tests_and_disease(user_symptoms, data, vectorizer):
    # Vectorize the symptoms using TF-IDF
    tfidf_matrix = vectorizer.fit_transform(data["Symptoms"])
    user_tfidf = vectorizer.transform([user_symptoms])
    
    # Compute similarity between user input and dataset entries
    similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    # Find the most similar entry
    best_match_idx = similarities.argmax()
    
    # Fetch recommendations from the dataset
    recommended_tests = data.loc[best_match_idx, "Tests"]
    probable_disease = data.loc[best_match_idx, "Disease"]
    
    return recommended_tests, probable_disease

# Step 6: Initialize and save the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words="english")
with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
print("TF-IDF vectorizer saved as 'tfidf_vectorizer.pkl'.")

# Step 7: User Input and Recommendations
user_input = input("Describe your symptoms: ")  # e.g., "I have a fever, cough, and body aches."

try:
    # Reload the saved objects
    with open("processed_dataframe.pkl", "rb") as df_file:
        loaded_df = pickle.load(df_file)
    with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)
    
    # Get recommendations
    recommended_tests, probable_disease = recommend_tests_and_disease(user_input, loaded_df, loaded_vectorizer)
    print(f"\nRecommended Tests: {recommended_tests}")
    print(f"Probable Disease: {probable_disease}")
except Exception as e:
    print(f"Error processing recommendation: {e}")
