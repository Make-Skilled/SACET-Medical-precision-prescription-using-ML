import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import json

# Load the preprocessed DataFrame and TF-IDF vectorizer
with open("processed_dataframe.pkl", "rb") as df_file:
    df = pickle.load(df_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load diseases and medications
with open("diseases.json", "rb") as diseases_file:
    diseases_data = json.load(diseases_file)

# Function to preprocess user input
def preprocess_input(text):
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = " ".join(text.split())
    return text

# Function to get medications for a disease
def get_medications(disease_name):
    # Clean up disease name (remove trailing periods and spaces)
    disease_name = disease_name.strip().rstrip('.')
    
    # Try exact match first
    for disease in diseases_data["diseases"]:
        if disease["name"].lower() == disease_name.lower():
            return disease["medications"]
    
    # If no exact match, try partial match
    for disease in diseases_data["diseases"]:
        if disease["name"].lower().startswith(disease_name.lower()) or \
           disease_name.lower().startswith(disease["name"].lower()):
            return disease["medications"]
    
    return []

# Function to recommend tests and diseases with confidence scores
def recommend_tests_and_diseases(user_symptoms, data, vectorizer):
    # Vectorize the symptoms using the loaded TF-IDF vectorizer
    tfidf_matrix = vectorizer.fit_transform(data["Symptoms"])
    user_tfidf = vectorizer.transform([user_symptoms])
    
    # Compute similarity between user input and dataset entries
    similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    # Get top match
    top_index = similarities.argmax()
    confidence_score = similarities[top_index]
    
    # Calculate confidence percentage
    confidence_score = (confidence_score * 100)
    
    # Get disease name
    disease = data.loc[top_index, "Disease"]
    
    # Fetch recommendation from the dataset
    recommendation = {
        'tests': data.loc[top_index, "Tests"],
        'disease': disease,
        'confidence': confidence_score,
        'medications': get_medications(disease)
    }
    
    return recommendation

# Set page config
st.set_page_config(
    page_title="Medical Recommendation System",
    page_icon="🏥",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .recommendation-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: #f0f2f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .confidence-score {
        color: #0066cc;
        font-weight: bold;
    }
    .medication-section {
        margin-top: 1.5rem;
        padding: 1rem;
        background-color: #e8f4ff;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🏥 Medical Recommendation System")
st.markdown("""
    This system helps recommend medical tests based on your symptoms and suggests possible conditions.
    Please describe your symptoms in detail for better recommendations.
""")

# Disclaimer
st.warning("""
    ⚠️ **Medical Disclaimer**: This system is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
""")

# Example symptoms
with st.expander("📝 Example Symptoms"):
    st.markdown("""
        Here are some example ways to describe your symptoms:
        - "I have a persistent cough, fever, and difficulty breathing"
        - "I'm experiencing chest pain and shortness of breath"
        - "I have frequent headaches and blurred vision"
        - "I notice unexplained weight loss and fatigue"
    """)

# User input section
st.header("🔍 Describe Your Symptoms")
user_input = st.text_area(
    "Please enter your symptoms in detail:",
    placeholder="E.g., I have been experiencing persistent cough and fever for the past few days...",
    height=100
)

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Get Recommendation", key="recommend_button"):
    if user_input.strip():
        try:
            # Preprocess user input
            processed_input = preprocess_input(user_input)
            
            # Get recommendation
            recommendation = recommend_tests_and_diseases(processed_input, df, vectorizer)
            
            # Display results
            st.header("📋 Medical Recommendation")
            
            with st.container():
                # Display disease and confidence
                st.markdown(f"""
                <div class="recommendation-box">
                    <h3>🏥 Probable Condition:</h3>
                    <p style='font-size: 1.2em; font-weight: bold;'>{recommendation['disease']}</p>
                    <p class='confidence-score'>Confidence: {recommendation['confidence']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display recommended tests
                st.subheader("🔬 Recommended Tests:")
                tests = [test.strip() for test in recommendation['tests'].split(',')]
                for test in tests:
                    st.markdown(f"- {test}")
                
                # Display medications in a separate styled section
                if recommendation['medications']:
                    st.markdown("""
                    <div class="medication-section">
                        <h3>💊 Common Medications:</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    for med in recommendation['medications']:
                        st.markdown(f"- {med}")
                
            # Add note about confidence scores
            st.markdown("""
                <br>
                <small>Note: The confidence score indicates the system's certainty in the recommendation. 
                Higher percentages suggest better matches with your described symptoms.</small>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred while processing your request: {str(e)}")
    else:
        st.warning("⚠️ Please enter some symptoms before requesting recommendations.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <small>Made with ❤️ by the Medical Recommendation System Team</small>
    </div>
""", unsafe_allow_html=True)
