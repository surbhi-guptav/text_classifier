import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("text_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_category(input_text, threshold=0.7):
    # Vectorize the input text
    input_vec = vectorizer.transform([input_text])
    
    # Get the probabilities of each class
    probabilities = model.predict_proba(input_vec)[0]
    max_prob = max(probabilities)
    predicted_label = model.classes_[probabilities.argmax()]

    # If the highest probability is below the threshold, return "Unknown"
    if max_prob < threshold:
        return "Unknown"
    else:
        return predicted_label

# Streamlit app UI
st.title("Text Category Classifier")
st.write("Enter a text, and I'll classify it into a category!")

input_text = st.text_area("Enter your text here:")

if st.button("Classify"):
    if input_text.strip():
        predicted_category = predict_category(input_text)
        st.write(f"**Predicted Category:** {predicted_category}")
    else:
        st.warning("Please enter some text to classify.")
