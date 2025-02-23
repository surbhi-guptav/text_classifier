import streamlit as st
import joblib

model = joblib.load("text_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("text classification model")
st.write("Enter a sentence, and I'll tell you its category!")

user_ip = st.text_area("Enter your input here....")

button = st.button("Pedict")

if button:
    if user_ip.strip():
        ip_vec = vectorizer.transform([user_ip])
        result = model.predict(ip_vec)[0]
        st.success(f"**Predicted Category:** {result}")
    else:
        st.warning("Please input your data!")
        