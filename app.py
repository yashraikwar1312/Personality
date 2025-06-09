import streamlit as st
import joblib
import numpy as np

model = joblib.load("personality_model.pkl")

st.title("🧠 Personality Detector (Introvert or Extrovert)")

# Input sliders and dropdowns
alone = st.slider("Time spent alone daily (hours)", 0, 11)
stage_fear = st.selectbox("Do you have stage fear?", ["Yes", "No"])
social = st.slider("Social event attendance (0–10)", 0, 10)
outside = st.slider("How often do you go outside? (0–7)", 0, 7)
drained = st.selectbox("Do you feel drained after socializing?", ["Yes", "No"])
friends = st.slider("Close friends count (0–15)", 0, 15)
posts = st.slider("Social media post frequency (0–10)", 0, 10)

# Encode Yes/No
stage_fear = 1 if stage_fear == "Yes" else 0
drained = 1 if drained == "Yes" else 0

features = np.array([[alone, stage_fear, social, outside, drained, friends, posts]])

if st.button("Predict Personality"):
    result = model.predict(features)
    personality = "Extrovert" if result[0] == 1 else "Introvert"
    st.success(f"You are likely an {personality}")
