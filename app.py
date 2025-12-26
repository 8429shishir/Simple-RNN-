import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🧠",
    layout="centered"
)

# ------------------------------
# Title & Description
# ------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🧠 Sentiment Analysis on Movie Reviews</h1>
    <p style="text-align:center; font-size:18px;">
    Deep Learning based NLP application using <b>Simple RNN</b> trained on the IMDB dataset.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_trained_model():
    return load_model("model.h5", compile=False)

model = load_trained_model()

# ------------------------------
# IMDB Word Index
# ------------------------------
word_index = imdb.get_word_index()

# ------------------------------
# Text Preprocessing
# ------------------------------
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )
    return padded_review

# ------------------------------
# Prediction Function
# ------------------------------
def predict_sentiment(review):
    processed_review = preprocess_text(review)
    prediction = model.predict(processed_review, verbose=0)[0][0]

    sentiment = "Positive 😊" if prediction >= 0.5 else "Negative 😞"
    confidence = round(float(prediction) * 100, 2)

    return sentiment, confidence

# ------------------------------
# User Input Section
# ------------------------------
st.subheader("✍️ Enter a Movie Review")

review = st.text_area(
    "Type your review below:",
    height=150,
    placeholder="Example: The movie was absolutely fantastic with great acting and story."
)

# ------------------------------
# Prediction Button
# ------------------------------
if st.button("🔍 Analyze Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review before clicking the button.")
    else:
        sentiment, confidence = predict_sentiment(review)

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        if "Positive" in sentiment:
            st.success(f"**Sentiment:** {sentiment}")
        else:
            st.error(f"**Sentiment:** {sentiment}")

        st.info(f"**Confidence Score:** {confidence}%")
        st.markdown("---")
        positive_prob = confidence
        negative_prob = round(100 - confidence, 2)

        # Sentiment label
        if "Positive" in sentiment:
            st.success(f"**Sentiment:** {sentiment}")
        else:
            st.error(f"**Sentiment:** {sentiment}")

        # Probability Bars
        st.markdown("### 📈 Confidence Distribution")

        st.markdown("**Positive Sentiment Probability**")
        st.progress(int(positive_prob))
        st.write(f"{positive_prob}%")

        st.markdown("**Negative Sentiment Probability**")
        st.progress(int(negative_prob))
        st.write(f"{negative_prob}%")




# ------------------------------
# Footer
# ------------------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:14px;">
    Built using TensorFlow, Keras & Streamlit | NLP Project
    </p>
    """,
    unsafe_allow_html=True
)

