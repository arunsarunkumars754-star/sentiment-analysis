import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# -----------------------------------
# NLTK
# -----------------------------------

@st.cache_resource
def load_nlp_resources():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()


stop_words, lemmatizer = load_nlp_resources()


# -----------------------------------
# Load Model and BoW
# -----------------------------------

with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("bow_vectorizer.pkl", "rb") as f:
    bow = pickle.load(f)

SENTIMENT_LABELS = {
    0: "Negative",
    1: "Positive",
    2: "Neutral",
}


# -----------------------------------
# NLP Preprocessing
# -----------------------------------

def preprocess_text(text):

    # Convert to string
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Tokenization
    words = text.split()

    # Stopword removal
    words = [
        word
        for word in words
        if word not in stop_words
    ]

    # Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)


# -----------------------------------
# Prediction Function
# -----------------------------------

def predict_sentiment(review):

    # Preprocess
    cleaned_text = preprocess_text(review)

    # BoW transformation
    vector = bow.transform([cleaned_text])

    prediction = int(model.predict(vector)[0])
    probabilities = model.predict_proba(vector)[0]
    class_index = list(model.classes_).index(prediction)
    confidence = float(probabilities[class_index])

    return SENTIMENT_LABELS[prediction], confidence, cleaned_text


# -----------------------------------
# Streamlit Page
# -----------------------------------

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="😊",
    layout="centered"
)


# -----------------------------------
# Title
# -----------------------------------

st.title("😊 AI Sentiment Analyzer")

st.write("Classify a review as positive, neutral, or negative.")


# -----------------------------------
# Input
# -----------------------------------
# -----------------------------------
# Input + Enter to Predict
# -----------------------------------

with st.form("sentiment_form"):

    review = st.text_input(
        "Enter your review",
        placeholder="Type your review and press Enter..."
    )

    submitted = st.form_submit_button(
        "🔍 Analyze Sentiment"
    )


# -----------------------------------
# Prediction
# -----------------------------------

if submitted:

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        sentiment, confidence, cleaned_text = predict_sentiment(
            review
        )

        st.divider()

        st.subheader("Prediction")

        if sentiment == "Positive":
            st.success(sentiment)

        elif sentiment == "Negative":
            st.error(sentiment)

        else:
            st.info(sentiment)

        st.subheader("Confidence")

        st.progress(float(confidence))

        st.write(
            f"{confidence * 100:.2f}%"
        )

        st.subheader("Preprocessed Text")

        st.info(cleaned_text)
