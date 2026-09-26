import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# -----------------------------------
# NLTK
# -----------------------------------

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# -----------------------------------
# Load Model and BoW
# -----------------------------------

with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("bow_vectorizer.pkl", "rb") as f:
    bow = pickle.load(f)


# -----------------------------------
# NLP Preprocessing
# -----------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


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

    # Prediction
    prediction = model.predict(vector)[0]

    # Probability
    probability = model.predict_proba(vector)[0]

    if prediction == 1:

        sentiment = "Positive 😊"
        confidence = probability[1]

    else:

        sentiment = "Negative 😞"
        confidence = probability[0]

    return sentiment, confidence, cleaned_text


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

st.write(
    "NLP + Machine Learning based sentiment "
    "classification using Bag of Words."
)


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

        if "Positive" in sentiment:
            st.success(sentiment)

        elif "Negative" in sentiment:
            st.error(sentiment)

        else:
            st.warning(sentiment)

        st.subheader("Confidence")

        st.progress(float(confidence))

        st.write(
            f"{confidence * 100:.2f}%"
        )

        st.subheader("Preprocessed Text")

        st.info(cleaned_text)
        # Prediction
        st.subheader("Prediction")

        if "Positive" in sentiment:

            st.success(sentiment)

        else:

            st.error(sentiment)


        # Confidence
        st.subheader("Confidence")

        st.progress(float(confidence))

        st.write(
            f"{confidence * 100:.2f}%"
        )


        # Preprocessed text
        st.subheader("Preprocessed Text")

        st.info(cleaned_text)