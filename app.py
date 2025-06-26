import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
import joblib

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#load th emodel and vetorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

#clean the text
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]"," ",text).lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


st.set_page_config(page_title="Sentiment Analysis App for movies", page_icon=":smiley:", layout="wide")

st.title("Sentiment Analysis App for Movies")
st.subheader("Enter a movie review to analyze its sentiment")
user_input = st.text_area("Enter the review here:")

if st.button("Predict sentiment"):
    cleaned= clean_text(user_input)
    vectorized= vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    st.success(f"The sentiment of the review is: {sentiment}")
