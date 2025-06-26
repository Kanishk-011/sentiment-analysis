# prepare a sentiment analysis of movie dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score
import nltk
import re
from nltk.corpus import stopwords

# downloadd stopword
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

print(stop_words)

# load the dataset
df = pd.read_csv('IMDB Dataset.csv')

df["sentiment"].value_counts()

# mapping the sentiment to some numarical value
df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

#clean the text
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]"," ",text).lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# apply thee clean_text function on review
df["cleaned_review"] = df["review"].apply(clean_text)

df["cleaned_review"].head()

# feature extraction
vectorizer = CountVectorizer(max_features = 5000)
X = vectorizer.fit_transform(df["cleaned_review"])

y= df["sentiment"]

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=1)

y = df["sentiment"]

# train the model
model = MultinomialNB()
model.fit(X_train,y_train)

#make the pred
y_pred = model.predict(X_test)

#calculate the performance matrix
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_pred,y_test)
recall = recall_score(y_pred,y_test)
f1 = f1_score(y_pred,y_test)
cm = confusion_matrix(y_pred,y_test)
classification_rep = classification_report(y_test,y_pred)

print("Accuracy : ", accuracy)
print("Precision : ", precision)
print("Recall : ", recall)
print("F1 Score : ", f1)
print("Confusion Matrix : \n", cm)
print("Classification Report : \n", classification_rep)

#save the moel and vectorizer
import joblib
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved successfully.")



