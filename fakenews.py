# Fake News Detection using NLP (Advanced Project)
import pandas as pd
import numpy as np
import string
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data = pd.read_csv("Fake.csv")

# Label the data
data["label"] = 0  # 0 for fake
real = pd.read_csv("True.csv")
real["label"] = 1  # 1 for real

# Merge and shuffle
df = pd.concat([data, real], axis=0)
df = df.sample(frac=1).reset_index(drop=True)

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # Remove brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', ' ', text)  # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words with numbers
    return text

df['text'] = df['title'] + " " + df['text']
df['text'] = df['text'].apply(clean_text)

# Splitting the data
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# PassiveAggressiveClassifier
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Live Prediction Function
def predict_news(news):
    cleaned = clean_text(news)
    vectorized = tfidf_vectorizer.transform([cleaned])
    pred = model.predict(vectorized)
    return "Real News 🟢" if pred[0] == 1 else "Fake News 🔴"

# Try it on a custom input
print("\n--- Try it on a sample news ---")
sample_news = input("Enter a news headline or article: ")
print("Prediction:", predict_news(sample_news))
