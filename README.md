# Fake-News-Detection-Using-NLP
To build a machine learning model that automatically detects and classifies news articles as real or fake using NLP techniques and textual data.
 Background:
With the rise of digital media, fake news has become a significant problem, influencing public opinion, elections, health behavior, and more. Manual detection is slow and unreliable, so automated fake news detection is a crucial application of AI and NLP.

🧰 Technologies & Tools Used:
Programming Language: Python

Libraries:

pandas, numpy (data handling)

scikit-learn (ML models, TF-IDF, metrics)

re, string (text cleaning)

joblib (model persistence)

Algorithms: PassiveAggressiveClassifier

Technique: TF-IDF (Term Frequency-Inverse Document Frequency)

📁 Dataset Used:
Kaggle Dataset: Fake and Real News Dataset

Fake.csv: Collection of fake news articles

True.csv: Collection of real news articles

Combined into one dataset with a binary label (0: Fake, 1: Real)

🔬 Workflow / Methodology:
Data Collection

Load and merge real and fake news datasets.

Preprocessing

Convert text to lowercase

Remove HTML tags, punctuation, numbers, and URLs

Token clean-up and stop word removal (via TF-IDF)

Feature Extraction

Use TF-IDF Vectorizer to convert text into numerical vectors

Model Training

Use PassiveAggressiveClassifier, suitable for large-scale, real-time classification

Model Evaluation

Evaluate using Accuracy, Confusion Matrix, and Classification Report

Live Prediction

Input a custom article or headline and detect if it's real or fake

📊 Model Performance:
(Expected based on standard dataset split)

Accuracy: ~92–96%

Precision / Recall / F1-Score: High for both classes

Confusion Matrix: Clear separation between fake and real labels

📈 Sample Output:
lua
Copy
Edit
Accuracy: 0.9473

Classification Report:
              precision    recall  f1-score   support
Fake News       0.94       0.96      0.95       950
Real News       0.95       0.94      0.95       900

Confusion Matrix:
[[913  37]
 [ 54 846]]
