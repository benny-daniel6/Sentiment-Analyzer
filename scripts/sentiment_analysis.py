# sentiment_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def load_data():
    """Load the IMDb dataset."""
    df = pd.read_csv('./data/IMDB_Dataset.csv')
    return df

def clean_text(text):
    """Clean the text data."""
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase & remove spaces
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

def preprocess_data(df):
    """Preprocess the data for sentiment analysis."""
    df['cleaned_review'] = df['review'].apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_review']).toarray()
    y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    return X, y, vectorizer

def train_model(X_train, y_train):
    """Train a Logistic Regression model."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def save_model(model, vectorizer):
    """Save the model and vectorizer."""
    joblib.dump(model, 'models/sentiment_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

def main():
    """Main function to run the sentiment analysis pipeline."""
    # Load data
    df = load_data()
    
    # Preprocess data
    X, y, vectorizer = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, vectorizer)

if __name__ == "__main__":
    # Download stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    # Run the pipeline
    main()