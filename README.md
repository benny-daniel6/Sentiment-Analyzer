# Sentiment Analysis on Movie Reviews

This project analyzes the sentiment of movie reviews (positive/negative) using the **IMDb Movie Reviews dataset**. It uses a **Logistic Regression model** to classify reviews and achieves high accuracy in predicting sentiment.

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

---

## Overview

Sentiment analysis is the process of determining whether a piece of text expresses positive or negative sentiment. In this project, we:
- Clean and preprocess the text data.
- Convert text into numerical features using **TF-IDF**.
- Train a **Logistic Regression model** to classify reviews as positive or negative.
- Evaluate the model's performance using accuracy, precision, recall, and F1-score.

---

## Dataset

The dataset used is the [IMDb Movie Reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download), which contains:
- **50,000 movie reviews**.
- **Labels**: `positive` or `negative`.

The dataset is balanced, with an equal number of positive and negative reviews.

---

## Project Structure

The project is organized as follows:
sentiment_analysis/
│
├── data/
│ └── IMDB Dataset.csv # Raw dataset
├── models/
│ └── sentiment_model.pkl # Trained Logistic Regression model
│ └── tfidf_vectorizer.pkl # TF-IDF vectorizer
├── scripts/
│ └── sentiment_analysis.py # Python script for the project
├── images/ # Visualizations (optional)
│ └── sentiment_distribution.png
├── README.md # Project documentation
└── requirements.txt # List of dependencies
---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/benny-daniel6/Sentiment_Analyzer.git
   cd sentiment_analysis
2. Install the required dependencies:
   pip install -r requirements.txt
3. Download the dataset:
   Download the IMDb Movie Reviews dataset.
   Place the IMDB Dataset.csv file in the data/ folder.
## Usage
Running the Script

To train the model and evaluate its performance, run the Python script:
bash:
    python scripts/sentiment_analysis.py
What the Script Does:
1. Loads the dataset from data/IMDB Dataset.csv.
2. Preprocesses the text data:
Removes HTML tags, special characters, and stopwords.
Converts text to lowercase.
3. Converts text to numerical features using TF-IDF.
4. Trains a Logistic Regression model on the training set.
5. Evaluates the model on the test set and prints:
Accuracy
Classification report (precision, recall, F1-score)
Confusion matrix
6. Saves the trained model and vectorizer to the models/ folder.
## Results
Model Performance
Accuracy: The model achieves an accuracy of ~89% on the test set.

Classification Report:
            precision    recall  f1-score   support

         0       0.89      0.89      0.89      5000
         1       0.89      0.89      0.89      5000

  accuracy                           0.89     10000
 macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000

Confusion Matrix:
[[4455  545]
 [ 543 4457]]

 ## Contributing
Contributions are welcome! If you'd like to contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/YourFeature).
3. Commit your changes (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature/YourFeature).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE] file for details.
