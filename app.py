import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    return text

# Load dataset
dataset_path = "https://gist.githubusercontent.com/ArbilShofiyurrahman/44ba802820f22b10103ce3fb08b0bf7a/raw/ab90c71f3ddbaa464ef116be55089d0fc859fce6/BBC%2520News%2520Train.csv"
dataset = pd.read_csv(dataset_path)

# Preprocess dataset
dataset['Text'] = dataset['Text'].apply(preprocess_text)

# Split dataset into X and y
x_data = dataset['Text']
y_data = dataset['CategoryId']

# Vectorize text data
cv = CountVectorizer(max_features=5000)
x_data_vectorized = cv.fit_transform(x_data).toarray()

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_data_vectorized, y_data, test_size=0.3, random_state=0, shuffle=True)

# Load trained model
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)

# Streamlit app
def main():
    st.title("BBC News Classifier")
    st.subheader("Enter your news article below:")

    # Text input for news article
    article_text = st.text_area("Input Text", "")

    if st.button("Classify"):
        # Preprocess text
        processed_text = preprocess_text(article_text)
        # Vectorize text
        text_vectorized = cv.transform([processed_text])
        # Predict category
        prediction = classifier.predict(text_vectorized)
        # Map prediction to category
        categories = {0: "Business News", 1: "Tech News", 2: "Politics News", 3: "Sports News", 4: "Entertainment News"}
        predicted_category = categories.get(prediction[0], "Unknown")
        # Display result
        st.write("Predicted Category:", predicted_category)

if __name__ == "__main__":
    main()
