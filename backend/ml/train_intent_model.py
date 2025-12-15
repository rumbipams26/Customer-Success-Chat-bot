import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

def train_model(intents_file="intents.csv"):
    df = pd.read_csv(intents_file)
    X = df["text"]
    y = df["intent"]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=500)
    model.fit(X_vec, y)

    with open("intent_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()