import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def load_data():
    fake = pd.read_csv('dataset/fake.csv')
    real = pd.read_csv('dataset/real.csv')
    fake['label'] = 0
    real['label'] = 1
    data = pd.concat([fake, real]).sample(frac=1).reset_index(drop=True)
    return data

def preprocess_and_train(data):
    x_train, x_test, y_train, y_test = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression()
    model.fit(x_train_vec, y_train)
    
    y_pred = model.predict(x_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc*100:.2f}%")

    joblib.dump(model, 'app/model.pkl')
    joblib.dump(vectorizer, 'app/vectorizer.pkl')

    return model, vectorizer

if __name__ == "__main__":
    data = load_data()
    model, vectorizer = preprocess_and_train(data)