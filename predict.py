import joblib

def predict_news(text):
    model = joblib.load('app/model.pkl')
    vectorizer = joblib.load('app/vectorizer.pkl')
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)
    return "Real" if prediction[0] == 1 else "Fake"