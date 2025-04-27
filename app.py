from flask import Flask, request, render_template_string
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

html_template = '''
<!doctype html>
<title>Fake News Detection</title>
<h2>Enter News Article:</h2>
<form method="post">
  <textarea name="news" rows="10" cols="50"></textarea><br><br>
  <input type="submit" value="Predict">
</form>
{% if prediction %}
  <h3>Prediction: {{ prediction }}</h3>
{% endif %}
'''

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        news = request.form['news']
        transformed = vectorizer.transform([news])
        pred = model.predict(transformed)
        prediction = "Real" if pred[0] == 1 else "Fake"
    return render_template_string(html_template, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)