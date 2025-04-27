# Fake News Detection Using Machine Learning

## Project Description
This project aims to classify news articles as Real or Fake using Machine Learning techniques such as Natural Language Processing (NLP) and Logistic Regression.

## Feature Scope
- Text Preprocessing: Cleaning and tokenizing the news articles.
- Feature Extraction: TF-IDF Vectorization.
- Model Training: Logistic Regression Classifier.
- Prediction System: Classify new articles.
- Evaluation: Accuracy, Precision, Recall, F1 Score.
- Web Interface: Simple Flask-based web app.

## Setup Instructions
1. Clone the Repository
```bash
git clone <your-repo-link>
```

2. Navigate to Project Directory
```bash
cd Fake-News-Detection
```

3. Create Virtual Environment and Install Dependencies
```bash
python -m venv venv
source venv/bin/activate (Linux/Mac)
venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

4. Run the Jupyter Notebook
```bash
jupyter notebook notebooks/fake_news_detection.ipynb
```

5. (Optional) Run Flask App
```bash
python app/app.py
```

## Dependencies
- pandas
- numpy
- scikit-learn
- nltk
- Flask
- jupyter

## License
MIT License
