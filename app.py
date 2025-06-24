from flask import Flask, render_template, request
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

df = pd.read_csv('IMDB Dataset.csv')

le = LabelEncoder()
df['sentiment'] = le.fit_transform(df['sentiment'])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove digits
    text = text.strip()
    return text

df['review'] = df['review'].apply(clean_text)

tfidf = TfidfVectorizer(max_features=5000)
X_vec = tfidf.fit_transform(df['review'])
y = df['sentiment']

lr_model = LogisticRegression()
lr_model.fit(X_vec, y)

joblib.dump(lr_model, 'logistic_regression_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    cleaned_review = clean_text(review)
    review_vec = tfidf.transform([cleaned_review])
    prediction = lr_model.predict(review_vec)[0]
    sentiment = le.inverse_transform([prediction])[0]
    return render_template('index.html', prediction=sentiment, review=review)

if __name__ == '__main__':
    app.run(debug=True)