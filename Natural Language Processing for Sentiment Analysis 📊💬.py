import nltk
from flask import Flask, request, jsonify, render_template
from transformers import pipeline, BertTokenizer, TFBertForSequenceClassification
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
nltk.download('stopwords')

app = Flask(__name__)

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = TFBertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    user_input = request.form['text']
    preprocessed_text = preprocess_text(user_input)
    sentiment = classifier(preprocessed_text)
    return jsonify(sentiment)

if __name__ == '__main__':
    app.run(debug=True)
