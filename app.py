import pandas as pd
import numpy as np
import os
import pickle
import re
from flask import Flask, render_template, request, redirect, session
import mysql.connector
import os
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('all')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, Embedding, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model


app = Flask(__name__)


model = load_model('gru_model.h5')

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

porter_stem = PorterStemmer()




def clean_text(text):
  res= text.lower()
  res= re.sub(r'/d+', '', text)
  res= re.sub(r'/W', '', text)
  res= re.sub(r'/w/s', '', text)
  res= re.sub(r'^a-zA-Z0-9', '', text)
  res= re.sub(r'\s+', ' ', text).strip()
  res= [porter_stem.stem(word) for word in res if not word in stopwords.words('english')]
  res = ''.join(res)

  return res


def remove_tags(text, remove_extra_spaces=True):
  res = re.sub(r'<[^>]+>','', text)

  if remove_extra_spaces:
    res= re.sub(r'\s+', '', text)

  return res


def predictive_system(review):
  review = clean_text(review)
  review= remove_tags(review)
  sequences = tokenizer.texts_to_sequences([review])
  padded_sequences = pad_sequences(sequences, maxlen=200)
  prediction = model.predict(padded_sequences)
  sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
  return sentiment, float(prediction)



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    return render_template('res.html')



@app.route('/predicts', methods=['GET','POST'])
def predicts():
    result = None
    error = None

    if request.method == 'POST':
        texts = request.form.get('texts')

        if len(texts) < 2:
            return render_template('res.html', error="Text too short")

        if texts:
            result= predictive_system(texts)    
            print(f"Prediction result:")
       
         
    return render_template('res.html', result=result,error=error)



if __name__ == "__main__":
    app.run(debug=True)


