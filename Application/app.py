import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from joblib import load

app = Flask(__name__)
model = pickle.load(open('randomf.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction",methods=['POST'])
def predict():
    x_test = [[x for x in request.form.values()]]
    sc = load('scalar_1.save') 
    prediction = model.predict(sc.transform(x_test))
    output=prediction[0]
    if output==0:
        pred = "Transaction Detected"
        return render_template('index.html', prediction_text='Genuine {}'.format(pred))
    else:
        pred = "Transaction Detected"
        return render_template('index.html', prediction_text='Fraudulent {}'.format(pred))

if __name__ == "__main__":
    app.run(debug=True)