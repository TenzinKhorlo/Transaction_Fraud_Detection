from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('logistic_regression_model.sav', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        step = int(request.form['step'])
        amount=float(request.form['amount'])
        oldbalanceOrig=float(request.form['oldbalanceOrig'])
        newbalanceOrig=float(request.form['newbalanceOrig'])
        oldbalanceDest=float(request.form['oldbalanceDest'])
        newbalanceDest=float(request.form['newbalanceDest'])
        
        output=model.predict([[step,amount,oldbalanceOrig,newbalanceOrig,oldbalanceDest,newbalanceDest]]);
        if output==0:
            return render_template('index.html',prediction_text="No Fradulent Transaction Detected")
        else:
            return render_template('index.html',prediction_text="Fradulent Transaction Detected")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)