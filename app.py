import flask
import numpy as np

from flask import  Flask,render_template,request

import sklearn
from sklearn.linear_model import LinearRegression
import numpy
import pickle

with open("simple_liner_regression_model.pkl","rb") as f:
  model=pickle.load(f)


app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html",prediction_text=None,
        experience=None)

@app.route("/predict" ,methods=['POST'])

def prediction():
    value = float(request.form["experience"])

    value = np.array([[value]])

    sol = model.predict(value)[0]
    return render_template("index.html" , prediction_text=sol)

if __name__=="__main__":
    app.run(debug=True)