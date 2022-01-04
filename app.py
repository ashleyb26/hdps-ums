import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template


# Load ML model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("Heart Disease Classifier.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [float(i) for i in request.form.values()]

    array_features = [np.array(features)]

    array_features = scaler.transform(array_features)
    prediction = model.predict(array_features)
    output = prediction
    if output == 0:
        return render_template(
            "Heart Disease Classifier.html",
            result_no="Result: Not Affected with Heart Disease",
        )
    else:
        return render_template(
            "Heart Disease Classifier.html",
            result_yes="Result: Affected with Heart Disease",
        )


if __name__ == "__main__":
    app.run(debug=True)
