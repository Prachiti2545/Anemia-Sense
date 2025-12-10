import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

#Load your trained model
with open('model.pkl','rb') as f:
    model = pickle.load(f)

@app.route('/', methods = ["GET"])# route to display the home page
def index():
    return render_template('index.html')

@app.route('/predict', methods = ["POST"])
def predict():

    Gender = float(request.form["Gender"])
    Hemoglobin = float(request.form["Hemoglobin"])
    MCH = float(request.form["MCH"])
    MCHC = float(request.form["MCHC"])
    MCV = float(request.form["MCV"])

    features_values = np.array([[Gender,Hemoglobin,MCH,MCHC,MCV]]) # reshape to 2D array

    df = pd.DataFrame(features_values, columns=["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"])
    print(df)

    prediction = model.predict(df)
    print(prediction[0])
    result = prediction[0]

    if prediction[0] == 0:
        result = "You don't have any Anemic Disease"
    elif prediction[0] == 1:
        result = "You have anemic disease"

    return render_template(
        "predict.html",
        Gender=Gender,
        Hemoglobin=Hemoglobin,
        MCH=MCH,
        MCHC=MCHC,
        MCV=MCV,
        result = result
         )


if __name__ == "__main__":
    app.run(debug= True , port = 5000)