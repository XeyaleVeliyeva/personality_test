from flask import Flask,render_template,request
import joblib
import numpy as np
import pandas as pd
app = Flask(__name__)
model = joblib.load(open("model_7features.joblib","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    f1 = float(request.form['Time_spent_Alone'])
    f2 = float(request.form['Stage_fear'])
    f3 = float(request.form['Social_event_attendance'])
    f4 = float(request.form['Going_outside'])
    f5 = float(request.form['Drained_after_socializing'])
    f6 = float(request.form['Friends_circle_size'])
    f7 = float(request.form['Post_frequency'])

    input_dict = {
    'Time_spent_Alone': [f1],
    'Stage_fear': [f2],
    'Social_event_attendance': [f3],
    'Going_outside': [f4],
    'Drained_after_socializing': [f5],
    'Friends_circle_size': [f6],
    'Post_frequency': [f7]
}

    input_df = pd.DataFrame(input_dict)

    prediction = model.predict(input_df)


    return render_template("index.html",prediction_text = f"Output: {prediction[0]}")

if __name__ == '__main__':
    app.run(debug = True)


