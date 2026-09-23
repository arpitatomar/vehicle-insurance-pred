import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
application = Flask(__name__)
app=application

#route for home page
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['POST', 'GET'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  #home .html contains the form to take input from user
    else:
        data = CustomData(
            Gender=request.form.get('Gender'),
            Age=int(request.form.get('Age')),
            Driving_License=int(request.form.get('Driving_License')),
            Region_Code=int(request.form.get('Region_Code')),
            Previously_Insured=int(request.form.get('Previously_Insured')),
            Vehicle_Age=request.form.get('Vehicle_Age'),
            Vehicle_Damage=request.form.get('Vehicle_Damage'),
            Annual_Premium=float(request.form.get('Annual_Premium')),
            Policy_Sales_Channel=int(request.form.get('Policy_Sales_Channel')),
            Vintage=int(request.form.get('Vintage'))
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])  #display the result in home.html

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)





