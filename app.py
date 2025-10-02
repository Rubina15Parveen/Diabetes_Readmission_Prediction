from flask import Flask, request, render_template 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application= Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
                race = request.form.get('race'), 
                gender = request.form.get('gender'), 
                age = request.form.get('age'), 
                time_in_hospital = request.form.get('time_in_hospital'),
                num_lab_procedures = request.form.get('num_lab_procedures'),
                num_procedures = request.form.get('num_procedures'),
                num_medications = request.form.get('num_medications'),
                number_outpatient = request.form.get('number_outpatient'),
                number_emergency = request.form.get('number_emergency'),
                number_inpatient = request.form.get('number_inpatient'),
                number_diagnoses = request.form.get('number_diagnoses'), 
                metformin = request.form.get('metformin'),
                change = request.form.get('change'),
                diabetesMed = request.form.get('diabetesMed'), 
                medical_specialty = request.form.get('medical_specialty'),
                diag_1 = request.form.get('diag_1'), 
                diag_2 = request.form.get('diag_2'),  
                diag_3 = request.form.get('diag_3'),  
                discharge_disposition_id = request.form.get('discharge_disposition_id'), 
                admission_source_id = request.form.get('admission_source_id'), 
                admission_type_id = request.form.get('admission_type_id'), 
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)