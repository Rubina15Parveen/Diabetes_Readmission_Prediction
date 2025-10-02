import sys 
import pandas as pd 
from src.exception import CustomException
from src.utilis import load_object 


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path ='artifacts/model.pkl' 
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds 
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, 
                 race: str, 
                 gender: str, 
                 age: str, 
                time_in_hospital: int,
                num_lab_procedures: int,
                num_procedures: int,
                num_medications: int,
                number_outpatient: int,
                number_emergency: int,
                number_inpatient: int,
                number_diagnoses: int, 
                metformin: str,
                change: str,
                diabetesMed:str, 
                medical_specialty: str,
                diag_1:str,
                diag_2: str, 
                diag_3:str, 
                discharge_disposition_id:int, 
                admission_source_id:int, 
                admission_type_id:int
            ): 
        self.race = race
        self.gender = gender 
        self.age = age
        self.time_in_hospital = time_in_hospital
        self.num_lab_procedures = num_lab_procedures
        self.num_procedures = num_procedures
        self.num_medications = num_medications
        self.number_outpatient = number_outpatient
        self.number_emergency = number_emergency
        self.number_inpatient = number_inpatient
        self.number_diagnoses = number_diagnoses
        self.metformin = metformin
        self.change = change
        self.diabetesMed = diabetesMed 
        self.medical_specialty = medical_specialty
        self.diag_1 = diag_1
        self.diag_2 = diag_2
        self.diag_3 = diag_3 
        self.discharge_disposition_id = discharge_disposition_id
        self.admission_source_id = admission_source_id
        self.admission_type_id = admission_type_id

    def get_data_as_dataframe(self): 
        try:
            custom_data_input_dict = {
                "race": [self.race], 
                "gender": [self.gender], 
                "age": [self.age], 
                "time_in_hospital":[self.time_in_hospital],
                "num_lab_procedures": [self.num_lab_procedures],
                "num_procedures": [self.num_procedures],
                "num_medications": [self.num_medications],
                "number_outpatient": [self.number_outpatient],
                "number_emergency": [self.number_emergency],
                "number_inpatient": [self.number_inpatient],
                "number_diagnoses": [self.number_diagnoses], 
                "metformin": [self.metformin],
                "change": [self.change],
                "diabetesMed":[self.diabetesMed], 
                "medical_specialty": [self.medical_specialty],
                "diag_1":[self.diag_1],
                "diag_2": [self.diag_2], 
                "diag_3": [self.diag_3], 
                "discharge_disposition_id": [self.discharge_disposition_id], 
                "admission_source_id": [self.admission_source_id], 
                "admission_type_id":[self.admission_type_id]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
    
    