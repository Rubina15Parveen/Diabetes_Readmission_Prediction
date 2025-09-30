import os 
import sys 
import numpy as np 
import pandas as pd 
from src.exception import CustomException
import dill 
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from scipy import sparse


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try: 
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            y_train_pred_proba = model.predict_proba(X_train)[:,1]
            y_test_pred_proba  = model.predict_proba(X_test)[:,1]  

            train_model_score = roc_auc_score(y_train, y_train_pred_proba)
            test_model_score = roc_auc_score(y_test, y_test_pred_proba)
            report[list(models.keys())[i]] = test_model_score
        return report

    
    except Exception as e:
        raise CustomException(e, sys)
            