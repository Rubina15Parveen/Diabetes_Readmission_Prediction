import os 
import sys 
import pandas as pd 
import numpy as np 
from dataclasses import dataclass
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve
import xgboost as xgb
from xgboost import XGBClassifier
from scipy import sparse
from src.exception import CustomException
from src.logger import logging
from src.utilis import save_object, evaluate_model



@dataclass
class ModelTrainerConfig: 
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting the training and test input data")
            def split_X_y(arr):
                X = arr[:, :-1]
                y_col = arr[:, -1]          # (n, 1) sparse/dense
                if sparse.issparse(y_col):
                    y = y_col.toarray().ravel()   # -> (n,)
                else:
                    y = np.asarray(y_col).ravel() # -> (n,)
                return X, y
            X_train, y_train = split_X_y(train_array)
            X_test,  y_test  = split_X_y(test_array)
            

            models = {
                "Logistic Regression" : LogisticRegression(
                    max_iter=2000, 
                    class_weight='balanced'
                ), 
                "Random Forest Classifier" : RandomForestClassifier(), 
                "XGBoost Classifier" : XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="auc",
                    tree_method="hist",
                    random_state=42,
                    n_jobs=-1
                ), 
                #"CatBoost Classifier" : CatBoostClassifier()
            }

            model_report: dict=evaluate_model(X_train=X_train, y_train=y_train, X_test = X_test, y_test = y_test, 
                                              models=models)
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model")
            
            logging.info("Best model found on both train and test datasets.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, 
                obj=best_model
            )
            probabilities = best_model.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, probabilities)
            probabilities_threshold = (probabilities >= 0.5).astype(int)
            classify_report = classification_report(y_test, probabilities_threshold)
            logging.info(f"Test AUC: {auc:.4f}")
            print(best_model_name)
            return classify_report

        except Exception as e:
            raise CustomException(e, sys)
            