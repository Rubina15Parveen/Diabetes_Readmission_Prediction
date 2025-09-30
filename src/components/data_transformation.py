import sys 
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from pandas.api.types import is_numeric_dtype, is_categorical_dtype 
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve
import xgboost as xgb
from xgboost import XGBClassifier
from scipy import sparse
from src.exception import CustomException
from src.logger import logging
from src.utilis import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function handles the data transformation
        """
        try:
            age_order = [['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)',
                        '[60-70)', '[70-80)', '[80-90)', '[90-100)']]
            
            numerical_col = ['time_in_hospital',
                            'num_lab_procedures',
                            'num_procedures',
                            'num_medications',
                            'number_outpatient',
                            'number_emergency',
                            'number_inpatient',
                            'number_diagnoses']
            
            categorical_col = ['race','gender',
                                'metformin',
                                'change',
                                'diabetesMed']
            
            high_cardinal_cat_col = ['medical_specialty','diag_1', 'diag_2', 'diag_3',]

            cat_ids = ['discharge_disposition_id', 'admission_source_id', 'admission_type_id']

            ordinal_col = ['age']
            


            numeric_transformer = Pipeline(steps=[
                                ("imputer", SimpleImputer(strategy='median')), 
                                ("scale", StandardScaler())    
                            ])
            
            categorical_transformer = Pipeline(steps=[
                                ("imputer", SimpleImputer(strategy='most_frequent')), 
                                ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=True))
                            ])
            high_cat_transformer = Pipeline(steps=[
                                ("imputer", SimpleImputer(strategy='most_frequent')), 
                                ("encoder", OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=100, sparse_output=True))
                            ])
            ordinal_transformer = Pipeline(steps=[
                                ('imputer', SimpleImputer(strategy='most_frequent')),
                                ('ord', OrdinalEncoder(categories=age_order))
                            ])

            cat_int_transformer = Pipeline(steps=[
                                ('imputer', SimpleImputer(strategy='most_frequent')),
                                ('to_str', FunctionTransformer(lambda s: s.astype(str))),
                                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
                            ])
            logging.info("Numerical and Categorical Pipelines are made")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical", numeric_transformer, numerical_col), 
                    ("categorical", categorical_transformer, categorical_col), 
                    ("high cardinality", high_cat_transformer, high_cardinal_cat_col), 
                    ("ordinal", ordinal_transformer, ordinal_col), 
                    ("categorical int", cat_int_transformer, cat_ids)

                ], remainder='drop'
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading the train and test data completed")
            logging.info("Obtaining preprocessing object")

            train_df['readmission_binary'] = train_df['readmitted'].apply(lambda x: 0 if x =='NO' else 1 )
            test_df['readmission_binary'] = test_df['readmitted'].apply(lambda x: 0 if x =='NO' else 1 )

            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = 'readmission_binary'

            input_feature_train_df = train_df.drop(columns=['readmitted', 'repaglinide', 'nateglinide', 'chlorpropamide',
                                                    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 
                                                    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                                                    'metformin-pioglitazone','max_glu_serum', 'weight','A1Cresult', 
                                                    'encounter_id','patient_nbr', 'payer_code', target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=['readmitted', 'repaglinide', 'nateglinide', 'chlorpropamide',
                                                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
                                                'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
                                                'tolazamide', 'examide', 'citoglipton','insulin',
                                                'glyburide-metformin', 'glipizide-metformin',
                                                'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                                                'metformin-pioglitazone','max_glu_serum', 'weight','A1Cresult',
                                                'encounter_id','patient_nbr', 'payer_code', target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            y_train = np.asarray(target_feature_train_df).reshape(-1, 1)
            y_test  = np.asarray(target_feature_test_df).reshape(-1, 1)
            
            if sparse.issparse(input_feature_train_arr):
                train_arr = sparse.hstack([input_feature_train_arr, y_train], format='csr')
                test_arr  = sparse.hstack([input_feature_test_arr,  y_test],  format='csr')
            else:
                train_arr = np.hstack([input_feature_train_arr, y_train])
                test_arr  = np.hstack([input_feature_test_arr,  y_test])

            #train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            #test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved processing objects.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path, 
            )
        
        
        
        
        
        except Exception as e:
            raise CustomException(e, sys) 