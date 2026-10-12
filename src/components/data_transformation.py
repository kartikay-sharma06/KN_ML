import sys
from dataclasses  import dataclass
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder,StandardScaler 

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features=[ 'reading_score', 'writing_score']
            categorical_features=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            numerical_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="median")),
                    ('scaler',StandardScaler())
                ]
            )
            categorical_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ("onehotencodeing",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Categorical columns are created:{categorical_features}")
            logging.info(f"Numerical columns are created:{numerical_features}")

            preprocessor=ColumnTransformer(
                [
                    ("numerical_pipeline",numerical_pipeline,numerical_features),
                    ("categorical_pipeline",categorical_pipeline,categorical_features)
                ]
                )

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            return preprocessor

            
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_trans(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj=self.get_data_transformer_object()
            target_column_name='math_score'
            numerical_features=[ 'reading_score', 'writing_score']
            categorical_features=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            input_feature_train_df = train_df.drop(columns=[target_column_name])

            input_feature_test_df = test_df.drop(columns=[target_column_name])

        except Exception as e:
            raise CustomException(e, sys)