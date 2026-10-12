
import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        'artifacts', 'preprocessor.pkl'
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = ['reading_score', 'writing_score']

            categorical_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehotencoding', OneHotEncoder(
                        handle_unknown='ignore'
                    )),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(
                f'Categorical columns are created: {categorical_features}'
            )
            logging.info(
                f'Numerical columns are created: {numerical_features}'
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        'numerical_pipeline',
                        numerical_pipeline,
                        numerical_features
                    ),
                    (
                        'categorical_pipeline',
                        categorical_pipeline,
                        categorical_features
                    )
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys.exc_info())

    def initiate_data_trans(self, train_path, test_path):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            logging.info('Obtaining preprocessing object')

            preprocessor_obj = self.get_data_transformer_object()

            # Target column
            target_column_name = 'math_score'

            # Separate input and target features
            input_feature_train_df = train_df.drop(
                columns=[target_column_name]
            )

            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name]
            )

            target_feature_test_df = test_df[target_column_name]

            logging.info(
                'Applying preprocessing object on training and testing data'
            )

            # Fit and transform training data
            input_feature_train_arr = preprocessor_obj.fit_transform(
                input_feature_train_df
            )

            # Transform testing data using the fitted preprocessor
            input_feature_test_arr = preprocessor_obj.transform(
                input_feature_test_df
            )

            # Combine input features and target values
            train_arr = np.c_[
                input_feature_train_arr,
                np.asarray(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.asarray(target_feature_test_df)
            ]

            logging.info('Saved preprocessing object')

            # Ensure the artifacts directory exists
            os.makedirs(
                os.path.dirname(
                    self.data_transformation_config.preprocessor_obj_file_path
                ),
                exist_ok=True
            )

            # Save the fitted preprocessing object
            save_object(
                file_path=(
                    self.data_transformation_config
                    .preprocessor_obj_file_path
                ),
                obj=preprocessor_obj
            )

            logging.info('Data transformation completed successfully')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys.exc_info())
