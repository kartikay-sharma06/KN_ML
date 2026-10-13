
import sys

from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:

    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            # Step 1: Data Ingestion
            data_ingestion = DataIngestion()

            train_path, test_path = (
                data_ingestion.initiate_data_ingestion()
            )

            # Step 2: Data Transformation
            data_transformation = DataTransformation()

            train_array, test_array, _ = (
                data_transformation.initiate_data_transformation(
                    train_path,
                    test_path
                )
            )

            # Step 3: Model Training
            model_trainer = ModelTrainer()

            model_trainer.initiate_model_training(
                train_array,
                test_array
            )

        except Exception as e:
            raise CustomException(e, sys.exc_info())
