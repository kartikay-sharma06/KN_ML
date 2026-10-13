from src.components.data_ingestion import data_ingestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer



if __name__=="__main__":


    object=data_ingestion()
    train_data,test_data=object.initiate_data_ingestion()     

    data_transformation=DataTransformation()
    train_array, test_array, preprocessor_path = (
    data_transformation.initiate_data_trans(train_data, test_data)
)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_training(train_array,test_array))