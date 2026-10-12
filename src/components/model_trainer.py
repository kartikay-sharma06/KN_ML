
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score as r2_score_metric

from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from src.components.hyperparameter_tuning import tune_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        'artifacts', 'model.pkl'
    )


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info(
                "Splitting dependent and independent variable "
                "from train and test data"
            )

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define models
            models = {
                "Linear Regression": LinearRegression(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(
                    verbose=False
                ),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # Initial model evaluation
            logging.info("Starting initial model evaluation")

            initial_model_report = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            logging.info(
                f"Initial model evaluation completed: "
                f"{initial_model_report}"
            )

            # Hyperparameter tuning
            tuned_models, cv_model_report = tune_models(
                models=models,
                X_train=X_train,
                y_train=y_train
            )

            # Evaluate tuned models on test data
            model_report = {}

            logging.info(
                "Evaluating tuned models on test dataset"
            )

            for model_name, model in tuned_models.items():
                predicted = model.predict(X_test)

                score = r2_score_metric(y_test, predicted)
                model_report[model_name] = score

                logging.info(
                    f"{model_name} test R2 score: {score:.4f}"
                )

            # Select the best model using test R2 score
            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model = tuned_models[best_model_name]
            best_model_score = model_report[best_model_name]

            # Best model details
            logging.info("=" * 60)
            logging.info("BEST MODEL SELECTED")
            logging.info(f"Best Model Name: {best_model_name}")
            logging.info(
                f"Best CV R2 Score: "
                f"{cv_model_report[best_model_name]:.4f}"
            )
            logging.info(
                f"Best Test R2 Score: {best_model_score:.4f}"
            )
            logging.info("=" * 60)

            if best_model_score < 0.6:
                raise CustomException(
                    "No best model found with an R2 score "
                    "of at least 0.6",
                    sys
                )

            # Create artifacts directory
            os.makedirs(
                os.path.dirname(
                    self.model_trainer_config.trained_model_file_path
                ),
                exist_ok=True
            )

            # Save best model
            save_object(
                file_path=(
                    self.model_trainer_config.trained_model_file_path
                ),
                obj=best_model
            )

            logging.info(
                f"Best Model Name: {best_model_name}"
            )
            logging.info(
                f"Best tuned model saved at "
                f"{self.model_trainer_config.trained_model_file_path}"
            )

            # Final R2 score
            predicted = best_model.predict(X_test)
            r2_score = r2_score_metric(y_test, predicted)

            logging.info(
                f"Final R2 score: {r2_score:.4f}"
            )

            return r2_score

        except Exception as e:
            raise CustomException(e, sys.exc_info())
