
import os
import sys
from dataclasses import dataclass

import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score as r2_score_metric

from xgboost import XGBRegressor


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

            models = {
                "Linear Regression": LinearRegression(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # Initial model evaluation
            model_report: dict = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            logging.info(
                f"Initial model evaluation completed: {model_report}"
            )

            # Hyperparameter grids for each model
            params = {
                "Linear Regression": {},

                "Gradient Boosting Regressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [2, 3],
                    "subsample": [0.8, 1.0]
                },

                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2]
                },

                "Decision Tree": {
                    "criterion": ["squared_error", "absolute_error"],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                },

                "Random Forest Regressor": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                },

                "XGBRegressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5],
                    "subsample": [0.8, 1.0]
                },

                "CatBoosting Regressor": {
                    "iterations": [100, 200],
                    "learning_rate": [0.03, 0.1],
                    "depth": [4, 6]
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1, 1.0],
                    "loss": ["linear", "square"]
                }
            }

            logging.info("Starting hyperparameter tuning")

            tuned_models = {}
            model_report = {}

            for model_name, model in models.items():

                logging.info(
                    f"Hyperparameter tuning started for {model_name}"
                )

                # GridSearchCV with cross-validation
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=params[model_name],
                    scoring="r2",
                    cv=3,
                    n_jobs=-1,
                    verbose=0,
                    error_score="raise"
                )

                grid_search.fit(X_train, y_train)

                # Best estimator after hyperparameter tuning
                best_model = grid_search.best_estimator_

                # Evaluate tuned model on test data
                predicted = best_model.predict(X_test)

                score = r2_score_metric(y_test, predicted)

                tuned_models[model_name] = best_model
                model_report[model_name] = score

                logging.info(
                    f"{model_name} best parameters: "
                    f"{grid_search.best_params_}"
                )

                logging.info(
                    f"{model_name} best CV R2 score: "
                    f"{grid_search.best_score_}"
                )

                logging.info(
                    f"{model_name} test R2 score: {score}"
                )

            # Select the best model using test R2 score
            best_model_score = max(model_report.values())

            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model = tuned_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(
                    "No best model found with an R2 score "
                    "of at least 0.6",
                    sys
                )

            logging.info(
                f"Best found model on testing dataset is "
                f"{best_model_name} with R2 score: "
                f"{best_model_score}"
            )

            # Create artifacts directory if it does not exist
            os.makedirs(
                os.path.dirname(
                    self.model_trainer_config.trained_model_file_path
                ),
                exist_ok=True
            )

            # Save the best tuned model
            save_object(
                file_path=(
                    self.model_trainer_config.trained_model_file_path
                ),
                obj=best_model
            )

            logging.info(
                f"Best tuned model saved at "
                f"{self.model_trainer_config.trained_model_file_path}"
            )

            # Final R2 score
            predicted = best_model.predict(X_test)

            r2_score = r2_score_metric(y_test, predicted)

            logging.info(f"Final R2 score: {r2_score}")

            return r2_score

        except Exception as e:
            raise CustomException(e, sys.exc_info())
