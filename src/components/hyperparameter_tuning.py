
import sys

from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def get_hyperparameter_grids():
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

    return params


def tune_models(models, X_train, y_train):
    try:
        params = get_hyperparameter_grids()

        tuned_models = {}
        model_report = {}

        logging.info("Starting hyperparameter tuning")

        for model_name, model in models.items():

            logging.info(
                f"Hyperparameter tuning started for {model_name}"
            )

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

            best_model = grid_search.best_estimator_

            tuned_models[model_name] = best_model
            model_report[model_name] = grid_search.best_score_

            logging.info(
                f"{model_name} best parameters: "
                f"{grid_search.best_params_}"
            )

            logging.info(
                f"{model_name} best CV R2 score: "
                f"{grid_search.best_score_:.4f}"
            )

        logging.info("Hyperparameter tuning completed")

        return tuned_models, model_report

    except Exception as e:
        raise CustomException(e, sys.exc_info())
