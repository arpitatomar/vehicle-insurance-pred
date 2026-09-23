import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "CatBoost": CatBoostClassifier(verbose=False),
                "XGBoost": XGBClassifier(
                    eval_metric="logloss"
                ),
                "LightGBM": LGBMClassifier(
        
                )
            }

            params = {

                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ["liblinear", "lbfgs"]
                },

                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },

                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },

                "AdaBoost": {
                    "learning_rate": [0.01, 0.1, 0.5, 1],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },

                "CatBoost": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100]
                },

                "XGBoost": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    "max_depth": [3, 5, 7]
                },

                "LightGBM": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    "num_leaves": [15, 31, 63]
                }
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            ## To get best model score from dict

            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.60:
                raise CustomException("No best model found", sys)

            logging.info(
                f"Best found model: {best_model_name} "
                f"with score: {best_model_score}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            f1 = f1_score(y_test, predicted)
            precision = precision_score(y_test, predicted)
            recall = recall_score(y_test, predicted)

            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"F1 Score: {f1}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")

            return accuracy, f1, precision, recall

        except Exception as e:
            raise CustomException(e, sys)