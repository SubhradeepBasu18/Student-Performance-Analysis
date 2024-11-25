import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1], # All columns except the last one
                train_arr[:,-1], # Last column
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            param_grid = {
                    "Random Forest": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "bootstrap": [True, False],
                    },
                    "Decision Tree": {
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                    },
                    "Gradient Boosting": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 0.2, 0.3],
                        "max_depth": [3, 5, 10],
                        "subsample": [0.6, 0.8, 1.0],
                    },
                    "Linear Regression": {
                        # No significant hyperparameters for tuning
                    },
                    "XGBRegressor": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 0.2, 0.3],
                        "max_depth": [3, 5, 10],
                        "subsample": [0.6, 0.8, 1.0],
                        "colsample_bytree": [0.6, 0.8, 1.0],
                        "gamma": [0, 1, 5],
                        "reg_alpha": [0, 0.01, 0.1],
                        "reg_lambda": [1, 1.5, 2],
                    },
                    "CatBoosting Regressor": {
                        "iterations": [100, 200, 300],
                        "learning_rate": [0.01, 0.1, 0.2],
                        "depth": [4, 6, 8, 10],
                        "l2_leaf_reg": [1, 3, 5, 7],
                    },
                    "AdaBoost Regressor": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 0.2, 1],
                        "loss": ["linear", "square", "exponential"],
                    },
            }


            model_report = evaluate_models(X_train,X_test,y_train,y_test,models,params=param_grid)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found",sys)
            
            logging.info(f"Best found model: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            best_r2_score = r2_score(y_test,predicted)
            return best_r2_score

        except Exception as e:
            raise CustomException(e,sys)