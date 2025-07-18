import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from scr.exception import CustomException
from scr.logger import logging
from scr.utils import save_object
from scr.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'XGBoost': XGBRegressor(),
                'KNeighbors': KNeighborsRegressor(),
                'CatBoost': CatBoostRegressor(verbose=False),
                'AdaBoost': AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
            },
            "Random Forest": {
                    'n_estimators': [8, 16, 32, 64],
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Gradient Boosting": {
                    'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
                    'learning_rate': [0.01, 0.1, 0.2, 0.3],
                    'n_estimators': [50, 100, 200]
                },
                "XGBoost": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7]
                },
                "KNeighbors": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                },
                "CatBoost": {
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.1],
                    'iterations': [100, 200]
                }
            }

            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test=y_test ,
                                                models=models, params=params)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)
            
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_score
        except Exception as e:
            logging.error("Error in model training: %s", str(e))   
            raise CustomException(e, sys) from e
