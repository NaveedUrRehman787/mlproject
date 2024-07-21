import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
from src.utils import save_object,evoluate_model

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

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split data into train and test set")
            X_train,y_train,X_test,y_test = (train_array[:,:-1],
                                             train_array[:,-1],
                                             test_array[:,:-1],
                                             test_array[:,-1])
            
            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "KNeighbors Regressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "AdaBoost Regressor":AdaBoostRegressor()
            }

            params = {
                'Decision Tree':{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson']
                },
                'Random Forest':{
                    'n_estimators':[8,16,32,64,128,256]
                },
                'Gradient Boosting':{
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'subsample':[0.6,0.7,0.075,0.8,0.85,0.9],
                    'n_estimators':[8,16,32,64,128,256],
                },
                'Linear Regression':{},

                'KNeighbors Regressor':{
                    'n_neighbors':[5,7,9,11]                   
                },
                
                'XGBRegressor':{
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                'AdaBoost Regressor':{
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'n_estimators':[8,16,32,64,128,256]
                }

            }

            model_report:dict=evoluate_model(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test,
                                             models=models,
                                             param=params)

            # to get best model
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("There is no best model")
            logging.info("Best model found")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_sqr = r2_score(y_test,predicted)
            return r2_sqr,best_model_name
        

        except Exception as e:
            raise CustomException(e,sys)