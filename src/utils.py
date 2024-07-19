import os
import sys
import pickle
# import dill

import numpy as np
import pandas as pd

from src.exception import CustomException

from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            # dill.dump(obj,file_obj)
            pickle.dump(obj,file_obj)


    except Exception as e:
       raise CustomException(e,sys)
        

def evoluate_model(x_train,y_train,x_test,y_test,models):
    try:
        reports = {}

        for i in range(len(list(models.values()))):
            model = list(models.values())[i]
            model.fit(x_train,y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            reports[list(models.keys())[i]] = test_model_score
        return reports
    except Exception as e:
        raise CustomException(e,sys)