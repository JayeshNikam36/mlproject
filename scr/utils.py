import os
import sys
import numpy as np
import pandas as pd
import dill
from scr.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(obj, file_path):
    """
    Save an object to a file using pickle.
    
    Parameters:
    obj: The object to save.
    file_path: The path where the object will be saved.
    """
    import pickle
    try:
       dir_path = os.path.dirname(file_path)
       os.makedirs(dir_path, exist_ok=True)
       with open(file_path, 'wb') as file:
          dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys) from e
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = models[model_name]
            model_params = params.get(model_name)  # returns None if not found

            if model_params:  # only apply GridSearch if params exist
                gs = GridSearchCV(model, model_params, cv=3)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys) from e
            