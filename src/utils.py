import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import dill

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

        
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            r2 = r2_score(y_test, y_pred)
            report[model_name] = r2
        return report
    except Exception as e:
        raise CustomException(e, sys)