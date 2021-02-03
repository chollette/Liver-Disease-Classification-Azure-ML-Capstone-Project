import os
import json
import time
import joblib
import logging
import xgboost
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from azureml.core.model import Model

def init():
    global model
    global inputs_dc
    global prediction_dc
    try:       
        #Print statement for appinsights custom traces:
        logging.basicConfig(level=logging.DEBUG)
        print ("model initialized" + time.strftime("%H:%M:%S"))
       
        #loading model from registry
        print(Model.get_model_path(model_name='model'))
        model_path = Model.get_model_path('model')
        model = joblib.load(model_path)
    except Exception as e:
        print(e)
    
def run(data):
    try:
        #load as json file
        data = json.loads(data)['data']  
        data = pd.DataFrame(data)
        
        # make prediction    
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        print("ERROR: " + error + " " + time.strftime("%H:%M:%S"))
        return error