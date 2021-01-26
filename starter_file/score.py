import json
import numpy as np
import pandas as pd
import os
import joblib
#import azureml
import time
#from werkzeug.utils import cached_property
from azureml.core.model import Model
import logging
import xgboost
from xgboost import XGBClassifier



def init():
    global model
    global inputs_dc
    global prediction_dc
    try:       
	#Print statement for appinsights custom traces:
    	print ("model initialized" + time.strftime("%H:%M:%S"))
        logging.basicConfig(level=logging.DEBUG)
        
        #loading model from registry
        print(Model.get_model_path(model_name='model'))
        model_path = Model.get_model_path('model')
        model = joblib.load(model_path)
    except Exception as e:
        print(e)
    
def run(data):
    
    try:
        #convert to json and load
        #data = data.to_json()
        #data = data.loads(data)
        
        # make prediction    
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        print("ERROR: " + error + " " + time.strftime("%H:%M:%S"))
        return error