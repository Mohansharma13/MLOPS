import os
import pandas as pd
import joblib
from prediction_model.config import config

# function to load data
def load_dataset(file_name):
    # Use raw string for file path
    filepath = os.path.join(config.DATAPATH, file_name)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    _data = pd.read_csv(filepath)
    return _data

# serilization function

def save_pipeline(pipeline_to_save):
    save_path= os.path.join(config.SAVE_MODEL_PATH,config.SAVE_MODEL_NAME)
    joblib.dump(pipeline_to_save,save_path)
    print("model has been saved ")
    
# deserilization
def load_pipeline(pipeline_to_load):
    save_path= os.path.join(config.SAVE_MODEL_PATH,pipeline_to_load)
    model_loaded=joblib.load(save_path)
    print("model has been loaded ")
    return(model_loaded)
    