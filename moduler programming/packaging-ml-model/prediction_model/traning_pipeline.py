import pandas as pd
import numpy as np
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset,save_pipeline
import prediction_model.processing.preprocessing as pp 
import prediction_model.pipeline as pipe

def perform_training():
    train_data=load_dataset(config.TRAIN_FILE)
    train_y=train_data[config.TRAGET].map({'N':0,'Y':1})
    pipe.classification_pipeline.fit(train_data[config.FEATURE],train_y)
    save_pipeline(pipeline_to_save=pipe.classification_pipeline)


# to run the python script when ever we call the pipeline 
# we can define the fuction like this
if __name__ =='__main__':
    perform_training()