from prediction_model.config import config
import pandas as pd
import numpy as np
import joblib
from prediction_model.processing.data_handling import load_pipeline,load_dataset

classification_pipeline=load_pipeline(pipeline_to_load=config.SAVE_MODEL_NAME)

# def genrate_prediction(data_input):
#     data =pd.DataFrame(data_input)
#     pred= classification_pipeline.predict(data[config.FEATURE])
#     output=np.where(pred==1,'Y',"N")
#     result={"predictions":output}
#     return result

def genrate_prediction():
    test_data=load_dataset(config.TEST_FILE)
    pred= classification_pipeline.predict(test_data[config.FEATURE])
    output=np.where(pred==1,'Y',"N")
    result={"predictions":output}
    return result

if __name__=='__main__':
    genrate_prediction()