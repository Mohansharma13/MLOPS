# we are going to folow pytest module documentaion for this
# https://docs.pytest.org/en/8.2.x/

import pytest 
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
from prediction_model.predict import genrate_prediction


# fixture in pytest
# fixture are the function that run before execution of each test function

def single_prediction():
    test_dataset=load_dataset(config.TEST_FILE)
    single_row=test_dataset[:1]
    result= genrate_prediction(single_row)
    return(result)