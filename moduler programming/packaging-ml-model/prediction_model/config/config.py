# this file contaion important path and features that we are going to be using 

import pathlib
import os
import prediction_model  # it contain all the importtant file including config

PACKAGE_ROOT= pathlib.path(prediction_model.__file__) # this will return the path to __init__ file path 
                                                      # i.e path for predicition_model path

# datapath is path of mmodule contaion all the dataset
DATAPATH= os.path.join(PACKAGE_ROOT,"datasets")

# name of train and test files
TRAIN_FILE="train.csv"
TEST_FILE="test.csv"

SAVE_MODEL_NAME="classification.pkl"
# path to save model files
SAVE_MODEL_PATH=os.path.join(PACKAGE_ROOT,"trained_models")

# tagets and feature
TRAGET ='Loan_Status'

FEATURE=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
       'Property_Area','CoApplicantIncome']
#numarical columns in feature
NUM_FEATURES=['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
# catagorical column in feature

CAT_FEATURES=['Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Credit_History',
 'Property_Area']

# for transormation
# feature to encode it will be catagorical columns as we only encode them
FEATURES_TO_ENCODE=['Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Credit_History',
 'Property_Area']

FEATURE_TO_MODIFY='ApplicantIncome'
FEATURE_TO_ADD='CoApplicantIncome'

DROP_FEATURES=['CoApplicantIncome']

LOG_FEATURES=['ApplicantIncome','LoanAmount']