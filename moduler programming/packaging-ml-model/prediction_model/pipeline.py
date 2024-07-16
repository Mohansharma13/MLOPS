from sklearn.pipeline import Pipeline
from prediction_model.config import config
import prediction_model.processing.preprocessing as pp 
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

classification_pipeline=Pipeline(
    
    [
        ('mean_imputation',pp.MeanImputer(variables=config.NUM_FEATURES)),
        ('mode_imputer',pp.ModeImputer(variables=config.CAT_FEATURES)),
        ('Domian_processing',pp.Domainprocessing(variables_to_modify=config.FEATURE_TO_MODIFY,variable_to_add=config.FEATURE_TO_ADD)),
        ('drop features',pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('label_encoder',pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('log transformation',pp.LogTransforms(variables=config.LOG_FEATURES)),
        ('minmax_sacler',MinMaxScaler()),
        ('logistic_regression',LogisticRegression(random_state=0))
    ]
    
)