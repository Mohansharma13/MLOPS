# we are going to make custom data preprocessing functions
# Numerical Imputation - mean
from sklearn.base import BaseEstimator,TransformerMixin
from prediction_model.config import config
import numpy as np

#mean imputer

class MeanImputer(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
    
    def fit(self,X,y=None):
        self.mean_dict = {}
        for col in self.variables:
            self.mean_dict[col] = X[col].mean()
        return self
    
    def transform(self,X):
        X = X.copy()
        for col in self.variables:
            X[col].fillna(self.mean_dict[col],inplace=True)
        return X


# mode imputer transformer
class ModeImputer(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
    
    def fit(self,X,y=None):
        self.mode_dict = {}
        for col in self.variables:
            self.mode_dict[col] = X[col].mode()[0]
        return self
    
    def transform(self,X):
        X = X.copy()
        for col in self.variables:
            X[col].fillna(self.mode_dict[col],inplace=True)
        return X

# drop the cloumns accoding to preprocessing
class DropColumns(BaseEstimator,TransformerMixin):
    def __init__(self,variables_to_drop=None):
        self.variables_to_drop = variables_to_drop
    
    def fit(self,X,y=None):
        return self 
    
    def transform(self,X):
        X = X.copy()
        X=X.drop(Columns=self.variables_to_drop)
        return X

# mergering two coumns accoding to our preprocssing

class Domainprocessing(BaseEstimator,TransformerMixin):
    def __init__(self,variables_to_modify=None,variable_to_add=None):
        self.variables_to_modify = variables_to_modify
        self.variable_to_add = variable_to_add
    def fit(self,X,y=None):
        return self 
    
    def transform(self,X):
        X = X.copy()
        for feature in self.variables_to_modify:
            X[feature]=X[feature]+X[self.variable_to_add]
        return X  
    
# label encoder you can also directly use skleran encoder but for demo we are going to code it
 
class CustomLabelEncoder(BaseEstimator,TransformerMixin):
    def __init__(self, variables=None):
        self.variables=variables
    
    def fit(self, X,y):
        self.label_dict = {}
        for var in self.variables:
            t = X[var].value_counts().sort_values(ascending=True).index 
            self.label_dict[var] = {k:i for i,k in enumerate(t,0)}
        return self
    
    def transform(self,X):
        X=X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.label_dict[feature])
        return X


# Try out Log Transformation
class LogTransforms(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        for col in self.variables:
            X[col] = np.log(X[col])
        return X