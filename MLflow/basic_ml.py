import os 
import mlflow 
import argparse
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd

def load_data():
    URL="http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        df=pd.read_csv(URL,sep=";")
        return df
    except Exception as e:
        raise e         

def eval_function(y_test,y_pred):
    R2=r2_score(y_true=y_test,y_pred=y_pred)
    mae=mean_absolute_error(y_true=y_test,y_pred=y_pred)
    mse=mean_squared_error(y_true=y_test,y_pred=y_pred)
    return mae,mse,R2

def main(alpha, l1_ratio):
    
    df=load_data()
    print(df.head())
    feature=df.drop(["quality"],axis=1)
    targets=df["quality"]
    x_train,x_test,y_train,y_test=train_test_split(feature,targets,test_size=0.2,random_state=2)
    
    # running experiment in mlflow
    mlflow.set_experiment("ml model 1")
    with mlflow.start_run():
        
        # logging the parameters
        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio",l1_ratio)
        
        model=ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=2)
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        
        mae,mse,r2=eval_function(y_test,y_pred)
        
        # logging eval metrics
        mlflow.log_metric("mse",mse)
        mlflow.log_metric("mae",mae)
        mlflow.log_metric("r2",r2)
        
        # logging our model also
        mlflow.sklearn.log_model(model,"trained_model") # model , folder
        
        
        





# to use function with parameter we use argparase import 

if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--alpha","-a",type=float,default=0.2)
    args.add_argument("--l1_ratio","-p2",type=float,default=0.3)
    # to parse these argument 
    parsed_args=args.parse_args()
    
    # now to passing imput parameter to main function
    main(parsed_args.alpha,parsed_args.l1_ratio)
    