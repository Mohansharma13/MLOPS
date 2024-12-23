from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

app=FastAPI()
model_name='my_trained_model_v1.pkl'
model=joblib.load(model_name)

# parsing class from Pydantic
class Loan(BaseModel):
	Gender: float
	Married: float
	Dependents: float
	Education: float
	Self_Employed: float
	ApplicantIncome: float 
	LoanAmount: float
	Loan_Amount_Term: float
	Credit_History: float
	Property_Area: float
	
@app.get("/") # Decorator for GET requests to the root ("/") route
def index():
    return {"message":"Welcome to Loan Prediction API"} 

@app.post("/predict") # Decorator for POST requests to the "/predict" route
def predict_loan_status(loan_details: Loan):
	data = loan_details.model_dump()
	gender = data['Gender']
	married = data['Married']
	dependents = data['Dependents']
	education = data['Education']
	self_employed = data['Self_Employed']
	loan_amt = data['LoanAmount']
	loan_term = data['Loan_Amount_Term']
	credit_hist = data['Credit_History']
	property_area = data['Property_Area']
	income = data['ApplicantIncome']

	# Making predictions 
	prediction = model.predict([[gender, married, dependents, education, self_employed, income,loan_amt, loan_term, credit_hist, property_area]])
	
	logging.info(f"Incoming request payload: {prediction}")
	if prediction == 'N':
		pred = 'Rejected'
	if prediction == 'Y':
		pred = 'Approved'
	if prediction != 'N' and prediction != 'Y':
		pred = 'Error'

	return {'Status of Loan Application':pred}

if __name__ == '__main__':
	uvicorn.run(app, host='127.0.0.1', port=8000)
	
# Run the following command in the terminal to start the server
# '{
#   "Gender": 1.0,
#   "Married": 0.0,
#   "Dependents": 0.0,
#   "Education": 0.0,
#   "Self_Employed": 1.0,
#   "ApplicantIncome": 0.39517964,
#   "LoanAmount": 0.51301475,
#   "Loan_Amount_Term": 0.9220137,
#   "Credit_History": 1,
#   "Property_Area": 0.5
# }'

# {
#         "Gender": 1.0,
#         "Married": 0.0,
#         "Dependents": 0.0,
#         "Education": 0.0,
#         "Self_Employed": 1.0,
#         "ApplicantIncome": 3951.79,
#         "LoanAmount": 513.01,
#         "Loan_Amount_Term": 922.01,
#         "Credit_History": 1.0,
#         "Property_Area": 0.5
#     }