from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd
import numpy as np

app=FastAPI()
model_name='RF_Loan_model.joblib'
model=joblib.load(model_name)

# parsing class from Pydantic
class Loan(BaseModel):
	Gender: float
	Married: float
	Dependents: float
	Education: float
	Self_Employed: float
	LoanAmount: float
	Loan_Amount_Term: float
	Credit_History: float
	Property_Area: float
	TotalIncome: float 
	
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
	income = data['TotalIncome']

	# Making predictions 
	prediction = model.predict([[gender, married, dependents, education, self_employed, loan_amt, loan_term, credit_hist, property_area,income]])

	if prediction == 0:
		pred = 'Rejected'
	else:
		pred = 'Approved'

	return {'Status of Loan Application':pred}

if __name__ == '__main__':
	uvicorn.run(app, host='127.0.0.1', port=8000)
	
#     curl -X 'POST' \
#   'http://127.0.0.1:8000/predict' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: application/json' \
#   -d '{
#   "Gender": 0,
#   "Married": 0,
#   "Dependents": 0,
#   "Education": 0,
#   "Self_Employed": 0,
#   "LoanAmount": 0,
#   "Loan_Amount_Term": 0,
#   "Credit_History": 0,
#   "Property_Area": 0,
#   "TotalIncome": 0
# }'