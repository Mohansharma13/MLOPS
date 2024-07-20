import mlflow

mlflow.set_tracking_cri("http://localhost:5000")

exp_id=mlflow.create_experiment('Loan_prediction')


# with is the keyword for reading the file and it can handle all the exceptions
# example with {open("file.txt",'W') as w:}

with mlflow.start_run(run_name="decisiontreeclass") as run:
    pass


mlflow.end_run()