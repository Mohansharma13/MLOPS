import os 
import mlflow 
import argparse
import time

def eval(p1,p2):
    output_mertic=p1**2 + p2**2
    return output_mertic


def main(inp1,inp2):
    mlflow.set_experiment("Demo_experiment")
    with mlflow.start_run():
        
        # setting tag
        mlflow.set_tag("version","1.0.0")
        
        # setting parameter
        mlflow.log_param("param1",inp1)
        mlflow.log_param("param2",inp2)
        
        metric=eval(inp1,inp2)
        # setting eval metric
        mlflow.log_metric("Eval_metric",metric)
        
        # making dir and writting someting in it
        os.makedirs("dummy",exist_ok=True)
        with open("dummy/example.txt","wt") as f:
            f.write(f"artifact created {time.asctime()}")
        
        # stroing dir under artifact
        mlflow.log_artifact("dummy")
        
        

# to use function with parameter we use argparase import 

if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--param1","-p1",type=int,default=5)
    args.add_argument("--param2","-p2",type=int,default=10)
    # to parse these argument 
    parsed_args=args.parse_args()
    
    # now to passing imput parameter to main function
    main(parsed_args.param1,parsed_args.param2)
    