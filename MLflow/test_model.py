import mlflow
from pyspark.sql.functions import struct, col
logged_model = 'runs:/f6699a2f1bda401ba6029c87eebbac01/trained_model'

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

# Predict on a Spark DataFrame.
df.withColumn('predictions', loaded_model(struct(*map(col, df.columns))))