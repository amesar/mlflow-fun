# Databricks notebook source
import mlflow.pyfunc

# COMMAND ----------

class WrappedModelForUDF(mlflow.pyfunc.PythonModel):

    def __init__(self, ordered_df_columns, model_artifact):
        self.ordered_df_columns = ordered_df_columns
        self.model_artifact = model_artifact

    def load_context(self, context):
        import mlflow.pyfunc
        self.spark_pyfunc = mlflow.pyfunc.load_model(context.artifacts[self.model_artifact])

    def predict(self, context, model_input):
        renamed_input = model_input.rename(
            columns={
                str(index): column_name for index, column_name
                    in list(enumerate(self.ordered_df_columns))
            }
        )
        return self.spark_pyfunc.predict(renamed_input)

# COMMAND ----------

def log_udf_model(artifact_path, ordered_columns, run_id):
    udf_artifact_path = "udf-{}".format(artifact_path)
    model_uri = "runs:/{}/{}".format(run_id, artifact_path)
    mlflow.pyfunc.log_model(
        artifact_path = udf_artifact_path,
        python_model = WrappedModelForUDF(ordered_columns, artifact_path),
        artifacts={ artifact_path: model_uri }
    )
    return udf_artifact_path

# COMMAND ----------

import mlflow.spark

def log_spark_and_udf_models(model, artifact_path, run_id, ordered_columns):   
  mlflow.spark.log_model(model, artifact_path)
  return log_udf_model(artifact_path, ordered_columns, run_id)