from setuptools import setup

setup(name='mlflow_fun_metrics',
      version='0.0.1',
      description='Create Spark tables from MLflow API with all run details to find best metrics',
      author='Andre',
      packages=['mlflow_fun',
                'mlflow_fun.metrics'],
      zip_safe=False)
