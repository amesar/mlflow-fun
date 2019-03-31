from setuptools import setup

setup(name='mlflow_fun',
      version='0.0.1',
      description='MLflowFun tools',
      author='Andre',
      packages=['mlflow_fun',
                'mlflow_fun.common',
                'mlflow_fun.analytics',
                'mlflow_fun.metrics',
                'mlflow_fun.metrics.spark',
                'mlflow_fun.metrics.pandas'],
      zip_safe=False)
