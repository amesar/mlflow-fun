import os

''' Normal local file system '''
class LocalFileApi(object):
    def makedirs(self, path):
        os.makedirs(path)

''' Running within Databricks '''
class DbfsFileApi(object):
    def makedirs(self, path):
        os.makedirs(self.make_fuse_path(path))
    def mk_fuse_path(self,path):
        return path.replace("dbfs:/","/dbfs/")

''' Using Databricks Connect '''
class DbfsFileConnectApi(object):
    def __init__(self):
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        self.dbutils = DBUtils(spark.sparkContext)
    def makedirs(self, path):
        self.dbutils.fs.mkdirs(path)

''' Get appropriate FileApi implementation per path name and context '''
def get_file_api(path):
    if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
        return DbfsFileApi()
    return DbfsFileConnectApi() if path.startswith("dbfs:") else LocalFileApi()
