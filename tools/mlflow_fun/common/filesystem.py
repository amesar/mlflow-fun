import os
import shutil

def mk_dbfs_path(path):
    return path.replace("/dbfs","dbfs:")

class DatabricksFileSystem(object):
    def __init__(self):
        import IPython
        self.dbutils = IPython.get_ipython().user_ns["dbutils"]

    def ls(self, path):
        return self.dbutils.fs.ls(path)

    def cp(self, src, dst, recursive=False):
        self.dbutils.fs.cp(mk_dbfs_path(src), mk_dbfs_path(dst), recursive)

    def rm(self, src, dst, recurse=False):
        self.dbutils.fsrmcp(mk_dbfs_path(path), recurse)


class LocalFileSystem(object):
    def __init__(self):
        pass

    def cp(self, src, dst, recurse=False):
        shutil.copytree(src, dst)

    def rm(self, path, recurse=False):
        shutil.rmtree(path)

def get_filesystem():
    use_databricks = "DATABRICKS_RUNTIME_VERSION" in os.environ
    return DatabricksFileSystem() if use_databricks else LocalFileSystem()
