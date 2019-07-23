from __future__ import print_function
import os, json, requests
from mlflow_fun.common import mlflow_utils
from mlflow_fun.common import MlflowFunException

API_PATH = "api/2.0/preview/mlflow"

''' Wrapper for get and post methods for MLflow REST API. '''
class HttpClient(object):
    def __init__(self, base_uri=None, token=None):
        (host,token) = mlflow_utils.get_mlflow_host_token(base_uri)
        if host is None:
            raise MlflowFunException("MLflow host or token is not configured correctly")
        base_uri = os.path.join(host,API_PATH)
        self.base_uri = base_uri
        self.token = token

    def get(self, resource):
        """ Executes an HTTP GET call
        :param resource: Relative path name of resource such as runs/search
        """
        uri = self._mk_uri(resource)
        rsp = requests.get(uri, headers=self._mk_headers())
        self._check_response(rsp, uri)
        return json.loads(rsp.text)

    def post(self, resource, data):
        """ Executes an HTTP POST call
        :param resource: Relative path name of resource such as runs/search
        :param data: Post request payload
        """
        uri = self._mk_uri(resource)
        data=json.dumps(data)
        rsp = requests.post(uri, headers=self._mk_headers(), data=data)
        self._check_response(rsp,uri)
        return json.loads(rsp.text)

    def _mk_headers(self):
        return {} if self.token is None else {'Authorization': 'Bearer '+self.token}

    def _mk_uri(self, resource):
        return self.base_uri + "/" + resource

    def _check_response(self, rsp, uri):
        if rsp.status_code < 200 or rsp.status_code > 299:
            raise MlflowFunException("HTTP status code: {} Reason: {} URL: {}".format(str(rsp.status_code),rsp.reason,uri))
