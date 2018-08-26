# Sample Decision Tree Classifier using MLflow

from __future__ import print_function
import sys, os
from sklearn import datasets, metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
plt.rcdefaults()
import mlflow
import mlflow.sklearn
import mlflow_utils
from mlflow import version

def train(min_samples_leaf, max_depth, dataset_data, dataset_target):
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("max_depth", max_depth)

    clf = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, max_depth=max_depth)
    print("Classifier:",clf)
    clf.fit(dataset_data, dataset_target)
    mlflow.sklearn.log_model(clf, "model") 

    expected = dataset_target
    predicted = clf.predict(dataset_data)

    auc = metrics.auc(expected, predicted)
    accuracy_score = metrics.accuracy_score(expected, predicted)
    zero_one_loss = metrics.zero_one_loss(expected, predicted)

    mlflow.log_metric("auc", auc)
    mlflow.log_metric("accuracy_score", accuracy_score)
    mlflow.log_metric("zero_one_loss", zero_one_loss)

    print("Params:  min_samples_leaf={} max_depth={}".format(min_samples_leaf,max_depth))
    print("Metrics: auc={} accuracy_score={} zero_one_loss={}".format(auc,accuracy_score,zero_one_loss))

    write_artifact('confusion_matrix.txt',str(metrics.confusion_matrix(expected, predicted)))
    write_artifact('classification_report.txt',metrics.classification_report(expected, predicted))
    
    fig = make_simple_plot(auc, accuracy_score, zero_one_loss)
    plot_filename = "simple_plot.png"
    fig.savefig(plot_filename)
    mlflow.log_artifact(plot_filename)

def write_artifact(file, data):
    with open(file, 'w') as f:
        f.write(data)
    mlflow.log_artifact(file)

def make_simple_plot(auc, accuracy_score, zero_one_loss):
    fig = plt.figure()
    objects = ('auc', 'accuracy_score', 'zero_one_loss')
    y_pos = range(len(objects))
    performance = [auc, accuracy_score, zero_one_loss]
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Value') 
    plt.title('Metrics')
    plt.close(fig)
    return fig

if __name__ == "__main__":
    min_samples_leaf = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    dataset = datasets.load_iris()

    print("MLflow Version:", version.VERSION)
    assert "0.4.2" == version.VERSION
    print("MLflow Tracking URI:", mlflow.get_tracking_uri())

    experiment_name = "Iris/DecisionTree"
    experiment_id = mlflow_utils.get_or_create_experiment_id(experiment_name)
    source_name = os.path.basename(__file__)

    print("Params: min_samples_leaf={} max_depth={}".format(min_samples_leaf,max_depth))
    with mlflow.start_run(experiment_id=experiment_id, source_name=source_name):
        train(min_samples_leaf, max_depth, dataset.data, dataset.target)
