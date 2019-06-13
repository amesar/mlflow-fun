# Sample Decision Tree Classifier using MLflow

from __future__ import print_function
import matplotlib
matplotlib.use('Agg')

import sys, os, platform
from sklearn import datasets, metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
plt.rcdefaults()
import mlflow
import mlflow.sklearn
from mlflow import version

experiment_name = "py/sk/DecisionTree/Iris"

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
    print("Tag:     tag={}".format(tag))

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
    min_samples_leaf = int(sys.argv[1]) 
    max_depth = int(sys.argv[2]) 
    tag = sys.argv[3] if len(sys.argv) > 3 else ""
    dataset = datasets.load_iris()

    print("MLflow Version:", version.VERSION)
    print("MLflow Tracking URI:", mlflow.get_tracking_uri())

    print("experiment_name:",experiment_name)
    mlflow.set_experiment(experiment_name)
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    print("experiment_id:",experiment_id)

    source_name = os.path.basename(__file__)

    print("source_name:",source_name)
    with mlflow.start_run(source_name=source_name) as run:
        run_id = run.info.run_uuid
        print("run_id:",run_id)
        train(min_samples_leaf, max_depth, dataset.data, dataset.target)
        mlflow.set_tag("runner", tag)
        mlflow.set_tag("mlflow_version", version.VERSION)
        mlflow.set_tag("experiment_id", experiment_id)
        mlflow.set_tag("experiment_name", experiment_name)
        mlflow.set_tag("platform", platform.system())
