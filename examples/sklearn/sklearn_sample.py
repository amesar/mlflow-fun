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


def run(min_samples_leaf, max_depth, dataset_data, dataset_target, idx):
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

    print("Index:",idx)
    print("Params:  min_samples_leaf={} max_depth={}".format(min_samples_leaf,max_depth))
    print("Metrics: auc={} accuracy_score={} zero_one_loss={}".format(auc,accuracy_score,zero_one_loss))
    
    fig = make_simple_plot(auc, accuracy_score, zero_one_loss)
    plot_filename = "simple_plot.png"
    fig.savefig(plot_filename)
    mlflow.log_artifact(plot_filename)


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
    num_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    min_samples_leaf = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    dataset = datasets.load_iris()

    current_file = os.path.basename(__file__)
    #experiment_name = current_file.replace(".py","")
    experiment_name = "SklearnDecisionTree.py"
    experiment_id = mlflow_utils.get_or_create_experiment_id(experiment_name)

    print("Params: num_iters={} min_samples_leaf={} max_depth={}".format(num_iters,min_samples_leaf,max_depth))
    for j in range(num_iters):
        with mlflow.start_run(experiment_id=experiment_id, source_name=current_file):
            run(min_samples_leaf+j, max_depth+j, dataset.data, dataset.target, j)
