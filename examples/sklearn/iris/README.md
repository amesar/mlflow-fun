# mlflow-fun - sklearn - Iris Example

Simple Scikit-learn [DecisionTreeClassifier](http://scikit-learn.org/stable/modules/tree.html) that:
* Logs parameters and metrics
* Saves text artifacts: confusion_matrix.txt and classification_report.txt
* Saves plot artifact: simple_plot.png
* Saves model as a pickle file

Source: [train_iris_decision_tree.py](train_iris_decision_tree.py)

To run with standard main function:
```
cd iris
python train_iris_decision_tree.py 5 3
```

To run locally with the [MLproject](MLproject) file:
```
mlflow run . -Pmin_samples_leaf=5 -Pmax_depth=3
```

To run from git with the [MLproject](MLproject) file:
```
mlflow run https://github.com/amesar/mlflow-fun.git#examples/scikit-learn/iris -Pmin_samples_leaf=5 -Pmax_depth=3 -Ptag=RunFromGit
```
