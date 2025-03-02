import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD THE DATASET & SPLIT THE DATASET
iris = load_iris()
X = iris.data
y = iris.target
X_train , X_test, y_train , y_test = train_test_split(X , y , test_size = 0.25 , random_state = 0)

# MLFLOW EXPERIMENT

mlflow.set_experiment('Iris_Classification_Experiment')

# LOGISTIC REGRESSION MODEL

with mlflow.start_run():

    lr_model = LogisticRegression(solver = 'liblinear' , C =1000 , max_iter = 50)
    lr_model.fit(X_train , y_train)


    lr_predictions = lr_model.predict(X_test)

    lr_accuracy = accuracy_score(y_test , lr_predictions)

    lr_conf_matrix = confusion_matrix(y_test , lr_predictions)

    mlflow.log_param('model' , 'Logistic Regression')
    mlflow.log_metric('accuracy', lr_accuracy)

    plt.figure(figsize = (6,6))
    sns.heatmap(lr_conf_matrix , annot = True , fmt = "d", cmap = 'Blues', xticklabels = iris.target_names, yticklabels = iris.target_names)
    plt.title("Confusion Matrix - Logistic Regression")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('lr_confusion_matrix.png')
    mlflow.log_artifact('lr_confusion_matrix.png')

    mlflow.sklearn.log_model(lr_model , 'logistic_regression_model')

    log_model_uri = "runs:/{}/logistic_regression_model".format(mlflow.active_run().info.run_id)
    registered_model_name = mlflow.register_model(log_model_uri , 'Logistic_Regression_Model')

    # RANDOM FOREST MODEL
    mlflow.end_run()  # End any active MLflow run

    print(f'Logistic Regression Model Accuracy : {lr_accuracy}')
    print('Confusion Matrix:')
    print(lr_conf_matrix)  


    with mlflow.start_run():
        rf_model = RandomForestClassifier(n_estimators = 10, max_depth = 5, min_samples_split = 10 , random_state = 0, criterion = 'entropy')            
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)

        rf_accuracy = accuracy_score(y_test , rf_predictions)
        print(f'Random Forest Model Accuracy : {rf_accuracy}')

        rf_conf_matrix = confusion_matrix(y_test , rf_predictions)
        print(f"Random Forest Confusion Matrix : {rf_conf_matrix}")

        print(rf_conf_matrix)

        mlflow.log_param('model' , 'Random Forest')
        mlflow.log_metric('accuracy' , 'rf_accuracy')

    plt.figure(figsize = (6,6))
    sns.heatmap(rf_conf_matrix , annot = True , fmt = "d" , cmap = 'Blues', xticklabels = iris.target_names, yticklabels = iris.target_names)
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('rf_confusion_matrix.png')
    mlflow.log_artifact('rf_confusion_matrix.png')

    mlflow.sklearn.log_model(rf_model , 'random_forest_model')

    rf_model_uri = "runs:/{}/random_forest_model".format(mlflow.active_run().info.run_id)
    registered_model_name = mlflow.register_model(log_model_uri , 'Random_Forest_Model')

    mlflow.end_run()

    print(f'Random Forest Model Accuracy :{rf_accuracy}')
    print('Confusion Matrix:')
    print(rf_conf_matrix)

    logistic_model_uri = "models:/Logistic_Regression_Model/latest"
    loaded_lr_model = mlflow.sklearn.load_model(logistic_model_uri)

    random_forest_model_uri = "models:/Random_Forest_Model/latest"
    loaded_rf_model = mlflow.sklearn.load_model(random_forest_model_uri)









        






