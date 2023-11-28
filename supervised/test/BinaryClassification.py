# 

from ...supervised import Classification
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def ConfigureModel(model = Classification()):

    # Loading a model 
    model = model

    # Adding a new estimator
    model.add_estimator(estimator = ("New_Random_Forest", RandomForestClassifier()))

    # Remove old RandomForest
    model.remove_estimator("Random Forest")

    # Pass params to the New_Random_Forest estimator 
    model.pass_params(estimator = (RandomForestClassifier, {"n_estimators":300, "criterion":"gini"}))

    return model

def data():
    return load_breast_cancer(return_X_y = True)


if __name__ == '__main__':

    # Model/Data
    model = ConfigureModel()
    X, y = data()

    # Training
    model.fit_train(X, y, CV = 5)

    # Summary
    print(model.summary(pandas = False))