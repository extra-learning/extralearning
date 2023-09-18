# Authors: Liam Arguedas <iliamftw2013@gmail.com>
# License: BSD 3 clause

import pandas as pd
import numpy as np
import sklearn, xgboost, lightgbm, catboost

class Classification():
    def __init__(self, random_state = None, n_jobs = -1):
        self.random_state = random_state
        self.n_jobs = n_jobs
    
    def set_random_state(self, random_state):
        """      
        Controls both the randomness of the algorithms and reproducibility of the results
        
        Parameters
        ----------
        random_state: int or None, default = None
        """
        
        if not isinstance(random_state, (int, None)):
            raise TypeError(f"random_state must be int or None, not {type(random_state)}.")
        
        self.random_state = random_state
        
    def return_pandas(self, pandas = True):
        """
        The default output in the form of a pandas DataFrame or pandas Series, depending of the return.
        
        Parameters
        ----------
        pandas: Bool, default = True
        """
        
        if not isinstance(pandas, bool)):
            raise TypeError(f"Parameter must be Bool, not {type(pandas)}.")
        
        self.pandas = pandas
    
    def __binary__estimators(self):
        
        return {"Logistic Regression":sklearn.linear_model.LogisticRegression(random_state = self.random_state),
        "Decision Tree":sklearn.tree.DecisionTreeClassifier(random_state = self.random_state),
        "Random Forest":sklearn.ensemble.RandomForestClassifier(random_state = self.random_state),
        "Linear Support Vector Machine":sklearn.svm.SVC(random_state = self.random_state, probability = True, kernel = "linear"),
        "Linear Support Vector Machine":sklearn.svm.SVC(random_state = self.random_state, probability = True, kernel = "rbf"),
        "K-nearest neighbors":sklearn.neighbors.KNeighborsClassifier(),
        "Naive Bayes":sklearn.naive_bayes.GaussianNB(),
        "Gradient Boosting":sklearn.ensemble.GradientBoostingClassifier(random_state = self.random_state),
        "XGBoost":xgboost.XGBClassifier(random_state = self.random_state),
        "LightGBM":lightgbm.LGBMClassifier(random_state = self.random_state, verbose = -1),
        "CatBoost":catboost.CatBoostClassifier(random_state = self.random_state, verbose = False),
        "AdaBoost":sklearn.ensemble.AdaBoostClassifier(random_state = self.random_state),
        "Linear Discriminant":sklearn.discriminant_analysis.LinearDiscriminantAnalysis(),
        "Quadratic Discriminant":sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(),
        "Extra Trees":sklearn.ensemble.ExtraTreesClassifier(random_state = self.random_state),
        "Gaussian Process":sklearn.gaussian_process.GaussianProcessClassifier(random_state = self.random_state),
        "Multilayer perceptron":sklearn.neural_network.MLPClassifier(random_state = self.random_state)}
            
    def __multiclass__estimators(self):
        pass
    
    def get_params(self):
        pass
        