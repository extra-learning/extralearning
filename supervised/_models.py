# Authors: Liam Arguedas <iliamftw2013@gmail.com>
# License: BSD 3 clause

import pandas as pd
import numpy as np
import sklearn, xgboost, lightgbm, catboost

class Classification():
    def __init__(self, multi_class = None, random_state = None, n_jobs = None, verbose = True):
        """
        Summarizes variety of Classification Machine Learning models into one instance.
        
        Parameters
        ----------
        multi_class: {"ovr", "ovo", "auto"} or None, default = None
            Categories (unique values) per feature:
            
            - None : Indicates that the problem requires a binary classification, and there is no need for multiclass functionality.
            - "ovr" : Each class is treated as a binary target while the rest are treated as the other class. 
            - "ovo" : The classifier is trained for each pair of classes to determine the class with the most votes from pairwise comparisons.
            
        random_state: int, default = None
            Controls the seed for generating random numbers, ensuring reproducibility in random processes such as data shuffling or initializations.
        
        n_jobs: int, default = None
            The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees.
            None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
        
        verbose: bool, default = True
            Controls the extralearning verbosity when fitting and predicting. 
            Note: All estimator verbose are set to 0 or False, meaning no output other than extralearning information.
        """
        self.multi_class = multi_class
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
            
    def return_pandas(self, pandas = True):
        """
        The default output in the form of a pandas DataFrame or pandas Series, depending of the return.
        
        Parameters
        ----------
        pandas: Bool, default = True
        """
        
        if not isinstance(pandas, bool)):
            raise TypeError(f"Parameter must be Bool, not {type(pandas)}.")
        
        self.__pandas = pandas
   
    def __estimators(self):
        
        if self.multi_class is None: 
            
            self.estimators = {
                "Logistic Regression":sklearn.linear_model.LogisticRegression(random_state = self.random_state, n_jobs = self.n_jobs),
                "Decision Tree":sklearn.tree.DecisionTreeClassifier(random_state = self.random_state, n_jobs = self.n_jobs),
                "Random Forest":sklearn.ensemble.RandomForestClassifier(random_state = self.random_state, n_jobs = self.n_jobs),
                "Linear Support Vector Machine":sklearn.svm.SVC(random_state = self.random_state, probability = True, kernel = "linear", n_jobs = self.n_jobs),
                "Linear Support Vector Machine":sklearn.svm.SVC(random_state = self.random_state, probability = True, kernel = "rbf", n_jobs = self.n_jobs),
                "K-nearest neighbors":sklearn.neighbors.KNeighborsClassifier(n_jobs = self.n_jobs),
                "Naive Bayes":sklearn.naive_bayes.GaussianNB(n_jobs = self.n_jobs),
                "Gradient Boosting":sklearn.ensemble.GradientBoostingClassifier(random_state = self.random_state, n_jobs = self.n_jobs),
                "XGBoost":xgboost.XGBClassifier(random_state = self.random_state, nthread = self.n_jobs),
                "LightGBM":lightgbm.LGBMClassifier(random_state = self.random_state, verbose = -1, n_jobs = self.n_jobs),
                "CatBoost":catboost.CatBoostClassifier(random_state = self.random_state, verbose = False, thread_count = self.n_jobs),
                "AdaBoost":sklearn.ensemble.AdaBoostClassifier(random_state = self.random_state, n_jobs = self.n_jobs),
                "Linear Discriminant":sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_jobs = self.n_jobs),
                "Quadratic Discriminant":sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(n_jobs = self.n_jobs),
                "Extra Trees":sklearn.ensemble.ExtraTreesClassifier(random_state = self.random_state, n_jobs = self.n_jobs),
                "Gaussian Process":sklearn.gaussian_process.GaussianProcessClassifier(random_state = self.random_state, n_jobs = self.n_jobs),
                "Multilayer perceptron":sklearn.neural_network.MLPClassifier(random_state = self.random_state, n_jobs = self.n_jobs)
                }
        else:
            self.estimators = {
                
            }          
    
    def add_estimator(self, estimator: tuple):
        if estimator not in self.__binary__estimators
    
    def get_params(self):
        pass
        