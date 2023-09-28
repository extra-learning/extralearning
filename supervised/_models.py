# Authors: Liam Arguedas <iliamftw2013@gmail.com>
# License: BSD 3 clause

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier

class Classification():
    
    def __init__(self, multi_class = None, random_state = None, n_jobs = None, verbose = True) -> None:
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
        
        assert multi_class is None or multi_class in ["ovr", "ovo", "auto"], TypeError(f'multi_class must be str: "ovr", "ovo", "auto" or None, not {type(multi_class)}.')
        assert random_state is None or isinstance(random_state, int), TypeError(f"random_state must be int or None, not {type(random_state)}.")
        assert n_jobs is None or isinstance(n_jobs, int), TypeError(f"n_jobs must be int or None, not {type(n_jobs)}.")
        assert isinstance(verbose, bool), TypeError(f"verbose must be bool, not {type(verbose)}.")
        
        self.multi_class = multi_class
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.__estimators = [
                ("Logistic Regression", LogisticRegression(random_state = self.random_state, n_jobs = self.n_jobs)),
                ("Decision Tree", DecisionTreeClassifier(random_state = self.random_state)),
                ("Random Forest", RandomForestClassifier(random_state = self.random_state, n_jobs = self.n_jobs)),
                ("Linear Support Vector Machine", SVC(random_state = self.random_state, probability = True, kernel = "linear")),
                ("RBF Support Vector Machine", SVC(random_state = self.random_state, probability = True, kernel = "rbf")),
                ("K-nearest neighbors", KNeighborsClassifier(n_jobs = self.n_jobs)),
                ("Naive Bayes", GaussianNB()),
                ("Gradient Boosting", GradientBoostingClassifier(random_state = self.random_state)),
                ("XGBoost", XGBClassifier(random_state = self.random_state, nthread = self.n_jobs)),
                ("LightGBM", LGBMClassifier(random_state = self.random_state, verbose = -1, n_jobs = self.n_jobs)),
                ("CatBoost", CatBoostClassifier(random_state = self.random_state, verbose = False, thread_count = self.n_jobs)),
                ("AdaBoost", AdaBoostClassifier(random_state = self.random_state)),
                ("Linear Discriminant", LinearDiscriminantAnalysis()),
                ("Quadratic Discriminant", QuadraticDiscriminantAnalysis()),
                ("Extra Trees", ExtraTreesClassifier(random_state = self.random_state, n_jobs = self.n_jobs)),
                ("Gaussian Process", GaussianProcessClassifier(random_state = self.random_state, n_jobs = self.n_jobs)),
                ("Multilayer perceptron", MLPClassifier(random_state = self.random_state))
            ]
            
    def return_pandas(self, pandas = True) -> None:
        """
        The default output in the form of a pandas DataFrame or pandas Series, depending of the return.
        
        Parameters
        ----------
        pandas: Bool, default = True
        """
        
        if not isinstance(pandas, bool):
            raise TypeError(f"Parameter must be Bool, not {type(pandas)}.")
        
        self.__pandas = pandas
   
    def get_estimators(self) -> list:
        """
        Returns a list of tuples containing the current estimators.
        """
        return self.__estimators
        
    def add_estimator(self, estimator: tuple) -> None:
        """
        Parameters
        ----------
        estimator: tuple, (name, estimator)
            - name: str, name of the estimator
            - estimator: A classifier that implements `.fit()`, `.predict()` and `.predict_proba()` if available.
                
        Examples
        --------
        >>> from extralearning import Classification
        >>> from sklearn.ensemble import RandomForestClassifier
        
        >>> model = Classification()
        
        >>> model.add_estimator(estimator = ("Random Forest", RandomForestClassifier()))
        
        Note: To check the default estimators, you can use the `.get_estimators()` method.
        """
         
        if self.multi_class is None:
            self.__estimators.append(estimator)
            
        else:
            pass # PENDING
    
    def remove_estimator(self, estimator) -> None:
        
        """
        Removes an estimator based on name or index position.
        
        Parameters
        ----------
        estimator: str or int,
            - str: name of estimator listed in `.get_estimators()` to be removed.
            - int: Index position of estimator listed in `.get_estimators()` to be removed.
        """
        assert isinstance(estimator, (str, int)), TypeError(f"estimator must be int or STR, not {type(estimator)}")
        
        if isinstance(estimator, str):
            self.__estimators.pop([model[0] for model in self.__estimators].index(estimator))
        
        else:
            self.__estimators.pop(estimator)    
    
    def reset_estimators(self) -> None:
        "Use to reset the estimators to be used to the original list"
        self.__estimators = [
                ("Logistic Regression", LogisticRegression(random_state = self.random_state, n_jobs = self.n_jobs)),
                ("Decision Tree", DecisionTreeClassifier(random_state = self.random_state)),
                ("Random Forest", RandomForestClassifier(random_state = self.random_state, n_jobs = self.n_jobs)),
                ("Linear Support Vector Machine", SVC(random_state = self.random_state, probability = True, kernel = "linear")),
                ("RBF Support Vector Machine", SVC(random_state = self.random_state, probability = True, kernel = "rbf")),
                ("K-nearest neighbors", KNeighborsClassifier(n_jobs = self.n_jobs)),
                ("Naive Bayes", GaussianNB()),
                ("Gradient Boosting", GradientBoostingClassifier(random_state = self.random_state)),
                ("XGBoost", XGBClassifier(random_state = self.random_state, nthread = self.n_jobs)),
                ("LightGBM", LGBMClassifier(random_state = self.random_state, verbose = -1, n_jobs = self.n_jobs)),
                ("CatBoost", CatBoostClassifier(random_state = self.random_state, verbose = False, thread_count = self.n_jobs)),
                ("AdaBoost", AdaBoostClassifier(random_state = self.random_state)),
                ("Linear Discriminant", LinearDiscriminantAnalysis()),
                ("Quadratic Discriminant", QuadraticDiscriminantAnalysis()),
                ("Extra Trees", ExtraTreesClassifier(random_state = self.random_state, n_jobs = self.n_jobs)),
                ("Gaussian Process", GaussianProcessClassifier(random_state = self.random_state, n_jobs = self.n_jobs)),
                ("Multilayer perceptron", MLPClassifier(random_state = self.random_state))
            ]
    
    def get_params(self)  -> list:
        """Returns a list of tuples containing the estimator and their default parameters"""
        return [(estimator[1], estimator[1].get_params()) for estimator in self.__estimators]   
        
    def pass_params(self, estimator, overwrite = True):
        """
        Pass custom parameters to an estimator.
        
        Parameters
        ----------
        estimator: tuple, (estimator, params)
            - estimator: A classifier that implements `.fit()`, `.predict()` and `.predict_proba()` if available.
            - params: A dictionary containing the params.
            
        overwrite: bool, default = True
            If `True` it will overwrite the current estimator in list, if `False`, method `.add_estimator()` will be called 
            and both estimators are going to be kept.
            
        Examples
        --------
        >>> from extralearning import Classification
            
        >>> model = Classification()
        
        >>> model.pass_params(estimator = (RandomForestClassifier, {"n_estimators":300, "criterion":"gini"}))
        """
        
        if not overwrite:
            self.add_estimator((str(type(estimator[0]())), estimator[0](**estimator[1])))
            
        else:
            if overwrite and type(estimator[0]()) not in [type(model[1]) for model in self.__estimators]:
                raise ValueError(f"{estimator[0]()} not in estimator list, if you want to add a new estimator you should use .add_estimator() or set overwrite to False")
            
            __ReplacedEstimator = [type(model[1]) for model in self.__estimators].index(type(estimator[0]()))
            
            self.__estimators[__ReplacedEstimator] = (self.__estimators[__ReplacedEstimator][0], estimator[0](**estimator[1]))
        
    def fit()