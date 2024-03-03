# Authors: Liam Arguedas <iliamftw2013@gmail.com>
# License: BSD 3 clause

# EstimatorClass
from .estimatorclass import *

# SummaryClass
from .summaryclass import *

# Core frameworks
import pandas as pd
import numpy as np
import warnings

# Sklearn metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Sklearn Model Selection
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier


class Classification(EstimatorClass, SummaryClass):
    def __init__(self, random_state=None, n_jobs=None, ignore_warnings=True) -> None:
        """
        Summarizes variety of Classification Machine Learning models into one instance.

        Parameters
        ----------
        random_state: int, default = None
            Controls the seed for generating random numbers, ensuring reproducibility in random processes such as data shuffling or initializations.

        n_jobs: int, default = None
            The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees.
            None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.

        ignore_warnings: bool, default = True
            Use to set warnings verbose level, if set to `True` verbose will be ignore, and if set to `False` verbose will be printed.

        Examples
        --------
        >>> from extralearning import Classification

        >>> model = Classification(random_state=None, n_jobs=None, ignore_warnings=True)
        """

        assert random_state is None or isinstance(random_state, int), TypeError(
            f"random_state must be int or None, not {type(random_state)}."
        )
        assert n_jobs is None or isinstance(n_jobs, int), TypeError(
            f"n_jobs must be int or None, not {type(n_jobs)}."
        )
        assert isinstance(ignore_warnings, bool), TypeError(
            f"ignore_warnings must be bool, not {type(ignore_warnings)}."
        )

        self.n_jobs = n_jobs
        self.random_state = random_state
        self.__estimators = [
            (
                "Logistic Regression",
                LogisticRegression(random_state=self.random_state, n_jobs=self.n_jobs),
            ),
            ("Decision Tree", DecisionTreeClassifier(random_state=self.random_state)),
            (
                "Random Forest",
                RandomForestClassifier(
                    random_state=self.random_state, n_jobs=self.n_jobs
                ),
            ),
            (
                "Linear Support Vector Machine",
                SVC(random_state=self.random_state, probability=True, kernel="linear"),
            ),
            (
                "RBF Support Vector Machine",
                SVC(random_state=self.random_state, probability=True, kernel="rbf"),
            ),
            ("Knearest neighbors", KNeighborsClassifier(n_jobs=self.n_jobs)),
            ("Naive Bayes", GaussianNB()),
            (
                "Gradient Boosting",
                GradientBoostingClassifier(random_state=self.random_state),
            ),
            (
                "XGBoost",
                XGBClassifier(random_state=self.random_state, nthread=self.n_jobs),
            ),
            (
                "LightGBM",
                LGBMClassifier(
                    random_state=self.random_state, verbose=-1, n_jobs=self.n_jobs
                ),
            ),
            (
                "CatBoost",
                CatBoostClassifier(
                    random_state=self.random_state,
                    verbose=False,
                    thread_count=self.n_jobs,
                ),
            ),
            ("AdaBoost", AdaBoostClassifier(random_state=self.random_state)),
            ("Linear Discriminant", LinearDiscriminantAnalysis()),
            ("Quadratic Discriminant", QuadraticDiscriminantAnalysis()),
            (
                "Extra Trees",
                ExtraTreesClassifier(
                    random_state=self.random_state, n_jobs=self.n_jobs
                ),
            ),
            (
                "Gaussian Process",
                GaussianProcessClassifier(
                    random_state=self.random_state, n_jobs=self.n_jobs
                ),
            ),
            ("Multilayer perceptron", MLPClassifier(random_state=self.random_state)),
        ]

        EstimatorClass.__init__(self,estimators=self.__estimators)

        # Settings
        if ignore_warnings:
            warnings.filterwarnings("ignore")

    def __init_evaluation_arrays(self) -> None:
        """
        Initializes the evaluation holding arrays
        """
        self.accuracy = list()
        self.precision = list()
        self.recall = list()
        self.f1score = list()
        self.auc_roc = list()
        self.estimator_name_list = list()
        self.fold = list()

    def __evaluate_model(
        self, estimator, estimator_name, X_train, X_validation, y_train, y_validation
    ) -> None:
        """
        Trains, predicts then evaluates performance of a model.
        """

        # Loading the model
        model = estimator

        # Fitting the model
        model.fit(X_train.values, y_train.values)

        # Predicting validation fold
        prediction = model.predict(X_validation)

        try:
            proba_prediction = model.predict_proba(X_validation)
            self.auc_roc.append(
                roc_auc_score(
                    y_validation,
                    proba_prediction[:, 1] if self.__labels < 3 else proba_prediction,
                    multi_class="ovr" if self.__labels > 2 else "raise",
                    average="micro" if self.__labels > 2 else "weighted",
                )
            )
        except:
            self.auc_roc.append(np.nan)

        # Recording results
        self.estimator_name_list.append(estimator_name)
        self.accuracy.append(accuracy_score(y_validation, prediction))
        self.precision.append(
            precision_score(y_validation, prediction, average="weighted")
        )
        self.recall.append(recall_score(y_validation, prediction, average="weighted"))
        self.f1score.append(f1_score(y_validation, prediction, average="weighted"))

    def __verbose(self, text: str, end="\n") -> None:
        if self.__verbose_status:
            print(text, end=end)

    def __CV(self, __stratified, __folds, __params) -> StratifiedKFold:
        """
        Defines the Cross-Validation algorithm to be use by the `.fit()` method.
        """

        __validator = StratifiedKFold if __stratified else KFold

        return (
            __validator(n_splits=__folds, random_state=self.random_state, shuffle=True)
            if __params is None
            else __validator(**__params)
        )

    def reset_estimators(self) -> None:
        """
        Use to reset the estimators to be used to the original list

        Examples
        --------
        >>> from extralearning import Classification

        >>> model = Classification()

        >>> model.remove_estimator(estimator = "Random Forest")
        >>> model.remove_estimator(estimator = 5)

        >>> model.reset_estimators()

        """

        self.__estimators = [
            (
                "Logistic Regression",
                LogisticRegression(random_state=self.random_state, n_jobs=self.n_jobs),
            ),
            ("Decision Tree", DecisionTreeClassifier(random_state=self.random_state)),
            (
                "Random Forest",
                RandomForestClassifier(
                    random_state=self.random_state, n_jobs=self.n_jobs
                ),
            ),
            (
                "Linear Support Vector Machine",
                SVC(random_state=self.random_state, probability=True, kernel="linear"),
            ),
            (
                "RBF Support Vector Machine",
                SVC(random_state=self.random_state, probability=True, kernel="rbf"),
            ),
            ("K-nearest neighbors", KNeighborsClassifier(n_jobs=self.n_jobs)),
            ("Naive Bayes", GaussianNB()),
            (
                "Gradient Boosting",
                GradientBoostingClassifier(random_state=self.random_state),
            ),
            (
                "XGBoost",
                XGBClassifier(random_state=self.random_state, nthread=self.n_jobs),
            ),
            (
                "LightGBM",
                LGBMClassifier(
                    random_state=self.random_state, verbose=-1, n_jobs=self.n_jobs
                ),
            ),
            (
                "CatBoost",
                CatBoostClassifier(
                    random_state=self.random_state,
                    verbose=False,
                    thread_count=self.n_jobs,
                ),
            ),
            ("AdaBoost", AdaBoostClassifier(random_state=self.random_state)),
            ("Linear Discriminant", LinearDiscriminantAnalysis()),
            ("Quadratic Discriminant", QuadraticDiscriminantAnalysis()),
            (
                "Extra Trees",
                ExtraTreesClassifier(
                    random_state=self.random_state, n_jobs=self.n_jobs
                ),
            ),
            (
                "Gaussian Process",
                GaussianProcessClassifier(
                    random_state=self.random_state, n_jobs=self.n_jobs
                ),
            ),
            ("Multilayer perceptron", MLPClassifier(random_state=self.random_state)),
        ]

    def fit_train(
        self,
        X,
        y,
        CV=3,
        CV_Stratified=True,
        CV_params=None,
        verbose=True,
    ) -> None:
        """
        Use to train each estimator, make predictions on test data and evaluate the estimator.

        Parameters
        ----------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted to dtype=np.float32.
            If a sparse matrix is provided, it will be converted into a sparse csc_matrix.

        y: array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).

        CV: int or None, default = 3
            Specifies the number of fold's to train each estimator, if set to `None` only one training will be done and evaluate.

        CV_Stratified: bool, default = True
            If set to `True` Stratified Cross-validation will be perform, if set to `False` normal KFold will be use.

        CV_params: dict or None, default = None
            Use to pass params to the StratifiedKFold or KFold cross-validation.

        verbose: bool, default = True
            Verbose status, if set to `True` all transformation verbose will be printed, if set to `False` transformer will be silenced.
        """

        if not isinstance(X, pd.DataFrame) and not isinstance(y, pd.Series):
            try:
                X, y = pd.DataFrame(X), pd.Series(y)
            except TypeError:
                print(
                    f"X must be array-like of shape (n_samples, n_features) or pandas.DataFrame and y array-like of shape (n_samples,) or (n_samples, n_outputs)"
                )

        # Update verbose
        self.__verbose_status = verbose

        # Initiation/reseting evaluation arrays
        self.__init_evaluation_arrays()

        # labels
        self.__labels = y.nunique()

        if CV is None:
            self.__verbose("")
            self.__verbose("Trainning:")
            self.__verbose("")
            for name, model in self.__estimators:
                X_train, X_validation, y_train, y_validation = train_test_split(
                    X, y, random_state=self.random_state
                )

                self.__verbose(f"- {name}")
                self.__evaluate_model(
                    model, name, X_train, X_validation, y_train, y_validation
                )

        else:
            CROSS_VALIDATION = self.__CV(CV_Stratified, CV, CV_params)

            for fold, (train_index, validation_index) in enumerate(
                CROSS_VALIDATION.split(X, y)
            ):
                X_train, y_train = X.loc[train_index], y.loc[train_index]

                X_validation, y_validation = (
                    X.loc[validation_index],
                    y.loc[validation_index],
                )

                self.__verbose("")
                self.__verbose(f"Trainning: Fold {fold + 1}")
                self.__verbose("")

                for name, model in self.__estimators:
                    self.__verbose(f"- {name}")

                    self.__evaluate_model(
                        model,
                        name,
                        X_train,
                        X_validation,
                        y_train,
                        y_validation,
                    )

                    self.fold.append(fold + 1)

        SummaryClass.__init__(
            self,
            frame={
                "Model": self.estimator_name_list,
                "AUC ROC": self.auc_roc,
                "Accuracy": self.accuracy,
                "Precision": self.precision,
                "Recall": self.recall,
                "F1-Score": self.f1score,
                "Fold": 1 if CV is None else self.fold,
            }, estimator_type = "classification"
        )
