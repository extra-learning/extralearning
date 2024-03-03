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
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_squared_log_error,
    explained_variance_score,
)

# Sklearn Model Selection
from sklearn.model_selection import KFold, train_test_split

# Models
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    HuberRegressor,
    TheilSenRegressor,
    PassiveAggressiveRegressor,
    BayesianRidge,
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor


class Regression(EstimatorClass, SummaryClass):
    def __init__(self, random_state=None, n_jobs=None, ignore_warnings=True) -> None:
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
            ("Linear Regression", LinearRegression(n_jobs=self.n_jobs)),
            (
                "Ridge Regression",
                Ridge(random_state=self.random_state),
            ),
            ("Lasso Regression", Lasso(random_state=self.random_state)),
            ("ElasticNet Regression", ElasticNet(random_state=self.random_state)),
            ("Support Vector Regression", SVR()),
            (
                "Decision Tree Regressor",
                DecisionTreeRegressor(random_state=self.random_state),
            ),
            (
                "Random Forest Regressor",
                RandomForestRegressor(
                    n_jobs=self.n_jobs, random_state=self.random_state
                ),
            ),
            (
                "Gradient Boosting Regressor",
                GradientBoostingRegressor(random_state=random_state),
            ),
            ("K-KNeighbors Regressor", KNeighborsRegressor(n_jobs=self.n_jobs)),
            ("Huber Regressor", HuberRegressor()),
            (
                "Theil-Sen Regressor",
                TheilSenRegressor(n_jobs=self.n_jobs, random_state=self.random_state),
            ),
            (
                "Passive Aggressive Regressor",
                PassiveAggressiveRegressor(random_state=self.random_state),
            ),
            ("Bayesian Ridge", BayesianRidge()),
        ]

        EstimatorClass.__init__(self, estimators=self.__estimators)

        # Settings
        if ignore_warnings:
            warnings.filterwarnings("ignore")

    def __init_evaluation_arrays(self) -> None:
        self.MAE = list()
        self.MSE = list()
        self.RMSE = list()
        self.R2 = list()
        self.MSLE = list()
        self.variance = list()
        self.estimator_name_list = list()
        self.fold = list()

    def __evaluate_model(
        self, estimator, estimator_name, X_train, X_validation, y_train, y_validation
    ) -> None:
        # Loading the model
        model = estimator

        # Fitting the model
        model.fit(X_train, y_train)

        # Predicting
        prediction = model.predict(X_validation)

        # Recording metrics
        self.estimator_name_list.append(estimator_name)
        self.MAE.append(mean_absolute_error(y_validation, prediction))
        self.MSE.append(mean_squared_error(y_validation, prediction))
        self.RMSE.append(np.sqrt(mean_squared_error(y_validation, prediction)))
        self.R2.append(r2_score(y_validation, prediction))
        self.MSLE.append(mean_squared_log_error(y_validation, prediction))
        self.variance.append(explained_variance_score(y_validation, prediction))

    def __verbose(self, text: str, end="\n") -> None:
        if self.__verbose_status:
            print(text, end=end)

    def reset_estimators(self) -> None:
        self.__estimators = [
            ("Linear Regression", LinearRegression(n_jobs=self.n_jobs)),
            (
                "Ridge Regression",
                Ridge(random_state=self.random_state),
            ),
            ("Lasso Regression", Lasso(random_state=self.random_state)),
            ("ElasticNet Regression", ElasticNet(random_state=self.random_state)),
            ("Support Vector Regression", SVR()),
            (
                "Decision Tree Regressor",
                DecisionTreeRegressor(random_state=self.random_state),
            ),
            (
                "Random Forest Regressor",
                RandomForestRegressor(
                    n_jobs=self.n_jobs, random_state=self.random_state
                ),
            ),
            (
                "Gradient Boosting Regressor",
                GradientBoostingRegressor(random_state=self.random_state),
            ),
            ("K-KNeighbors Regressor", KNeighborsRegressor(n_jobs=self.n_jobs)),
            ("Huber Regressor", HuberRegressor()),
            (
                "Theil-Sen Regressor",
                TheilSenRegressor(n_jobs=self.n_jobs, random_state=self.random_state),
            ),
            (
                "Passive Aggressive Regressor",
                PassiveAggressiveRegressor(random_state=self.random_state),
            ),
            ("Bayesian Ridge", BayesianRidge()),
        ]

    def fit_train(self, X, y, CV=3, CV_params=None, verbose=True) -> None:
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
            CROSS_VALIDATION = (
                KFold(n_splits=CV, random_state=self.random_state, shuffle=True)
                if CV_params is None
                else KFold(**CV_params)
            )

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
                        model, name, X_train, X_validation, y_train, y_validation
                    )

                    self.fold.append(fold + 1)

        SummaryClass.__init__(
            self,
            frame={
                "Model": self.estimator_name_list,
                "MAE": self.MAE,
                "MSE": self.MSE,
                "RMSE": self.RMSE,
                "R2-Score": self.R2,
                "MSLE": self.MSLE,
                "Explained Variance": self.variance,
                "Fold": 1 if CV is None else self.fold,
            }, estimator_type = "regression"
        )
