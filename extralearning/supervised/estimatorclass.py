class EstimatorClass:
    def __init__(self, estimators) -> None:
        """
        Base EstimatorClass for supervised learning.

        Allows the user to have the same estimator interaction with both Regression and Classification classes.

        Note: Should not be accessed by the user since its a parent class.
        """

        self.__estimators = estimators

    def get_estimators(self) -> list:
        """
        Returns a list of tuples containing the current estimators with format (name, estimator).

        Examples
        --------
        >>> from extralearning import Classification
        >>> from sklearn.ensemble import RandomForestClassifier

        >>> model = Classification()

        >>> model.get_estimators()[0]
        ("Random Forest", RandomForestClassifier()))
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

        Note 1: To check the default estimators, you can use the `.get_estimators()` method.
        Note 2: If you want to pass parameters to an existing estimator you can use `.pass_params()`
        """

        assert isinstance(estimator, tuple), TypeError(
            "`estimator` takes a tuple as argument with format (name, estimator)"
        )

        self.__estimators.append(estimator)

    def remove_estimator(self, estimator) -> None:
        """
        Removes an estimator based on name or index position from the list to `.fit_train()`.

        Parameters
        ----------
        estimator: str or int,
            - str: name of estimator listed in `.get_estimators()` to be removed.
            - int: Index position of estimator listed in `.get_estimators()` to be removed.

        Examples
        --------
        >>> from extralearning import Classification
        >>> from sklearn.ensemble import RandomForestClassifier

        >>> model = Classification()

        >>> model.remove_estimator(estimator = "Random Forest")
        >>> model.remove_estimator(estimator = 5)

        """
        assert isinstance(estimator, (str, int)), TypeError(
            f"estimator must be int or str, not {type(estimator)}"
        )

        if isinstance(estimator, str):
            self.__estimators.pop(
                [model[0] for model in self.__estimators].index(estimator)
            )

        else:
            self.__estimators.pop(estimator)

    def get_params(self) -> list:
        """
        Returns a list of tuples containing the estimator and their default parameters

        Examples
        --------
        >>> from extralearning import Classification
        >>> from sklearn.ensemble import RandomForestClassifier

        >>> model = Classification()

        >>> model.get_params()[0]
        (LogisticRegression(),
        {'C': 1.0,
        'class_weight': None,
        'dual': False,
        'fit_intercept': True,
        'intercept_scaling': 1,
        'l1_ratio': None,
        'max_iter': 100,
        'multi_class': 'auto',
        'n_jobs': None,
        'penalty': 'l2',
        'random_state': None,
        'solver': 'lbfgs',
        'tol': 0.0001,
        'verbose': 0,
        'warm_start': False})
        """
        return [
            (estimator[1], estimator[1].get_params()) for estimator in self.__estimators
        ]

    def pass_params(self, estimator, overwrite=True) -> None:
        """
        Pass custom parameters to an estimator listed in `.get_estimators()`

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
            self.add_estimator(
                (str(type(estimator[0]())), estimator[0](**estimator[1]))
            )

        else:
            if overwrite and type(estimator[0]()) not in [
                type(model[1]) for model in self.__estimators
            ]:
                raise ValueError(
                    f"{estimator[0]()} not in estimator list, if you want to add a new estimator you should use .add_estimator() or set overwrite to False"
                )

            __ReplacedEstimator = [type(model[1]) for model in self.__estimators].index(
                type(estimator[0]())
            )

            self.__estimators[__ReplacedEstimator] = (
                self.__estimators[__ReplacedEstimator][0],
                estimator[0](**estimator[1]),
            )