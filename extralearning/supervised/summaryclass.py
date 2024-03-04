import pandas as pd


class SummaryClass:

    def __init__(self, frame, estimator_type):
        """
        A class dedicated to summarizing the performance of supervised learning models.

        This class provides users with access to various methods for evaluating and observing the model's performance.
        It is intended to be initialized within each supervised learning set.
        """

        assert isinstance(frame, dict), TypeError("frame should be dict type")
        assert estimator_type in ["classification", "regression"], ValueError(
            "estimator_type needs to be classification or regression for supervised learning"
        )

        self.DataFrameSummary = pd.DataFrame(frame)
        self.estimator = estimator_type
        self.default_metric = "MSE" if self.estimator == "regression" else "Accuracy"

    def summary(self, pandas=True):
        """
        Generate a summary of the data stored in the object.

        Parameters:
        - pandas (bool, optional): If True, returns the summary as a Pandas DataFrame.
        If False, returns the summary as a NumPy array. Default is True.

        Returns:
        - pandas.DataFrame or numpy.ndarray: Summary of the data. If pandas is True,
        the summary is returned as a Pandas DataFrame; otherwise, it is returned
        as a NumPy array.

        Note:
        - Can only be used when model trainned, if model has not been fit class should not be acessed or called.

        Examples
        --------
        >>> from extralearning import Classification

        >>> model = Classification()
        >>> model.fit_train(X, y)

        >>> model.summary()
        """

        if pandas:
            return self.DataFrameSummary

        return self.DataFrameSummary.to_numpy()

    def fold_summary(self):
        """
        Calculate the mean summary of data grouped by 'Fold' and 'Model'.

        Returns:
        - pandas.DataFrame: Mean summary of the data grouped by 'Fold' and 'Model'.

        Examples
        --------
        >>> from extralearning import Classification

        >>> model = Classification()
        >>> model.fit_train(X, y)

        >>> model.fold_summary()
        """

        return self.DataFrameSummary.groupby(["Fold", "Model"]).mean()

    def best(self, metric=None, pandas=True):
        """
        Retrieve the best-performing data entry based on the specified metric.

        Parameters:
        - metric (str, optional): The metric by which to determine the best-performing entry.
        If not provided, the default metric specified in the object will be used.
        - pandas (bool, optional): If True, returns the result as a Pandas DataFrame with the
        single best entry. If False, returns the result as a NumPy array. Default is True.

        Returns:
        - pandas.DataFrame or numpy.ndarray: The best-performing data entry based on the specified
        metric, either as a Pandas DataFrame or a NumPy array.

        Examples
        --------
        >>> from extralearning import Classification

        >>> model = Classification()
        >>> model.fit_train(X, y)

        >>> model.best()
        """

        if pandas:
            return (
                self.DataFrameSummary.groupby("Model")
                .mean()
                .sort_values(
                    by=self.default_metric if metric is None else metric,
                    ascending=False,
                )
                .head(1)
            )
        return (
            self.DataFrameSummary.groupby("Model")
            .mean()
            .sort_values(
                by=self.default_metric if metric is None else metric, ascending=False
            )
            .bottom(1)
            .to_numpy()
        )

    def top(self, n=5, metric=None, pandas=True):
        """
        Retrieve the top N data entries based on the specified metric.

        Parameters:
        - n (int, optional): The number of top entries to retrieve. Default is 5.
        - metric (str, optional): The metric by which to determine the top entries.
        If not provided, the default metric specified in the object will be used.
        - pandas (bool, optional): If True, returns the result as a Pandas DataFrame with
        the top N entries. If False, returns the result as a NumPy array. Default is True.

        Returns:
        - pandas.DataFrame or numpy.ndarray: The top N data entries based on the specified
        metric, either as a Pandas DataFrame or a NumPy array.

        Examples
        --------
        >>> from extralearning import Classification

        >>> model = Classification()
        >>> model.fit_train(X, y)

        >>> model.top(n=5)
        """

        if pandas:
            return self.DataFrameSummary.sort_values(
                by=self.default_metric if metric is None else metric, ascending=False
            ).head(n)
        return (
            self.DataFrameSummary.sort_values(
                by=self.default_metric if metric is None else metric, ascending=False
            )
            .head(n)
            .to_numpy()
        )

    def bottom(self, n=5, metric=None, pandas=True):
        """
        Retrieve the bottom N data entries based on the specified metric.

        Parameters:
        - n (int, optional): The number of bottom entries to retrieve. Default is 5.
        - metric (str, optional): The metric by which to determine the bottom entries.
        If not provided, the default metric specified in the object will be used.
        - pandas (bool, optional): If True, returns the result as a Pandas DataFrame with
        the bottom N entries. If False, returns the result as a NumPy array. Default is True.

        Returns:
        - pandas.DataFrame or numpy.ndarray: The bottom N data entries based on the specified
        metric, either as a Pandas DataFrame or a NumPy array.

        Examples
        --------
        >>> from extralearning import Classification

        >>> model = Classification()
        >>> model.fit_train(X, y)

        >>> model.bottom(n=5)
        """

        if pandas:
            return self.DataFrameSummary.sort_values(
                by=self.default_metric if metric is None else metric, ascending=False
            ).head(n)
        return (
            self.DataFrameSummary.sort_values(
                by=self.default_metric if metric is None else metric, ascending=False
            )
            .bottom(n)
            .to_numpy()
        )

    def mean(self, metric=None, pandas=True):
        """
        Calculate the mean of data grouped by the specified metric.

        Parameters:
        - metric (str, optional): The metric by which to group the data and calculate the mean.
        If not provided, the default metric specified in the object will be used.
        - pandas (bool, optional): If True, returns the mean as a Pandas DataFrame.
        If False, returns the mean as a NumPy array. Default is True.

        Returns:
        - pandas.DataFrame or numpy.ndarray: The mean of data grouped by the specified metric,
        either as a Pandas DataFrame or a NumPy array.

        Examples
        --------
        >>> from extralearning import Classification

        >>> model = Classification()
        >>> model.fit_train(X, y)

        >>> model.mean()
        """
        if pandas:
            return self.DataFrameSummary.groupby("Model").mean()
        return self.DataFrameSummary.groupby("Model").mean().to_numpy()

    def median(self, metric=None, pandas=True):
        """
        Calculate the median of data grouped by the specified metric.

        Parameters:
        - metric (str, optional): The metric by which to group the data and calculate the median.
        If not provided, the default metric specified in the object will be used.
        - pandas (bool, optional): If True, returns the median as a Pandas DataFrame.
        If False, returns the median as a NumPy array. Default is True.

        Returns:
        - pandas.DataFrame or numpy.ndarray: The median of data grouped by the specified metric,
        either as a Pandas DataFrame or a NumPy array.

        Examples
        --------
        >>> from extralearning import Classification

        >>> model = Classification()
        >>> model.fit_train(X, y)

        >>> model.median()
        """

        if pandas:
            return self.DataFrameSummary.groupby("Model").median()
        return self.DataFrameSummary.groupby("Model").median().to_numpy()
