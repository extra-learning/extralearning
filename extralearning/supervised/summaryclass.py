import pandas as pd


class SummaryClass:

    def __init__(self, frame, estimator_type):

        # TODO VALIDATIONS

        self.DataFrameSummary = pd.DataFrame(frame)
        self.estimator = estimator_type

    def summary(self, pandas=True):
        if pandas:
            return self.DataFrameSummary

        return self.DataFrameSummary.to_numpy()

    def top(self, n=5, metric=None, pandas=True):

        if metric is None:
            metric = "MSE" if self.estimator == "regression" else "Accuracy"

        if pandas:
            return self.DataFrameSummary.sort_values(by=metric, ascending=False).head(n)
        return (
            self.DataFrameSummary.sort_values(by=metric, ascending=False)
            .head(n)
            .to_numpy()
        )

    def bottom(self, n=5, metric=None, pandas=True):
        if metric is None:
            metric = "MSE" if self.estimator == "regression" else "Accuracy"

        if pandas:
            return self.DataFrameSummary.sort_values(by=metric, ascending=False).head(n)
        return (
            self.DataFrameSummary.sort_values(by=metric, ascending=False)
            .bottom(n)
            .to_numpy()
        )

    def mean(self, metric=None, pandas=True):

        if metric is None:
            metric = "MSE" if self.estimator == "regression" else "Accuracy"

        if pandas:
            return self.DataFrameSummary.groupby(metric).mean()
        return self.DataFrameSummary.groupby(metric).mean().to_numpy()

    def median(self, metric=None, pandas=True):

        if metric is None:
            metric = "MSE" if self.estimator == "regression" else "Accuracy"

        if pandas:
            return self.DataFrameSummary.groupby(metric).median()
        return self.DataFrameSummary.groupby(metric).median().to_numpy()
