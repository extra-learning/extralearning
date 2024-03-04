import pandas as pd


class SummaryClass:

    def __init__(self, frame, estimator_type):

        assert isinstance(frame, dict), TypeError("frame should be dict type")
        assert estimator_type in ["classification", "regression"], ValueError("estimator_type needs to be classification or regression for supervised learning")
        self.DataFrameSummary = pd.DataFrame(frame)
        self.estimator = estimator_type
        self.default_metric = ("MSE" if self.estimator == "regression" else "Accuracy")

    def summary(self, pandas=True):
        if pandas:
            return self.DataFrameSummary

        return self.DataFrameSummary.to_numpy()
    
    def fold_summary(self):
        return self.DataFrameSummary.groupby(["Fold","Model"]).mean()

    def best(self, metric=None, pandas=True):
        if pandas:
            return self.DataFrameSummary.sort_values(by=self.default_metric if metric is None else metric, ascending=False).head(1)
        return (
            self.DataFrameSummary.sort_values(by=self.default_metric if metric is None else metric, ascending=False)
            .bottom(1)
            .to_numpy()
        )

    def top(self, n=5, metric=None, pandas=True):

        if pandas:
            return self.DataFrameSummary.sort_values(by=self.default_metric if metric is None else metric, ascending=False).head(n)
        return (
            self.DataFrameSummary.sort_values(by=self.default_metric if metric is None else metric, ascending=False)
            .head(n)
            .to_numpy()
        )

    def bottom(self, n=5, metric=None, pandas=True):
    
        if pandas:
            return self.DataFrameSummary.sort_values(by=self.default_metric if metric is None else metric, ascending=False).head(n)
        return (
            self.DataFrameSummary.sort_values(by=self.default_metric if metric is None else metric, ascending=False)
            .bottom(n)
            .to_numpy()
        )

    def mean(self, metric=None, pandas=True):
        if pandas:
            return (self.DataFrameSummary.groupby(
                self.default_metric
                if metric is None
                else metric).mean())
        return self.DataFrameSummary.groupby(self.default_metric if metric is None else metric).mean().to_numpy()

    def median(self, metric=None, pandas=True):
        if pandas:
            return self.DataFrameSummary.groupby(self.default_metric if metric is None else metric).median()
        return self.DataFrameSummary.groupby(self.default_metric if metric is None else metric).median().to_numpy()
