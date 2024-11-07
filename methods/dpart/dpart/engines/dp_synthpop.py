from dpart.dpart import dpart
from dpart.methods import LogisticRegression, LinearRegression
from dpart.methods import DecisionTreeClassifier


class DPsynthpop(dpart):
    # default_numerical = LinearRegression
    # default_categorical = LogisticRegression
    default_numerical = DecisionTreeClassifier
    default_categorical = DecisionTreeClassifier

    def __init__(self,
                 epsilon: dict = None,
                 methods: dict = None,
                 visit_order: list = None,
                 bounds: dict = None):
        super().__init__(methods=methods, epsilon=epsilon, visit_order=visit_order, bounds=bounds)
