from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from Data_Preprocessing import load_data
import numpy as np


class RandomRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, seed=None):
        # Parameters should have same name as attributes
        self.seed = seed

    def fit(self, X, y=None):
        self.min_ = np.min(y)
        self.max_ = np.max(y)

    def predict(self,X,y=None):
        try:
            getattr(self, "min_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        np.random.seed(self.seed)
        np.random.rand()
        y = np.random.randint(self.min_, self.max_ + 1, [np.size(X, 0), 1])
        return y

    # an example of evaluation score
    def score(self, X, y, sample_weight=None):
        return (np.sum(self.predict(X) == y)) / np.size(y)

number_of_folds = 10
(X, y) = load_data()
kfold = KFold(10000, True, random_state=123)
total_err = 0
counter = 0
for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    regressor = RandomRegressor(123)
    regressor.fit(X_train, y_train)
    f_test = regressor.predict(X_test)
    err = mean_absolute_error(y_test, f_test)
    total_err += err
    counter += 1
    if (counter == number_of_folds):
        break
    

total_err /= number_of_folds
print(total_err)
