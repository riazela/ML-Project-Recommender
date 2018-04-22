from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from Data_Preprocessing import load_data
import numpy as np


class UserAverageRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.freq_ = {}
        self.user_ = {}
        for i in range(0, np.size(X, 0)):
            user = X[i,0]
            rate = y[i]
            self.user_[user] = self.user_.get(user, 0) + rate
            self.freq_[user] = self.freq_.get(user, 0) + 1
            

    def predict(self,X,y=None):
        try:
            getattr(self, "user_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        pred = np.zeros([np.size(X, 0), 1])
        for i in range(0, np.size(X, 0)):
            user = X[i, 0]
            rate = self.user_.get(user, 0) / self.freq_.get(user, 1)
            pred[i] = rate
        return pred

    # an example of evaluation score
    def score(self, X, y, sample_weight=None):
        return (np.sum(self.predict(X) == y)) / np.size(y)



class ItemAverageRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.freq_ = {}
        self.item_ = {}
        for i in range(0, np.size(X, 0)):
            item = X[i,1]
            rate = y[i]
            self.item_[item] = self.item_.get(item, 0) + rate
            self.freq_[item] = self.freq_.get(item, 0) + 1
            

    def predict(self,X,y=None):
        try:
            getattr(self, "item_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        pred = np.zeros([np.size(X, 0), 1])
        for i in range(0, np.size(X, 0)):
            item = X[i, 1]
            rate = self.item_.get(item, 0) / self.freq_.get(item, 1)
            pred[i] = rate
        return pred

    # an example of evaluation score
    def score(self, X, y, sample_weight=None):
        return (np.sum(self.predict(X) == y)) / np.size(y)


# performing kfold for user average regressor
number_of_folds = 10
(X, y) = load_data()
kfold = KFold(10000, True, random_state=123)
total_err = 0
counter = 0
for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    regressor = UserAverageRegressor()
    regressor.fit(X_train, y_train)
    f_test = regressor.predict(X_test)
    err = mean_absolute_error(y_test, f_test)
    total_err += err
    counter += 1
    if (counter == number_of_folds):
        break
    

total_err /= number_of_folds
print('total MAE of user average predictor')
print(total_err)




number_of_folds = 10
(X, y) = load_data()
kfold = KFold(10000, True, random_state=123)
total_err = 0
counter = 0
for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    regressor = ItemAverageRegressor()
    regressor.fit(X_train, y_train)
    f_test = regressor.predict(X_test)
    err = mean_absolute_error(y_test, f_test)
    total_err += err
    counter += 1
    if (counter == number_of_folds):
        break

total_err /= number_of_folds
print('total MAE of item average predictor')
print(total_err)


