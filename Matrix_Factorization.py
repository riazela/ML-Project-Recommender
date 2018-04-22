from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from Data_Preprocessing import load_data
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise
import numpy as np
from sklearn.decomposition import NMF

class MatrixFactorization(BaseEstimator, RegressorMixin):
    def __init__(self, d = 10, number_of_iteration = 0):
        self.d = d
        self.number_of_iteration = number_of_iteration

    def fit(self, X, y=None):
        self.users_dict_ = {}
        self.item_dict_ = {}
        user_counter = 0
        item_counter = 0
        for i in range(0, np.size(X, 0)):
            user = X[i, 0]
            item = X[i, 1]
            if (self.users_dict_.get(user, -1) == -1):
                self.users_dict_[user] = user_counter
                user_counter += 1
            if (self.item_dict_.get(item, -1) == -1):
                self.item_dict_[item] = item_counter
                item_counter += 1

        self.user_item_matrix_ = np.zeros([user_counter, item_counter])
        self.temp = np.zeros([user_counter, item_counter])
        for i in range(0, np.size(X, 0)):
            user = self.users_dict_.get(X[i, 0])
            item = self.item_dict_.get(X[i, 1])
            rate = y[i]
            self.user_item_matrix_[user, item] = rate
            self.temp[user, item] = 1

        # normalize the data


        self.means_ = np.sum(self.user_item_matrix_, 0)
        self.means_ = self.means_.reshape(1, -1)
        self.means_ = self.means_ / (np.sum(self.temp, 0).reshape(1, -1))
        self.user_item_matrix_ = self.user_item_matrix_ - self.means_
        self.user_item_matrix_ = self.user_item_matrix_ * self.temp
        self.user_item_matrix_ = self.user_item_matrix_ + self.means_
        self.user_item_matrix_ = self.user_item_matrix_ + 11


        #start Learning
        print("start_factorization")
        model = NMF(n_components=self.d, init='random', random_state=123)
        W = model.fit_transform(self.user_item_matrix_)
        H = model.components_
        self.prediction_matrix = np.dot(W,H)
        print("end_factorization")
        print("start_iterations")
        for i in range(0,self.number_of_iteration):
            A = self.user_item_matrix_ * self.temp
            A[A==0] = self.prediction_matrix[A==0]
            W = model.transform(A)
            self.prediction_matrix = np.dot(W, H)


    def predict(self, X, y=None):
        try:
            getattr(self, "user_item_matrix_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        pred = np.zeros([np.size(X, 0), 1])
        for i in range(0, np.size(X, 0)):
            user = self.users_dict_.get(X[i, 0], -1)
            item = self.item_dict_.get(X[i, 1], -1)
            if (user == -1 or item == -1):
                pred[i] = 0
            else:
                pred[i] = self.prediction_matrix[user, item] - 11

        return pred

    # an example of evaluation score
    def score(self, X, y, sample_weight=None):
        return (np.sum(self.predict(X) == y)) / np.size(y)


# performing kfold for user average regressor
number_of_folds = 5
(X, y) = load_data()
print(X.shape)

# regressor = UserNeighborhoodRegressor()
# regressor.fit(X, y)

for number_of_dims in range(16,20,1):
    kfold = KFold(1000, True, random_state=123)
    total_err = 0
    counter = 0
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regressor = MatrixFactorization(number_of_iteration=10, d=number_of_dims)
        regressor.fit(X_train, y_train)
        f_test = regressor.predict(X_test)
        err = mean_absolute_error(y_test, f_test)
        total_err += err
        counter += 1

        if (counter == number_of_folds):
            break

    total_err /= number_of_folds
    print('number of dims')
    print(number_of_dims)
    print("MAE")
    print(total_err)
