from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from Data_Preprocessing import load_data
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise
import numpy as np


class UserNeighborhoodRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, similarity=pairwise.cosine_similarity):
        self.similarity = similarity
        pass

    def fit(self, X, y=None):
        self.users_dict_ = {}
        self.item_dict_ = {}
        user_counter = 0
        item_counter = 0
        for i in range(0, np.size(X, 0)):
            user = X[i,0]
            item = X[i,1]
            if (self.users_dict_.get(user, -1) == -1):
                self.users_dict_[user] = user_counter
                user_counter += 1
            if (self.item_dict_.get(item, -1) == -1):
                self.item_dict_[item] = item_counter
                item_counter += 1

        self.user_item_matrix_ = np.zeros([user_counter, item_counter])
        self.temp = np.zeros([user_counter, item_counter])
        for i in range(0, np.size(X, 0)):
            user = self.users_dict_.get(X[i,0])
            item = self.item_dict_.get(X[i,1])
            rate = y[i]
            self.user_item_matrix_[user, item] = rate
            self.temp[user, item] = 1



        # normalize the data
        self.predictions_ = self.user_item_matrix_*0
        self.means_ = np.sum(self.user_item_matrix_,1)
        self.means_ = self.means_.reshape(-1,1)
        self.means_ = self.means_/(np.sum(self.temp,1).reshape(-1,1))
        self.user_item_matrix_ = self.user_item_matrix_ - self.means_
        self.user_item_matrix_ = self.user_item_matrix_ * self.temp
        self.user_item_matrix_sizes_ = self.user_item_matrix_*self.user_item_matrix_;
        # print("start similarities")
        # for i in range(0, np.size(X, 0)):
        #     if (i % 100 == 0):
        #         print(i)
        #     target_user = self.user_item_matrix_[i,:].reshape(1,-1)
        #     target_user_size = self.user_item_matrix_sizes_[i,:].reshape(1,-1)
        #     target_ratings = self.temp[i,:].reshape(1,-1)
        #     second_norms = np.sum(self.user_item_matrix_sizes_*target_ratings,1)
        #     second_norms = np.sqrt(second_norms)
        #     second_norms = second_norms.reshape(-1,1)
        #     first_norms = np.dot(self.user_item_matrix_sizes_,np.transpose(target_user_size))
        #     first_norms = np.sqrt(first_norms)
        #     similarities = np.dot(self.user_item_matrix_, np.transpose(target_user))
        #     first_norms[first_norms==0] = 1
        #     second_norms[second_norms == 0] = 1
        #
        #     similarities = similarities / first_norms
        #     similarities = similarities / second_norms
        #     similarities[similarities < 0] = 0
        #
        #     weights = similarities*self.temp
        #     weights_sum = np.sum(weights,axis=0)
        #     weights_sum[weights_sum == 0] = 1
        #     prediction = np.sum(self.user_item_matrix_ * weights, axis=0)/weights_sum
        #     self.predictions_[i, :] = prediction
        #
        #
        # print("end similarities")


        # self.similarity_matrix_ = self.similarity(self.user_item_matrix_, self.user_item_matrix_)




    def predict(self,X,y=None):
        try:
            getattr(self, "user_item_matrix_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        self.predictions_ = self.user_item_matrix_ * 0
        pred = np.zeros([np.size(X, 0), 1])
        for i in range(0, np.size(X, 0)):
            user = self.users_dict_.get(X[i, 0], -1)
            item = self.item_dict_.get(X[i,1], -1)
            if (user == -1 or item == -1):
                pred[i] = 0
            else:
                if (self.predictions_[user, item] != 0):
                    pred[i] = self.predictions_[user, item] + self.means_[user]
                else:
                    target_user = self.user_item_matrix_[user, :].reshape(1, -1)
                    target_user_size = self.user_item_matrix_sizes_[user, :].reshape(1, -1)
                    target_ratings = self.temp[user, :].reshape(1, -1)
                    second_norms = np.sum(self.user_item_matrix_sizes_ * target_ratings, 1)
                    second_norms = np.sqrt(second_norms)
                    second_norms = second_norms.reshape(-1, 1)
                    first_norms = np.dot(self.user_item_matrix_sizes_, np.transpose(target_user_size))
                    first_norms = np.sqrt(first_norms)
                    similarities = np.dot(self.user_item_matrix_, np.transpose(target_user))
                    first_norms[first_norms == 0] = 1
                    second_norms[second_norms == 0] = 1

                    similarities = similarities / first_norms
                    similarities = similarities / second_norms
                    similarities[similarities < 0] = 0

                    weights = similarities * self.temp
                    weights_sum = np.sum(weights, axis=0)
                    weights_sum[weights_sum == 0] = 1
                    prediction = np.sum(self.user_item_matrix_ * weights, axis=0) / weights_sum
                    self.predictions_[user, :] = prediction
                    pred[i] = self.predictions_[user, item] + self.means_[user]

        return pred

    # an example of evaluation score
    def score(self, X, y, sample_weight=None):
        return (np.sum(self.predict(X) == y)) / np.size(y)


# performing kfold for user average regressor
number_of_folds = 10
(X, y) = load_data()
print(X.shape)
kfold = KFold(10000, True, random_state=123)
total_err = 0
# regressor = UserNeighborhoodRegressor()
# regressor.fit(X, y)



counter =0
for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    regressor = UserNeighborhoodRegressor()
    regressor.fit(X_train, y_train)
    f_test = regressor.predict(X_test)
    err = mean_absolute_error(y_test, f_test)
    total_err += err
    counter += 1
    print(y_test)
    print(f_test)
    if (counter == number_of_folds):
        break

    print("one round completed")

total_err /= number_of_folds
print('total MAE of user average predictor version 1 (without normalization)')
print(total_err)
