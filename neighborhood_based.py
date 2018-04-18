from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from Data_Preprocessing import load_data, 
from sklearn.metrics import pairwise
import numpy as np


class UserNeighborhoodRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, similarity=pairwise.cosine_similarity):
        self.similarity = similarity
        pass

    def fit(self, X, y=None):
        
            

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


