from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.metrics import mean_absolute_error
import numpy as np


class TestRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, param1 = 0, param2 = 0):
        # Parameters should have same name as attributes
        self.param1 = param1
        self.param2 = param2

    def fit(self,X, y=None):
        # Here we check the value of the parameters
        assert(type(self.param1) == int), "param1 should be integer"
        assert((self.param1) >= 0), "param1 should be positive"

        #Here we fit the model
        print(self.param1)
        print(self.param2)
        #for example this can be trained value for all y's
        self.treshold_ = np.average(y)

    def predict(self,X,y=None):
        try:
            getattr(self, "treshold_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        y = np.ones([np.size(X,0),1])*self.treshold_
        return y

    # an example of evaluation score
    def score(self, X, y, sample_weight=None):
        return (np.sum(self.predict(X)==y))/np.size(y)




a = np.array([1,2,3,4,5])
b = np.array([1,2,3,4,5])
T = TestRegressor()
T.set_params(param1=2)
T.fit(a,b)
y = T.predict(a)
score = T.score(a,b)
print(y)
print(score)

print(mean_absolute_error(b,y))