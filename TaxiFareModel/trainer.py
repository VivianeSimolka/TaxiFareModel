from matplotlib.pyplot import get
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
#from sklearn.model_selection import train_test_split

from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
#from TaxiFareModel.data import get_data, clean_data

class Trainer:

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train


    def set_pipeline(self):
     '''returns a pipelined model'''
     dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
     time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
     preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
     self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
     return self.pipeline

    def run(self):
        '''returns a trained pipelined model'''
        self.pipeline = self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        '''returns the value of the RMSE'''
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse


#    def get_clean_data(self, nrows=10000):
#        data = get_data(nrows)
#        self.data = clean_data(data)
#        return self.data

#    def holdout_method(self, test_size=0.2):
#        y = self.data["fare_amount"]
#        X = self.data.drop(columns="fare_amount")
#        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)
#        return self.X_train, self.X_test, self.y_train, self.y_test
