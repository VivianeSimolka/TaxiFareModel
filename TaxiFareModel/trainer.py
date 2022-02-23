#from matplotlib.pyplot import get
from lib2to3.pgen2.token import GREATER
from memoized_property import memoized_property
import mlflow
from  mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
#from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, Lasso, PoissonRegressor
#from sklearn.svm import SVR
#from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data

MODELS = {"forest_model10": RandomForestRegressor(n_estimators=10),
                    "forest_model50": RandomForestRegressor(n_estimators=50),
                    "forest_model100": RandomForestRegressor(),
                    "forest_model1000": RandomForestRegressor(n_estimators=1000),
                    }

class Trainer:

    MLFLOW_URI = "https://mlflow.lewagon.co/"
    EXPERIMENT_NAME = "[DE] [Berlin] [VivianeSimolka] Taxi fare challenge"
    MODELS = {"forest_model": RandomForestRegressor(n_estimators=100,
                                                                                            criterion='squared_error',
                                                                                            max_depth=None,
                                                                                            min_samples_split=50,
                                                                                            min_samples_leaf=4,
                                                                                            min_weight_fraction_leaf=0.0,
                                                                                            max_features='auto',
                                                                                            max_leaf_nodes=None,
                                                                                            min_impurity_decrease=0.0,
                                                                                            bootstrap=True,
                                                                                            oob_score=False,
                                                                                            n_jobs=None,
                                                                                            random_state=None,
                                                                                            verbose=0,
                                                                                            warm_start=False,
                                                                                            ccp_alpha=0.0,
                                                                                            max_samples=None,)
                       }

    def __init__(self, X_train, y_train, model="forest_model"):
        self.X_train = X_train
        self.y_train = y_train
        self.model_name = model
        self.model = self.MODELS[model]

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
            ('model', self.model)
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

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.EXPERIMENT_NAME)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.EXPERIMENT_NAME).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, f'{self.model_name}.joblib')



#    def get_clean_data(self, nrows=10000):
#        data = get_data(nrows)
#        self.data = clean_data(data)
#        return self.data

#    def holdout_method(self, test_size=0.2):
#        y = self.data["fare_amount"]
#        X = self.data.drop(columns="fare_amount")
#        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)
#        return self.X_train, self.X_test, self.y_train, self.y_test

if __name__ == '__main__':
    #model = "tree_model"
    df = get_data()
    df = clean_data(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    loop = False
    grid_status = False

    trainer = Trainer(X_train, y_train)
    trainer.run()
    rmse = trainer.evaluate(X_test, y_test)
    trainer.mlflow_log_param("model", trainer.model_name)
    trainer.mlflow_log_metric("rmse", rmse)

    if grid_status == True:

        trainer = Trainer(X_train, y_train)
        trainer.set_pipeline()
        #print(trainer.pipeline.get_params())

        grid = {#'model__ccp_alpha': [0],
                # 'model__criterion': ['squared_error'],
                    #'model__max_depth': None,
                    #'model__max_features': 'auto',
                    #'model__max_leaf_nodes': None,
                    #'model__max_samples': None,
                    'model__min_samples_leaf': [3,4],
                    'model__min_samples_split': [25,30, 50],
                    }

        clf= GridSearchCV(trainer.pipeline, param_grid=grid, n_jobs=-1, cv=5) #, scoring="rmse")

        clf.fit(X_train, y_train)

        print(clf.best_params_)

    if loop == True:
        for model in MODELS.keys():
            #import ipdb; ipdb.set_trace
            trainer = Trainer(X_train, y_train, model=model)
            trainer.run()
            rmse = trainer.evaluate(X_test, y_test)
            #trainer.mlflow_run()
            trainer.mlflow_log_param("model", trainer.model_name)
            trainer.mlflow_log_metric("rmse", rmse)
