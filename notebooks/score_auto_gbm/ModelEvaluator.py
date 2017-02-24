import cPickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from score_auto_gbm.FeatureTransformer import FeatureTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from datetime import datetime

# Class that return statistics about models
class ModelEvaluator(object):
    def __init__(self):
        pass

    def get_predict_vs_actuals(self, models, features, actuals):
        pass
    
    def print_rmse_summary(self, predction_tuple_array):
        pass